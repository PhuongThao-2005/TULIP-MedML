import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from util import *

tqdm.monitor_interval = 0
class Engine(object):
    """
    Base class cho training loop - dùng callback pattern.

    Thay vì viết vòng lặp train cứng, Engine chia thành các hook:
        on_start_epoch → on_start_batch → on_forward → on_end_batch → on_end_epoch

    Subclass chỉ cần override hook muốn thay đổi, không copy cả loop.
    Ví dụ: GCNMultiLabelMAPEngine chỉ override on_forward và on_start_batch.
    """

    def __init__(self, state={}):
        # state dict chứa toàn bộ config và trạng thái runtime
        self.state = state

        # Gán giá trị mặc định nếu chưa có trong state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()
        if self._state('image_size') is None:
            self.state['image_size'] = 224
        if self._state('batch_size') is None:
            self.state['batch_size'] = 64
        if self._state('workers') is None:
            self.state['workers'] = 4
        if self._state('device_ids') is None:
            self.state['device_ids'] = None       # None → dùng tất cả GPU có
        if self._state('evaluate') is None:
            self.state['evaluate'] = False         # True → chỉ validate, không train
        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0          # > 0 khi resume từ checkpoint
        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90
        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []          # các epoch sẽ giảm lr × 0.1

        # Meter đo loss trung bình
        self.state['meter_loss'] = _AverageMeter()
        self.state['batch_time'] = _AverageMeter()  # thời gian xử lý 1 batch
        self.state['data_time']  = _AverageMeter()  # thời gian load data

        if self._state('use_pb') is None:
            self.state['use_pb'] = True            # hiện tqdm progress bar
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0           # 0 → không in log từng batch

    def _state(self, name):
        # Truy cập state an toàn - trả None nếu key chưa tồn tại
        if name in self.state:
            return self.state[name]

    # ── Callbacks — subclass override những gì cần ───────────────────

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Reset meter đầu mỗi epoch để tính trung bình epoch mới
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].avg
        if display:
            if training:
                print('Epoch: [{0}]\tLoss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \tLoss {loss:.4f}'.format(loss=loss))
        return loss   # score để so sánh với best - subclass override trả mAP

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Base không làm gì — subclass override để unpack input tuple
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Ghi loss batch vào meter
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        # In log theo tần suất print_freq
        if display and self.state['print_freq'] != 0 \
                and self.state['iteration'] % self.state['print_freq'] == 0:
            loss       = self.state['meter_loss'].avg
            batch_time = self.state['batch_time'].avg
            data_time  = self.state['data_time'].avg
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    data_time_current=self.state['data_time_batch'],
                    data_time=data_time,
                    loss_current=self.state['loss_batch'],
                    loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    data_time_current=self.state['data_time_batch'],
                    data_time=data_time,
                    loss_current=self.state['loss_batch'],
                    loss=loss))

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        # Forward pass đơn giản — GCNMultiLabelMAPEngine override cái này
        input_var  = self.state['input']
        target_var = self.state['target']

        self.state['output'] = model(input_var)
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    # ── Setup transforms ──────────────────────────────────────────────

    def init_learning(self, model, criterion):
        """
        Tạo image transforms nếu chưa có.
        Gọi mean/std từ model thay vì hardcode → mỗi backbone có thể dùng
        chuẩn hóa khác nhau (ImageNet, BiomedCLIP, etc.)
        """
        if self._state('train_transform') is None:
            normalize = transforms.Normalize(
                mean=model.image_normalization_mean,
                std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                # Crop nhiều scale khác nhau → model học feature đa tỷ lệ
                MultiScaleCrop(self.state['image_size'],
                               scales=(1.0, 0.875, 0.75, 0.66, 0.5),
                               max_distort=2),
                transforms.RandomHorizontalFlip(),  # augmentation cơ bản
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(
                mean=model.image_normalization_mean,
                std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),     # chỉ resize, không crop random
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    # ── Main training loop ────────────────────────────────────────────

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
        self.init_learning(model, criterion)

        # Set transform vào dataset sau khi biết image_size
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform   = self.state['val_transform']
        val_dataset.target_transform   = self._state('val_target_transform')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.state['batch_size'],
            shuffle=True,
            num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.state['batch_size'],
            shuffle=False,
            num_workers=self.state['workers'])

        # Resume từ checkpoint nếu có
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score']  = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            # DataParallel tự phân chia batch qua nhiều GPU nếu có
            model = torch.nn.DataParallel(
                model, device_ids=self.state['device_ids']).cuda()

        if self.state['evaluate']:
            # Chỉ chạy validate một lần rồi thoát
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            self.train(train_loader, model, criterion, optimizer, epoch)
            score = self.validate(val_loader, model, criterion)

            # Lưu checkpoint mỗi epoch, đánh dấu is_best nếu score tốt nhất
            is_best = score > self.state['best_score']
            self.state['best_score'] = max(score, self.state['best_score'])
            self.save_checkpoint({
                'epoch'     : epoch + 1,
                'arch'      : self._state('arch'),
                'state_dict': model.module.state_dict()
                               if hasattr(model, 'module') else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)
            print('best score: ', self.state['best_score'])

        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()   # bật BatchNorm training mode, bật Dropout
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Train')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration']      = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input']  = input
            self.state['target'] = target

            # Subclass có thể unpack input tuple, remap labels ở đây
            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        model.eval()    # tắt BatchNorm training mode, tắt Dropout
        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        with torch.no_grad():
            # no_grad: không allocate bộ nhớ cho gradient
            for i, (input, target) in enumerate(data_loader):
                self.state['iteration']      = i
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])

                self.state['input']  = input
                self.state['target'] = target

                self.on_start_batch(False, model, criterion, data_loader)

                if self.state['use_gpu']:
                    self.state['target'] = self.state['target'].cuda(non_blocking=True)

                self.on_forward(False, model, criterion, data_loader)

                self.state['batch_time_current'] = time.time() - end
                self.state['batch_time'].add(self.state['batch_time_current'])
                end = time.time()

                self.on_end_batch(False, model, criterion, data_loader)

        return self.on_end_epoch(False, model, criterion, data_loader)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        # Lưu checkpoint mỗi epoch → dùng để resume nếu session hết giờ
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename  = os.path.join(self.state['save_model_path'], filename_)
            os.makedirs(self.state['save_model_path'], exist_ok=True)

        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)

        if is_best:
            # Lưu thêm bản tốt nhất với tên cố định
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(
                    self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)

            if self._state('save_model_path') is not None:
                # Xóa bản best cũ, lưu bản best mới kèm score trong tên file
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(
                    self.state['save_model_path'],
                    'model_best_{score:.4f}.pth.tar'.format(
                        score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Giảm lr x 0.1 tại các epoch trong epoch_step."""
        lr_list = []
        # Kiểm tra epoch hiện tại có trong danh sách epoch_step không
        decay = 0.1 if sum(
            self.state['epoch'] == np.array(self.state['epoch_step'])
        ) > 0 else 1.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)


class MultiLabelMAPEngine(Engine):
    """
    Extends Engine với mAP metric cho multi-label classification.
    Override on_end_epoch để tính và in mAP thay vì chỉ loss.
    """

    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        # ap_meter gom predictions toàn epoch rồi tính mAP 1 lần
        # chính xác hơn tính average của batch-level AP
        self.state['ap_meter'] = AveragePrecisionMeter(
            self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()   # xóa predictions epoch cũ

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Tính mAP từ tất cả predictions đã gom trong epoch
        map  = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].avg

        # OP/OR/OF1: Overall Precision/Recall/F1 — tính trên toàn bộ predictions
        # CP/CR/CF1: Per-Class mean — tính trung bình qua từng class
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()

        # Top-3: chỉ xét top 3 predictions có confidence cao nhất mỗi ảnh
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = \
            self.state['ap_meter'].overall_topk(3)

        if display:
            if training:
                print('Epoch: [{0}]\tLoss {loss:.4f}\tmAP {map:.3f}'.format(
                    self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\tOR: {OR:.4f}\tOF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\tCR: {CR:.4f}\tCF1: {CF1:.4f}'.format(
                    OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \tLoss {loss:.4f}\tmAP {map:.3f}'.format(
                    loss=loss, map=map))
                print('OP: {OP:.4f}\tOR: {OR:.4f}\tOF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\tCR: {CR:.4f}\tCF1: {CF1:.4f}'.format(
                    OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\tOR_3: {OR:.4f}\tOF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\tCR_3: {CR:.4f}\tCF1_3: {CF1:.4f}'.format(
                    OP=OP_k, OR=OR_k, OF1=OF1_k,
                    CP=CP_k, CR=CR_k, CF1=CF1_k))
        return map   # trả mAP để learning() so sánh với best_score

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Lưu ground truth gốc {-1, 0, 1} để tính mAP đúng
        self.state['target_gt'] = self.state['target'].clone()

        # Remap labels cho loss function (BCE cần {0, 1}):
        #   0  → 1  (unknown/uncertain → coi là positive khi tính loss)
        #   -1 → 0  (negative → 0)
        # NOTE: target_gt giữ nguyên {-1, 0, 1} cho ap_meter
        self.state['target'][self.state['target'] == 0]  = 1
        self.state['target'][self.state['target'] == -1] = 0

        # Unpack input tuple từ dataset.__getitem__:
        #   input[0] = img tensor  (B, 3, H, W)
        #   input[1] = filename    string
        # (MultiLabel không dùng word_vec, chỉ GCN mới dùng)
        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name']  = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        # Gom output và ground truth vào ap_meter để tính mAP cuối epoch
        self.state['ap_meter'].add(
            self.state['output'].data,
            self.state['target_gt'])   # dùng target_gt, không phải target đã remap

        if display and self.state['print_freq'] != 0 \
                and self.state['iteration'] % self.state['print_freq'] == 0:
            loss       = self.state['meter_loss'].avg
            batch_time = self.state['batch_time'].avg
            data_time  = self.state['data_time'].avg
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'],
                    len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    data_time_current=self.state['data_time_batch'],
                    data_time=data_time,
                    loss_current=self.state['loss_batch'],
                    loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time,
                    data_time_current=self.state['data_time_batch'],
                    data_time=data_time,
                    loss_current=self.state['loss_batch'],
                    loss=loss))


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    """
    Extends MultiLabelMAPEngine cho GCNResnet.

    2 thay đổi so với MultiLabelMAPEngine:
    1. on_forward: model cần 2 inputs (ảnh + word_vec) thay vì 1
    2. on_start_batch: unpack thêm input[2] = word_vec
    """

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        feature_var = self.state['feature'].float()  # ảnh (B, 3, H, W)
        target_var  = self.state['target'].float()   # labels (B, 14)
        inp_var     = self.state['input'].float().detach()
        # .detach(): word embeddings không tham gia backprop
        # gradient chỉ chạy qua GCN weights W1, W2 — không chạy về word_vec input

        if self.state['use_gpu']:
            feature_var = feature_var.cuda()
            inp_var     = inp_var.cuda()
            # target đã được .cuda() trong train/validate loop

        # GCNResnet.forward(feature, inp) → logits (B, 14)
        self.state['output'] = model(feature_var, inp_var)
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            # Clip gradient: nếu norm > 10 thì scale xuống
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # Remap labels giống MultiLabelMAPEngine
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0]  = 1
        self.state['target'][self.state['target'] == -1] = 0

        # Unpack input tuple — GCN cần cả 3 phần:
        #   input[0] = img       → state['feature']  (B, 3, H, W)
        #   input[1] = filename  → state['out']       string, để debug
        #   input[2] = word_vec  → state['input']     (14, 300)
        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out']     = input[1]
        self.state['input']   = input[2]   # on_forward sẽ đọc từ đây


class _AverageMeter:
    """
    Tính giá trị trung bình chạy (running average).
    Thay thế torchnet.AverageValueMeter để bớt dependency.

    Dùng để track loss trung bình trong epoch:
        meter = _AverageMeter()
        for batch in loader:
            loss = compute(batch)
            meter.add(loss)
        print(meter.avg)   # loss trung bình cả epoch
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def add(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def value(self):
        # Tương thích API với torchnet: trả tuple (mean, std)
        # std = 0 vì không track — chỉ cần mean
        return (self.avg, 0.0)