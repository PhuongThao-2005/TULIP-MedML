import os
import shutil
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from src.util import Warp, MultiScaleCrop, AveragePrecisionMeter

from src.evaluate import (
    compute_AUC_uncertain,
    compute_mAP,
    compute_mean_AUC,
    print_metrics,
)


# ─── Running average ─────────────────────────────────────────────────────────

class _AverageMeter:
    """Running average — tracks loss across batches within one epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def add(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return (self.avg, 0.0)


# ─── Base Engine ─────────────────────────────────────────────────────────────

class Engine:
    """
    Base training loop using a callback/hook pattern.

    Subclasses override hooks rather than copy the full loop:
        on_start_epoch → on_start_batch → on_forward → on_end_batch → on_end_epoch

    GCNMultiLabelMAPEngine overrides on_start_batch (unpack tuple + remap
    labels) and on_forward (single-input GCN forward pass — inp is a model
    buffer, not a dataset field).
    """

    def __init__(self, state: dict = {}):
        self.state = state

        defaults = {
            'use_gpu'     : torch.cuda.is_available(),
            'image_size'  : 224,
            'batch_size'  : 64,
            'workers'     : 4,
            'device_ids'  : None,   # None → use all GPUs
            'evaluate'    : False,  # True → validate only, no training
            'start_epoch' : 0,
            'max_epochs'  : 90,
            'epoch_step'  : [],     # epochs at which to decay lr × 0.1
            'use_pb'      : True,
            'print_freq'  : 0,
            # 'loss_type' : 'bce'  ← caller sets this via state dict
        }
        for k, v in defaults.items():
            if self._state(k) is None:
                self.state[k] = v

        self.state['meter_loss'] = _AverageMeter()
        self.state['batch_time'] = _AverageMeter()
        self.state['data_time']  = _AverageMeter()

    def _state(self, name):
        return self.state.get(name)

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].avg
        if display:
            tag = 'Epoch: [{0}]'.format(self.state['epoch']) if training else 'Test:'
            print(f'{tag}\tLoss {loss:.4f}')
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 \
                and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].avg
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {btc:.3f} ({bt:.3f})\t'
                      'Data {dtc:.3f} ({dt:.3f})\t'
                      'Loss {lc:.4f} ({l:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    btc=self.state['batch_time_current'],
                    bt=self.state['batch_time'].avg,
                    dtc=self.state['data_time_batch'],
                    dt=self.state['data_time'].avg,
                    lc=self.state['loss_batch'], l=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {btc:.3f} ({bt:.3f})\t'
                      'Data {dtc:.3f} ({dt:.3f})\t'
                      'Loss {lc:.4f} ({l:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    btc=self.state['batch_time_current'],
                    bt=self.state['batch_time'].avg,
                    dtc=self.state['data_time_batch'],
                    dt=self.state['data_time'].avg,
                    lc=self.state['loss_batch'], l=loss))

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        input_var  = self.state['input']
        target_var = self.state['target']

        self.state['output'] = model(input_var)
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    # ── Transforms ───────────────────────────────────────────────────────────

    def init_learning(self, model, criterion):
        """
        Tạo image transforms nếu chưa có.
        Gọi mean/std từ model thay vì hardcode → mỗi backbone có thể dùng
        chuẩn hóa khác nhau (ImageNet, BiomedCLIP, etc.)
        """
        mean = getattr(model, 'image_normalization_mean', [0.485, 0.456, 0.406])
        std  = getattr(model, 'image_normalization_std',  [0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=mean, std=std)

        if self._state('train_transform') is None:
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'],
                               scales=(1.0, 0.875, 0.75, 0.66, 0.5),
                               max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────

    def learning(self, model, criterion, train_dataset, val_dataset, val_uncertain_dataset=None, optimizer=None):
        self.init_learning(model, criterion)

        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform   = self.state['val_transform']
        val_dataset.target_transform   = self._state('val_target_transform')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.state['batch_size'],
            shuffle=True,
            num_workers=self.state['workers'],
            pin_memory=self.state['use_gpu'],
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.state['batch_size'],
            shuffle=False,
            num_workers=self.state['workers'],
            pin_memory=self.state['use_gpu'],
        )
        
        val_unc_loader = None
        if val_uncertain_dataset is not None:
            val_uncertain_dataset.transform = self.state['val_transform']
            val_uncertain_dataset.target_transform = self._state('val_target_transform')

            val_unc_loader = torch.utils.data.DataLoader(
                val_uncertain_dataset,
                batch_size=self.state['batch_size'],
                shuffle=False,
                num_workers=self.state['workers'],
                pin_memory=self.state['use_gpu'],
            )

        # Resume từ checkpoint nếu có
        resume_path = self._state('resume')

        #  AUTO RESUME nếu không có resume
        if resume_path is None:
            resume_path = self.find_latest_checkpoint()

        if resume_path is not None and os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location='cpu')

            self.state['start_epoch'] = checkpoint['epoch'] + 1
            self.state['best_score']  = checkpoint['best_score']

            model.load_state_dict(checkpoint['state_dict'])

            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                
                if self.state['use_gpu']:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

            print(f"=> resumed from epoch {checkpoint['epoch']}")
        else:
            print("=> no checkpoint found, train from scratch")

        if self.state['use_gpu']:
            model = torch.nn.DataParallel(
                model, device_ids=self.state['device_ids']).cuda()
            
        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            self.train(train_loader, model, criterion, optimizer, epoch)
            print("\n=== Val (Official) ===")
            score = self.validate(val_loader, model, criterion)

            if val_unc_loader is not None:
                print("\n=== Val (Uncertain) ===")
                _ = self.validate(val_unc_loader, model, criterion)

            # Lưu checkpoint mỗi epoch, đánh dấu is_best nếu score tốt nhất
            is_best = score > self.state['best_score']
            self.state['best_score'] = max(score, self.state['best_score'])
            self.save_checkpoint({
                'epoch'     : epoch,
                'arch'      : self._state('arch'),
                'state_dict': model.module.state_dict()
                               if hasattr(model, 'module') else model.state_dict(),
                'best_score': self.state['best_score'],
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            print('best score:', self.state['best_score'])

        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Train')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration']       = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input']  = input
            self.state['target'] = target

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
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Val')

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(data_loader):
                self.state['iteration']       = i
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

    def save_checkpoint(self, state, is_best, filename=None):
        if filename is None:
            filename = f'checkpoint_epoch_{state["epoch"]}.pth.tar'
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename  = os.path.join(self.state['save_model_path'], filename_)
            os.makedirs(self.state['save_model_path'], exist_ok=True)
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)

        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'],
                                             filename_best)
            shutil.copyfile(filename, filename_best)

            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    try:
                        os.remove(self._state('filename_previous_best'))
                    except OSError:
                        pass
                filename_best = os.path.join(
                    self.state['save_model_path'],
                    'model_best_{score:.4f}.pth.tar'.format(
                        score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

        # giữ tối đa N checkpoint
        max_keep = self.state.get('max_keep_ckpt', 5)

        import glob
        files = glob.glob(os.path.join(self.state['save_model_path'], 'checkpoint_epoch_*.pth.tar'))

        if len(files) > max_keep:
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            to_delete = files[:-max_keep]

            for f in to_delete:
                try:
                    os.remove(f)
                    print(f"Removed old checkpoint: {f}")
                except:
                    pass

    def adjust_learning_rate(self, optimizer):
        """
        Decay lr × 0.1 tại mỗi epoch trong epoch_step.
        Tính từ base_lrs gốc thay vì *= để tránh lỗi khi resume.
        """
        # Lưu lr gốc lần đầu tiên gọi
        if 'base_lrs' not in self.state:
            self.state['base_lrs'] = [pg['lr'] for pg in optimizer.param_groups]

        n_decays = sum(
            1 for s in self.state['epoch_step']
            if s <= self.state['epoch']
        )
        factor = 0.1 ** n_decays

        lr_list = []
        for pg, base_lr in zip(optimizer.param_groups, self.state['base_lrs']):
            pg['lr'] = base_lr * factor
            lr_list.append(pg['lr'])
        return np.unique(lr_list)
    
    def find_latest_checkpoint(self):
        import glob

        if self._state('save_model_path') is None:
            return None

        pattern = os.path.join(self.state['save_model_path'], 'checkpoint_epoch_*.pth.tar')
        files = glob.glob(pattern)

        if not files:
            return None

        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return files[-1]


# ─── Multi-label mAP Engine ──────────────────────────────────────────────────

class MultiLabelMAPEngine(Engine):
    """
    Extends Engine với mAP metric cho multi-label classification.
    """

    def __init__(self, state: dict):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(
            self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader,
                              optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        map_val = 100.0 * self.state['ap_meter'].value().mean()
        loss    = self.state['meter_loss'].avg
        if display:
            tag = ('Epoch: [{0}]'.format(self.state['epoch'])
                   if training else 'Test:')
            print(f'{tag}\tLoss {loss:.4f}\tmAP {map_val:.3f}')
        return map_val

    def on_start_batch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name']  = input[1]

    def on_end_batch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        Engine.on_end_batch(self, training, model, criterion, data_loader,
                            display=False)
        self.state['ap_meter'].add(
            self.state['output'].data,
            self.state['target_gt'])

        if display and self.state['print_freq'] != 0 \
                and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].avg
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {lc:.4f} ({l:.4f})'.format(
                    self.state['epoch'], self.state['iteration'],
                    len(data_loader),
                    lc=self.state['loss_batch'], l=loss))
            else:
                print('Test: [{0}/{1}]\tLoss {lc:.4f} ({l:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    lc=self.state['loss_batch'], l=loss))


# ─── GCN Engine ──────────────────────────────────────────────────────────────

class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    """
    Extends MultiLabelMAPEngine cho GCNResnet.

    Thay đổi chính so với phiên bản cũ:
      1. inp KHÔNG còn trong dataset — model tự load qua register_buffer.
         on_start_batch chỉ unpack (img, path), không cần lấy inp từ input[2].
      2. on_start_batch có 2 nhánh remap label tùy loss_type:
         - 'bce'    : -1 → 0  (uncertain coi như negative cho loss)
         - 'ua_asl' : giữ nguyên -1 (loss tự xử lý)
      3. on_forward gọi model(feature) một argument — inp là buffer trong model.
      4. Checkpoint chọn theo mAP (khớp proposal), fallback mean_auc.
      5. adjust_learning_rate tính từ base_lrs gốc, không lỗi khi resume.
    """

    def on_start_epoch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        MultiLabelMAPEngine.on_start_epoch(
            self, training, model, criterion, data_loader, optimizer)
        if not training:
            self.state['_val_scores']  = []
            self.state['_val_targets'] = []

    def on_start_batch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        # Preserve original labels for evaluation (keeps -1 uncertain)
        self.state['target_gt'] = self.state['target'].clone()

        loss_type = self.state.get('loss_type', 'bce')

        if loss_type == 'bce':
            # BCE không hiểu -1 → remap uncertain thành negative
            target = self.state['target'].clone().float()
            target[target < 0] = 0.0
            self.state['target'] = target
        else:
            # ua_asl nhận -1 trực tiếp, chỉ cast float
            self.state['target'] = self.state['target'].float()

        # Unpack (img, path)
        input = self.state['input']
        self.state['feature'] = input[0]   # [B, 3, H, W]
        self.state['out']     = input[1]   # list of path strings

    def on_end_batch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        MultiLabelMAPEngine.on_end_batch(
            self, training, model, criterion, data_loader, display)

        if not training:
            probs = torch.sigmoid(self.state['output']).detach().cpu().numpy()
            # Dùng target_gt (giữ -1) để compute_AUC_uncertain hoạt động đúng
            gt    = self.state['target_gt'].detach().cpu().numpy()
            self.state['_val_scores'].append(probs)
            self.state['_val_targets'].append(gt)

    def on_end_epoch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        loss = self.state['meter_loss'].avg

        if training:
            map_val = 100.0 * self.state['ap_meter'].value().mean()
            if display:
                print(f'Epoch: [{self.state["epoch"]}]\t'
                      f'Loss {loss:.4f}\tmAP {map_val:.3f}')
            return map_val

        # Validation
        val_scores  = self.state.get('_val_scores',  [])
        val_targets = self.state.get('_val_targets', [])

        if not val_scores:
            print(f'Val:\tLoss {loss:.4f}  (no predictions collected)')
            return 0.0

        scores  = np.concatenate(val_scores,  axis=0)   # [N, 14]
        targets = np.concatenate(val_targets, axis=0)   # [N, 14] — có thể có -1

        mAP, per_class_ap = compute_mAP(scores, targets)
        mean_auc, per_cls = compute_mean_AUC(scores, targets)
        unc_auc, per_class_unc_auc = compute_AUC_uncertain(scores, targets)

        results = {
            'map'          : round(mAP,      4) if not np.isnan(mAP)      else None,
            'mean_auc'     : round(mean_auc, 4) if not np.isnan(mean_auc) else None,
            # unc_auc = nan là bình thường khi val set không có -1 (CheXpert official val)
            'unc_auc'      : round(unc_auc,  4) if not np.isnan(unc_auc)  else None,
            'per_class_auc': per_cls,
            'per_class_ap' : per_class_ap,
            'per_class_unc_auc': per_class_unc_auc,
        }

        if display:
            print(f'\nVal:\tLoss {loss:.4f}')
            loss_type = self.state.get('loss_type', 'bce')
            print_metrics(results, show_unc=(loss_type == 'ua_asl'))

        # Dùng mAP làm primary score để chọn checkpoint (khớp proposal).
        # Fallback mean_auc nếu mAP không tính được.
        score = mAP if not np.isnan(mAP) else (mean_auc if not np.isnan(mean_auc) else 0.0)
        return float(score)

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        feature_var = self.state['feature'].float()
        target_var  = self.state['target'].float()

        if self.state['use_gpu']:
            feature_var = feature_var.cuda()

        # inp là register_buffer trong model — tự .cuda() cùng model,
        # không cần truyền qua đây nữa
        self.state['output'] = model(feature_var)   # logits [B, 14]
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()