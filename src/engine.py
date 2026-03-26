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

from util import Warp, MultiScaleCrop, AveragePrecisionMeter

from evaluate import (
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
        # Torchnet-compatible: (mean, std).  std not tracked → 0.
        return (self.avg, 0.0)


# ─── Base Engine ─────────────────────────────────────────────────────────────

class Engine:
    """
    Base training loop using a callback/hook pattern.

    Subclasses override hooks rather than copy the full loop:
        on_start_epoch → on_start_batch → on_forward → on_end_batch → on_end_epoch

    GCNMultiLabelMAPEngine overrides on_start_batch (unpack tuple + remap
    labels) and on_forward (two-input GCN forward pass).
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
            bt   = self.state['batch_time'].avg
            dt   = self.state['data_time'].avg
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {btc:.3f} ({bt:.3f})\t'
                      'Data {dtc:.3f} ({dt:.3f})\t'
                      'Loss {lc:.4f} ({l:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    btc=self.state['batch_time_current'], bt=bt,
                    dtc=self.state['data_time_batch'],    dt=dt,
                    lc=self.state['loss_batch'],           l=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {btc:.3f} ({bt:.3f})\t'
                      'Data {dtc:.3f} ({dt:.3f})\t'
                      'Loss {lc:.4f} ({l:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    btc=self.state['batch_time_current'], bt=bt,
                    dtc=self.state['data_time_batch'],    dt=dt,
                    lc=self.state['loss_batch'],           l=loss))

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
        if self._state('train_transform') is None:
            mean = getattr(model, 'image_normalization_mean',
                           [0.485, 0.456, 0.406])
            std  = getattr(model, 'image_normalization_std',
                           [0.229, 0.224, 0.225])
            normalize = transforms.Normalize(mean=mean, std=std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'],
                               scales=(1.0, 0.875, 0.75, 0.66, 0.5),
                               max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            mean = getattr(model, 'image_normalization_mean',
                           [0.485, 0.456, 0.406])
            std  = getattr(model, 'image_normalization_std',
                           [0.229, 0.224, 0.225])
            normalize = transforms.Normalize(mean=mean, std=std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
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
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Val')

        end = time.time()
        with torch.no_grad():
            # no_grad: không allocate bộ nhớ cho gradient
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

    def save_checkpoint(self, state, is_best,
                        filename='checkpoint.pth.tar'):
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

    def adjust_learning_rate(self, optimizer):
        """Decay lr × 0.1 at each epoch listed in state['epoch_step']."""
        decay = 0.1 if self.state['epoch'] in self.state['epoch_step'] else 1.0
        lr_list = []
        for pg in optimizer.param_groups:
            pg['lr'] *= decay
            lr_list.append(pg['lr'])
        return np.unique(lr_list)


# ─── Multi-label mAP Engine ──────────────────────────────────────────────────

class MultiLabelMAPEngine(Engine):
    """
    Extends Engine với mAP metric cho multi-label classification.
    Override on_end_epoch để tính và in mAP thay vì chỉ loss.
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

        # Unpack input tuple: (img, path)
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
            bt   = self.state['batch_time'].avg
            dt   = self.state['data_time'].avg
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
    Extends MultiLabelMAPEngine for GCNResnet.

    Changes vs MultiLabelMAPEngine:
      1. on_start_batch — unpacks 3-tuple (img, path, word_vec); remaps
         uncertain labels (-1 → 0) correctly for BCE loss while preserving
         ground-truth {-1, 0, 1} in target_gt for evaluation.
      2. on_forward — calls model(img, word_vec) with two inputs.
      3. on_start_epoch / on_end_batch — accumulate val predictions so that
         on_end_epoch can compute AUC/mAP via evaluate.py (single pass).
      4. on_end_epoch (val) — uses evaluate.py metrics and returns mean_auc
         as the score for checkpoint selection.
    """

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_start_epoch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        MultiLabelMAPEngine.on_start_epoch(
            self, training, model, criterion, data_loader, optimizer)
        if not training:
            # Buffers to accumulate predictions for evaluate.py metrics
            self.state['_val_scores']  = []
            self.state['_val_targets'] = []

    def on_start_batch(self, training, model, criterion, data_loader,
                       optimizer=None, display=True):
        # Preserve original labels for evaluation (keeps -1 uncertain)
        self.state['target_gt'] = self.state['target'].clone()

        # Remap for BCEWithLogitsLoss: uncertain (-1) → negative (0).
        # Positive (1) and explicit negative (0) stay unchanged.
        target = self.state['target'].clone().float()
        target[target < 0] = 0.0
        self.state['target'] = target

        # Unpack (img, path, word_vec) from CheXpertDataset.__getitem__
        input = self.state['input']
        self.state['feature'] = input[0]   # [B, 3, H, W]
        self.state['out']     = input[1]   # list of path strings
        self.state['input']   = input[2]   # [14, 300] word embeddings

    def on_end_batch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        MultiLabelMAPEngine.on_end_batch(
            self, training, model, criterion, data_loader, display)

        if not training:
            # Accumulate sigmoid scores and raw {-1,0,1} targets
            probs = torch.sigmoid(self.state['output']).detach().cpu().numpy()
            gt    = self.state['target_gt'].detach().cpu().numpy()
            self.state['_val_scores'].append(probs)
            self.state['_val_targets'].append(gt)

    def on_end_epoch(self, training, model, criterion, data_loader,
                     optimizer=None, display=True):
        loss = self.state['meter_loss'].avg

        if training:
            # Training: quick mAP from ap_meter (no extra pass needed)
            map_val = 100.0 * self.state['ap_meter'].value().mean()
            if display:
                print(f'Epoch: [{self.state["epoch"]}]\t'
                      f'Loss {loss:.4f}\tmAP {map_val:.3f}')
            return map_val

        # Validation: use evaluate.py for proper AUC + mAP metrics
        val_scores  = self.state.get('_val_scores',  [])
        val_targets = self.state.get('_val_targets', [])

        if not val_scores:
            print(f'Test:\tLoss {loss:.4f}  (no predictions collected)')
            return 0.0

        scores  = np.concatenate(val_scores,  axis=0)  # [N, 14]
        targets = np.concatenate(val_targets, axis=0)  # [N, 14]

        mAP                   = compute_mAP(scores, targets)
        mean_auc, per_class   = compute_mean_AUC(scores, targets)
        unc_auc               = compute_AUC_uncertain(scores, targets)

        results = {
            'map'          : round(mAP,      4) if not np.isnan(mAP)      else None,
            'mean_auc'     : round(mean_auc, 4) if not np.isnan(mean_auc) else None,
            'unc_auc'      : round(unc_auc,  4) if not np.isnan(unc_auc)  else None,
            'per_class_auc': per_class,
        }

        if display:
            print(f'\nVal:\tLoss {loss:.4f}')
            print_metrics(results)

        # Return mean_auc as the scalar score for checkpoint selection.
        # Falls back to mAP if AUC cannot be computed (e.g. all-zeros val set).
        score = mean_auc if not np.isnan(mean_auc) else (mAP if not np.isnan(mAP) else 0.0)
        return float(score)

    def on_forward(self, training, model, criterion, data_loader,
                   optimizer=None, display=True):
        feature_var = self.state['feature'].float()
        target_var  = self.state['target'].float()
        # word_vec detached: gradients flow through GCN weights, not embeddings
        inp_var = self.state['input'].float().detach()

        if self.state['use_gpu']:
            feature_var = feature_var.cuda()
            inp_var     = inp_var.cuda()
            # target already moved to GPU in train()/validate()

        self.state['output'] = model(feature_var, inp_var)  # logits [B, 14]
        self.state['loss']   = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            # Clip gradient: nếu norm > 10 thì scale xuống
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()