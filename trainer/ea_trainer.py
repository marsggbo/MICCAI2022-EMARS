# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import json
import sys
import logging

sys.path.append('..')

import numpy as np
import torch
import torch.optim as optim
from callbacks import CheckpointCallback
from kd_model import load_kd_model, loss_fn_kd
from nas.utils import flops_size_counter
from nni.nas.pytorch.callbacks import Callback
from utils import AverageMeterGroup

from .build import TRAINER_REGISTRY
from .default_trainer import Trainer, TorchTensorEncoder

__all__ = [
    'EATrainer',
]


@TRAINER_REGISTRY.register()
class EATrainer(Trainer):
    def __init__(self, cfg):
        """Initialize an EATrainer.
            Parameters
            ----------
            model : nn.Module
                PyTorch model to be trained.
            loss : callable
                Receives logits and ground truth label, return a loss tensor.
            metrics : callable
                Receives logits and ground truth label, return a dict of metrics.
            reward_function : callable
                Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
            optimizer : Optimizer
                The optimizer used for optimizing the model.
            num_epochs : int
                Number of epochs planned for training.
            dataset_train : Dataset
                Dataset for training. Will be split for training weights and architecture weights.
            dataset_valid : Dataset
                Dataset for testing.
            mutator : EnasMutator
                Use when customizing your own mutator or a mutator with customized parameters.
            batch_size : int
                Batch size.
            workers : int
                Workers for data loading.
            device : torch.device
                ``torch.device("cpu")`` or ``torch.device("cuda")``.
            log_frequency : int
                Step count per logging.
            callbacks : list of Callback
                list of callbacks to trigger at events.
            entropy_weight : float
                Weight of sample entropy loss.
            skip_weight : float
                Weight of skip penalty loss.
            baseline_decay : float
                Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
            mutator_lr : float
                Learning rate for RL controller.
            mutator_steps_aggregate : int
                Number of steps that will be aggregated into one mini-batch for RL controller.
            mutator_steps : int
                Number of mini-batches for each epoch of RL controller learning.
            aux_weight : float
                Weight of auxiliary head loss. ``aux_weight * aux_loss`` will be added to total loss.
        """
        cfg.defrost()
        cfg.mutator.name = 'EAMutator'
        cfg.freeze()
        super(EATrainer, self).__init__(cfg)
        self.cfg = cfg
        self.debug = cfg.debug

        # preparing dataset
        # n_train = len(self.dataset_train)
        # split = n_train // 3
        # indices = list(range(n_train))
        # random.shuffle(indices)
        # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        # self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
        #                                                 batch_size=self.batch_size,
        #                                                 sampler=train_sampler,
        #                                                 num_workers=self.workers,
        #                                                 pin_memory=True)
        # self.test_loader = torch.utils.data.DataLoader(self.dataset_train,
        #                                                 batch_size=self.batch_size,
        #                                                 sampler=valid_sampler,
        #                                                 num_workers=self.workers,
        #                                                 pin_memory=True)
        # self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
        #                                                batch_size=self.batch_size,
        #                                                num_workers=self.workers,
        #                                                pin_memory=True)

        # preparing dataset
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.workers,
                                                        pin_memory=True,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       shuffle=False)

        if hasattr(self.cfg, 'kd') and self.cfg.kd.enable:
            self.kd_model = load_kd_model(self.cfg).to(self.device)
            if len(self.cfg.trainer.device_ids) > 1:
                self.kd_model = torch.nn.DataParallel(self.kd_model, device_ids=self.cfg.trainer.device_ids)
                self.kd_model.eval()
        else:
            self.kd_model = None

        self.mutator.trainer = self

    def train(self, validate=True):
        self.resume()
        self.train_meters = None
        self.valid_meters = None
        for epoch in range(self.start_epoch, self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            self.logger.info("Epoch {} Training".format(epoch))
            self.train_meters = self.train_one_epoch(epoch)
            self.logger.info("Final training metric: {}".format(self.train_meters))

            if validate:
                # validation
                self.logger.info("Epoch {} Validatin".format(epoch))
                self.valid_meters = self.validate_one_epoch(epoch)
                self.logger.info("Final test metric: {}".format(self.valid_meters))

            for callback in self.callbacks:
                if isinstance(callback, CheckpointCallback):
                    if self.valid_meters:
                        meters = self.valid_meters
                    else:
                        meters = self.train_meters
                    callback.update_best_metric(meters.meters['save_metric'].avg)
                if not self.mutator.start_evolve and isinstance(callback, CheckpointCallback):
                    continue
                callback.on_epoch_end(epoch)

    def train_one_epoch(self, epoch):
        # Sample model and train
        self.model.train()
        self.mutator.eval()
        meters = AverageMeterGroup()
        if self.mutator.is_initialized:
            self.mutator.evolve()

        self.mutator.start_evolve = True if epoch >= self.mutator.warmup_epochs else False
        self.num_population = self.mutator.num_crt_population if self.mutator.start_evolve else 1
        for step, sample_batched in enumerate(self.train_loader):
            if self.debug and step > 0:
                break
            inputs, targets = sample_batched
            inputs = inputs.cuda()
            targets = targets.cuda()

            idx = step % self.num_population
            with torch.no_grad():
                self.mutator.reset()

            output = self.model(inputs)
            if isinstance(output, tuple):
                output, aux_output = output
                aux_loss = self.loss(aux_output, targets)
            else:
                aux_loss = 0.
            loss = self.loss(output, targets)
            loss = loss + self.cfg.model.aux_weight * aux_loss
            if self.kd_model:
                teacher_output = self.kd_model(inputs)
                loss = (1-self.cfg.kd.loss.alpha)*loss + loss_fn_kd(output, teacher_output, self.cfg.kd.loss)
            metrics = self.metrics(output, targets)

            # record loss and EPE
            metrics['loss'] = loss.item()
            meters.update(metrics)
            arch_code = self.mutator.encode_arch(self.mutator._cache)
            if not self.mutator.start_evolve:
                individual = {
                    'arch_code': arch_code,
                    'arch': self.mutator._cache,
                    'meters': meters['save_metric'].avg
                }
                if arch_code not in self.mutator.history:
                    flops, size = self.model_size('flops'), self.model_size('size')
                    individual['flops'] = flops
                    individual['size'] = size
                self.mutator.update_history(individual)
            else:
                self.mutator.update_individual(idx, meters)

            loss.backward()
            if (step+1) % self.cfg.trainer.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Model Epoch [{}/{}] Step [{}/{}] Model size: {:.4f}MB {}".format(
                    epoch + 1, self.num_epochs, step + 1, len(self.train_loader), self.model_size(), meters))
        return meters

    def validate_one_epoch(self, epoch):
        if not self.mutator.start_evolve:
            return None
        else:
            best_meters = None
            best = 0
            num_population = self.mutator.num_crt_population
            assert num_population == len(self.mutator.pools)
            for idx in range(num_population):
                self.mutator.reset()
                meters = self.test_one_epoch(epoch)        
                flops, size = [self.mutator.pools[idx][k] for k in ['flops', 'size']]         
                self.logger.info("Final model metric of {} arch (flops={:.4f} MFLOPS size={:.4f} MB) = {}".format(
                                                        idx, flops, size, meters.meters['save_metric'].avg))
                if meters['save_metric'].avg > best:
                    best = meters['save_metric'].avg
                    best_meters = meters
                self.mutator.update_individual(idx, meters)
            return best_meters

    def test_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()

        meters = AverageMeterGroup()
        for step, sample_batched in enumerate(self.test_loader):
            if self.debug and step > 0:
                break
            inputs, targets = sample_batched
            inputs = inputs.cuda()
            targets = targets.cuda()

            self.optimizer.zero_grad()
            output = self.model(inputs)
            if isinstance(output, tuple):
                output, aux_output = output
                aux_loss = self.loss(aux_output, targets)
            else:
                aux_loss = 0.
            loss = self.loss(output, targets)
            loss = loss + self.cfg.model.aux_weight * aux_loss
            if self.kd_model:
                teacher_output = self.kd_model(inputs)
                loss = (1-self.cfg.kd.loss.alpha)*loss + loss_fn_kd(output, teacher_output, self.cfg.kd.loss)

            metrics = self.metrics(output, targets)

            # record loss and EPE
            metrics['loss'] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Test: Step [{}/{}]  {}".format(step + 1, len(self.test_loader), meters))

        return meters

    # todo: 调试callbacks
    def generate_callbacks(self):
        '''
        Args：
            func: a function to generate other callbacks, must return a list
        Return:
            a list of callbacks.
        '''
        self.ckpt_callback = CheckpointCallback(
            checkpoint_dir=self.cfg.logger.path,
            name='best_search.pth',
            mode=self.cfg.callback.checkpoint.mode)
        self.arch_callback = ArchitectureCheckpoint(self.cfg.logger.path, self, self.mutator)
        callbacks = [
            # self.ckpt_callback,
            self.arch_callback,
        ]
        return callbacks


class ArchitectureCheckpoint(Callback):
    """
    Calls ``trainer.export()`` on every epoch ends.

    Parameters
    ----------
    checkpoint_dir : str
        Location to save checkpoints.
    """
    def __init__(self, checkpoint_dir, trainer, mutator):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.mutator = mutator
        self.trainer = trainer
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.report_file = os.path.join(self.checkpoint_dir, 'report.csv')
        with open(self.report_file, 'w') as f:
            f.write('epoch,idx,meters,flops(MFLOPS),size(MB)\n')

    def on_epoch_end(self, epoch):
        """
        Dump to ``/checkpoint_dir/epoch_{number}.json`` on epoch end.
        """
        if not self.mutator.start_evolve:
            return None
        for idx in self.mutator.pools:
            dest_path = os.path.join(self.checkpoint_dir, "epoch_{}_{}.json".format(epoch, idx))
            logging.info("Saving architecture to %s", dest_path)
            individual = self.mutator.pools[idx]
            arch, meters, flops, size, _ = [individual[k] for k in ['arch', 'meters', 'flops', 'size', 'arch_code']]
            with open(dest_path, "w") as f:
                json.dump(arch, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)
            with open(self.report_file, 'a') as f:
                f.write(f'{epoch},{idx},{meters},{flops},{size}\n')
        self.mutator.save_history(self.mutator.history)