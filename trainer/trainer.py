import re
import time
import queue
import threading
import json
import os
from collections import defaultdict
from .logger import Logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data_provider.train_loader import TrainLoader
from data_provider.validation_loader import ValidationLoader
# from common import config as cfg
from pathlib import Path
from model.finetune_model import FinetuneModel as model_fn


class Trainer:
    def __init__(self, local_rank, world_rank):
        self.local_rank = local_rank
        self.world_rank = world_rank

        self.device = torch.device("cuda:%i" % local_rank)    
        
        with open('../trainer/config.json', 'r') as f:
            self.params = json.load(f)
        self.global_batch_size = self.params['global_batch_size']
        self.local_batch_size = self.params['local_batch_size']
        self.warmup_step = self.params['warmup_step']
        self.lr = self.params['lr']
        self.nr_step = self.params['nr_step']
        self.mixed_precision = self.params['mixed_precision']
        self.chk_time_interval = self.params['chk_time_interval']
        self.chk_step_interval = [self.params['chk_step_interval']]
        # self.saved_dir = Path(self.params['saved_dir'])
        # self.pretrained_dir = Path(self.params['pretrained_dir'])
        # self.dataset_dir = Path(self.params['dataset_dir'])
        self.log_dir = Path(self.params['log_dir'])
        self.model_dir = Path(self.params['model_dir'])
        self.pretrain_model_dir = Path(self.params['pretrain_model_dir'])

        self.n_gpu = torch.distributed.get_world_size()

        self.n_accumulate = (
            self.global_batch_size // self.n_gpu // self.local_batch_size
        )
        
        
        self.dp = TrainLoader()
        self.vp = ValidationLoader()
        self.model = model_fn().to(self.device)
        if world_rank == 0:
            # self.logger = cfg.train_logger
            self.logger = Logger(self.log_dir / "train_log.txt")
            self.logger.info(str(self.model))
        self.model = DDP(self.model, device_ids=[self.local_rank])

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=0.001, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
        )

        self.step = 0
        self.load_checkpoint()

        if self.world_rank == 0:
            self.chk_worker = queue.Queue()
            threading.Thread(target=self.sync_checkpoint, daemon=True).start()

    def adjust_learning_rate(self):
        def inverse_sqrt_root_schedule(step, n_warmup, lr):
            factor = lr * n_warmup**0.5
            return factor * min(step**-0.5, step * n_warmup**-1.5)

        lr = inverse_sqrt_root_schedule(self.step + 1, self.warmup_step, self.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self):
        dp = iter(self.dp)
        cur_time = last_saved_time = start_time = time.time()
        n_trained = 0
        scaler = torch.amp.GradScaler('cuda')
        for i in range(self.step + 1, self.nr_step + 1):
            self.adjust_learning_rate()

            b_losses, b_metrics = defaultdict(list), defaultdict(list)

            def step_forward():
                nonlocal dp, scaler
                data = {k: v.to(self.device) for k, v in next(dp).items()}
                with torch.amp.autocast('cuda', enabled = self.mixed_precision):
                    output = self.model(data)
                    losses = {k: v for k, v in output.items() if k.endswith("loss")}
                    metrics = {
                        k: v
                        for k, v in output.items()
                        if k.endswith("_acc") or k.startswith("token") or k.endswith("f1")
                    }
                    loss = output["update_loss"]
                    loss = loss / self.n_accumulate
                    scaler.scale(loss).backward()
                for k, v in losses.items():
                    b_losses[k].append(v.item())
                for k, v in metrics.items():
                    b_metrics[k].append(v)

            with self.model.no_sync():
                for j in range(self.n_accumulate - 1):
                    step_forward()
            step_forward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            n_trained += 1
            self.step += 1

            loss_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_losses.items()]
            )
            metric_str = ", ".join(
                ["%s: %.6f" % (k, sum(v) / len(v)) for k, v in b_metrics.items()]
            )

            speed = 1.0 / (time.time() - cur_time)
            passed_time = (time.time() - start_time) / 3600

            estimate_time = (self.nr_step - self.step) / n_trained * passed_time
            log_str = (
                f"Train Step: [{self.step}/{self.nr_step}], "
                f"{loss_str}, {metric_str}, "
                f"Speed: {speed:.3f} m/s, "
                f"Passed: {passed_time:.3f} h, "
                f"Estimate: {estimate_time:.3f} h"
            )
            if self.world_rank == 0:
                if i % 10 == 0:
                    self.logger.info(log_str)

            if i % 10 == 0:
                print(log_str)


            cur_time = time.time()
            if (
                i % self.chk_step_interval[0] == 0
                or cur_time - last_saved_time > self.chk_time_interval
            ):
                self.save_checkpoint()
                last_saved_time = cur_time
                
                

    def save_checkpoint(self):
        if self.world_rank != 0:
            return
        state = {
            "step": self.step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        filename = self.model_dir / f"checkpoint-step-{self.step}.pth"

        self.logger.info(f"Saving checkpoint: {filename} ...")
        torch.save(state, filename)
        self.chk_worker.put(filename)

    def load_checkpoint(self):
        latest = -1
        for path in self.model_dir.iterdir():
            if path.stem.startswith("checkpoint-step-"):
                step = int(re.findall(r"\d+", path.stem)[0])
                latest = max(latest, step)
        if latest != -1:
            filename = self.model_dir / f"checkpoint-step-{latest}.pth"
            if self.world_rank == 0:
                self.logger.info(f"Loading checkpoint: {filename} ...")
            checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
            self.step = checkpoint["step"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.world_rank == 0:
                self.logger.info(f"Checkpoint '{filename}' (step {self.step}) loaded")
        else:
            pretrain_path = (
                self.pretrain_model_dir / f"pretrain_weights.pth"
            )
            if self.world_rank == 0:
                self.logger.info(f"Loading pretrain checkpoint: {pretrain_path} ...")
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            parsed_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("module."):
                    k = k[7:]
                parsed_dict[k] = v
            self.model.module.pretrain.load_state_dict(parsed_dict)
            if self.world_rank == 0:
                self.logger.info(f"Pretrain checkpoint '{pretrain_path}' loaded")

    def sync_checkpoint(self):
        while True:
            path = self.chk_worker.get()
            print(f"Finished {path}")
            cur_step = int(re.findall(r"\d+", path.stem)[0])
            for it in self.chk_step_interval:
                if cur_step % it == 0:
                    print(f"Clean checkpoint every {it} step")
                    for chk in self.model_dir.iterdir():
                        if chk.stem.startswith("checkpoint"):
                            chk_step = int(re.findall(r"\d+", chk.stem)[0])
                            if chk_step % it != 0:
                                chk_path = self.model_dir / chk
                                print(f"Remove {chk_path}")
                                chk_path.unlink()
            self.chk_worker.task_done()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_rank == 0:
            if hasattr(self.logger, "flush"):
                self.logger.flush()
            self.chk_worker.join()
