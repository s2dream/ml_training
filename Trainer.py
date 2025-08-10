import os
import math
import time
from typing import Dict, Any
from Evaluator import Evaluator
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.cuda.amp import autocast, GradScaler


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def ddp_setup(backend: str = "nccl"):
    """
    torchrun 으로 실행하면 LOCAL_RANK/ RANK/ WORLD_SIZE 가 환경변수로 들어옵니다.
    """
    if is_dist_avail_and_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = 0, 1

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    dist.barrier()

def ddp_cleanup():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    """
    float 스칼라를 all-reduce. world_size로 나누는 건 호출부에서 결정.
    """
    if not is_dist_avail_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(t, op=op)
    return float(t.item())

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out

def rewrap_with_sampler(loader, sampler) -> torch.utils.data.DataLoader:
    """
    기존 DataLoader를 sampler가 적용된 새 DataLoader로 재구성.
    collate_fn, num_workers 등은 기존 것을 최대한 유지합니다.
    """
    return type(loader)(
        dataset=loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        persistent_workers=getattr(loader, "persistent_workers", False)
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader_helper,
        save_dir: str = "./checkpoints",
        max_epochs: int = 10,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        log_interval: int = 50,
        scheduler_type: str = "cosine",  # "cosine" | "plateau" | None
        use_amp: bool = True,
        backend: str = "nccl"
    ):
        ddp_setup(backend=backend)

        self.helper = dataloader_helper
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.scheduler_type = scheduler_type
        self.use_amp = use_amp

        os.makedirs(save_dir, exist_ok=True)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

        # 모델 & DDP 래핑
        self.model = model.to(self.device)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank] if self.device.type == "cuda" else None,
            output_device=local_rank if self.device.type == "cuda" else None,
            find_unused_parameters=False
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=(use_amp and torch.cuda.is_available()))

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        else:
            self.scheduler = None

        # Evaluator는 .module 을 참조해야 실제 모델에 접근
        self.evaluator = Evaluator(self.model.module, self.device)

    def _build_loader_with_sampler(self, base_loader, shuffle: bool) -> torch.utils.data.DataLoader:
        if not is_dist_avail_and_initialized():
            return base_loader
        sampler = DistributedSampler(base_loader.dataset, shuffle=shuffle, drop_last=base_loader.drop_last)
        return rewrap_with_sampler(base_loader, sampler)

    def fit(self):
        # DataLoaders
        train_loader = self.helper.get_train_dataloader()
        valid_loader = getattr(self.helper, "get_valid_dataloader", None)
        has_valid = callable(valid_loader)
        if has_valid:
            valid_loader = self.helper.get_valid_dataloader()

        # Samplers
        # train_loader = self._build_loader_with_sampler(train_loader, shuffle=True)
        if has_valid:
            valid_loader = self._build_loader_with_sampler(valid_loader, shuffle=False)

        best_val = float("inf")
        best_path = os.path.join(self.save_dir, "best.pt")

        for epoch in range(1, self.max_epochs + 1):
            # epoch 설정을 sampler에 반영 (shuffle 재시드)
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if has_valid and isinstance(valid_loader.sampler, DistributedSampler):
                valid_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss, n_seen = 0.0, 0
            t0 = time.time()

            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                    y_pred = self.model(
                        x_metro_item_set_id=batch["x_metro_item_set_id"],
                        x_metro_item_value=batch["x_metro_item_value"],
                        x_metro_set_id=batch["x_metro_set_id"],
                        padding_mask_main_steps=batch["padding_mask_main_steps"]
                    )
                    loss = self.loss_fn(y_pred, batch["y"])

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                bs = batch["y"].size(0)
                epoch_loss += float(loss.item()) * bs
                n_seen += bs

                if is_main_process() and (step % self.log_interval == 0):
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"[Epoch {epoch}/{self.max_epochs}] Step {step} | loss={loss.item():.5f} | lr={lr:.3e}")

            # 분산 합산으로 train 평균 loss 계산(로깅용)
            world = get_world_size()
            epoch_loss = reduce_scalar(epoch_loss, op=dist.ReduceOp.SUM)
            n_seen = reduce_scalar(n_seen, op=dist.ReduceOp.SUM)
            train_avg = epoch_loss / max(n_seen, 1)

            if has_valid:
                val_metrics = self.evaluator.evaluate(valid_loader)
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                elif self.scheduler is not None:
                    self.scheduler.step()

                if is_main_process():
                    print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | "
                          f"train_loss={train_avg:.5f} | "
                          f"val_loss={val_metrics['loss']:.5f} | "
                          f"val_rmse={val_metrics['rmse']:.5f} | "
                          f"val_mae={val_metrics['mae']:.5f} | "
                          f"val_r2={val_metrics['r2']:.5f}")

                    # 베스트 체크포인트 저장
                    if val_metrics["loss"] < best_val:
                        best_val = val_metrics["loss"]
                        torch.save({"model_state": self.model.module.state_dict(),
                                    "epoch": epoch,
                                    "val_loss": best_val}, best_path)
                        print(f"  -> New best saved to {best_path} (val_loss={best_val:.5f})")
            else:
                if self.scheduler and self.scheduler_type != "plateau":
                    self.scheduler.step()
                if is_main_process():
                    print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | train_loss={train_avg:.5f}")

        # 학습 종료: best 로드(모든 rank 동기화 후)
        if is_main_process() and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.module.load_state_dict(ckpt["model_state"])
        if is_dist_avail_and_initialized():
            dist.barrier()

        return self.model.module  # 언랩하여 실제 model 반환