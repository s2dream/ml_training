import os
import math
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import r2_score


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss, n_samples = 0.0, 0
        preds_list, targets_list = [], []

        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            with autocast(enabled=torch.cuda.is_available()):
                y_pred = self.model(
                    x_metro_item_set_id=batch["x_metro_item_set_id"],
                    x_metro_item_value=batch["x_metro_item_value"],
                    x_metro_set_id=batch["x_metro_set_id"],
                    padding_mask_main_steps=batch["padding_mask_main_steps"]
                )
                loss = self.loss_fn(y_pred, batch["y"])

            bs = batch["y"].size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            preds_list.append(y_pred.detach().float().cpu())
            targets_list.append(batch["y"].detach().float().cpu())

        preds = torch.cat(preds_list, dim=0).squeeze(-1).numpy()
        targets = torch.cat(targets_list, dim=0).squeeze(-1).numpy()

        mse = ((preds - targets) ** 2).mean()
        rmse = math.sqrt(max(mse, 0.0))
        mae = (abs(preds - targets)).mean()
        try:
            r2 = float(r2_score(targets, preds))
        except Exception:
            r2 = float("nan")

        avg_loss = total_loss / max(n_samples, 1)
        return {"loss": avg_loss, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    @torch.no_grad()
    def test(self, dataloader_helper) -> Dict[str, float]:
        test_loader = dataloader_helper.get_test_dataloader()
        metrics = self.evaluate(test_loader)
        print(f"[TEST] loss={metrics['loss']:.5f} | mse={metrics['mse']:.5f} "
              f"| rmse={metrics['rmse']:.5f} | mae={metrics['mae']:.5f} | r2={metrics['r2']:.5f}")
        return metrics


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
        scheduler_type: str = "cosine",  # "cosine", "plateau", None
        use_amp: bool = True
    ):
        self.model = model
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=(use_amp and torch.cuda.is_available()))

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        else:
            self.scheduler = None

        self.evaluator = Evaluator(self.model, self.device)

    def fit(self):
        train_loader = self.helper.get_train_dataloader()
        valid_loader = getattr(self.helper, "get_valid_dataloader", None)
        has_valid = callable(valid_loader)
        if has_valid:
            valid_loader = self.helper.get_valid_dataloader()

        best_val = float("inf")
        best_path = os.path.join(self.save_dir, "best.pt")

        global_step = 0
        for epoch in range(1, self.max_epochs + 1):
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
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                epoch_loss += loss.item() * batch["y"].size(0)
                n_seen += batch["y"].size(0)
                global_step += 1

                if step % self.log_interval == 0:
                    print(f"[Epoch {epoch}/{self.max_epochs}] Step {step} | "
                          f"loss={loss.item():.5f} | lr={self.optimizer.param_groups[0]['lr']:.3e}")

            train_avg = epoch_loss / max(n_seen, 1)

            if has_valid:
                val_metrics = self.evaluator.evaluate(valid_loader)
                print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | "
                      f"train_loss={train_avg:.5f} | "
                      f"val_loss={val_metrics['loss']:.5f} | "
                      f"val_rmse={val_metrics['rmse']:.5f} | "
                      f"val_mae={val_metrics['mae']:.5f} | "
                      f"val_r2={val_metrics['r2']:.5f}")

                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                elif self.scheduler is not None:
                    self.scheduler.step()

                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    torch.save({"model_state": self.model.state_dict(), "epoch": epoch, "val_loss": best_val},
                               best_path)
                    print(f"  -> New best saved to {best_path} (val_loss={best_val:.5f})")
            else:
                if self.scheduler and self.scheduler_type != "plateau":
                    self.scheduler.step()
                print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | train_loss={train_avg:.5f}")

        if os.path.exists(best_path):
            print(f"Loading best checkpoint from {best_path}")
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
        return self.model
