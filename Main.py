# run_without_torchrun.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Trainer import Trainer
from Evaluator import Evaluator
from model.sequential.ETPredTransformer import ETPredRegressor
from dataloader.dataloader_v1 import DataloaderHelper
# from your_model_file import MetroRegressor
# from your_dataloader_helper import DataloaderHelper



MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"   # 사용 가능한 포트로 바꾸세요

def ddp_init(rank: int, world_size: int, backend: str = "nccl"):
    # torchrun 없이 수동 초기화
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    dist.barrier()

def ddp_cleanup():
    dist.barrier()
    dist.destroy_process_group()

def worker(rank: int, world_size: int, cfg: dict):
    ddp_init(rank, world_size, backend=cfg.get("backend", "nccl"))

    # --- 모델/데이터로더 구성 ---
    TOTAL_SIZE_METRO_ITEM = cfg["TOTAL_SIZE_METRO_ITEM"]
    TOTAL_SIZE_METRO_STEP = cfg["TOTAL_SIZE_METRO_STEP"]

    model = ETPredRegressor(
        TOTAL_SIZE_METRO_ITEM=TOTAL_SIZE_METRO_ITEM,
        TOTAL_SIZE_METRO_STEP=TOTAL_SIZE_METRO_STEP,
        dim=512,
        nhead=8,
        num_transformer_layers=4,
        combine_mode="concat_project"
    )

    helper = DataloaderHelper()  # 사용자의 구현체

    # --- 학습 ---
    trainer = Trainer(
        model=model,
        dataloader_helper=helper,
        save_dir=cfg.get("save_dir", "./checkpoints_ddp"),
        max_epochs=cfg.get("max_epochs", 10),
        lr=cfg.get("lr", 2e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        grad_clip=cfg.get("grad_clip", 1.0),
        log_interval=cfg.get("log_interval", 50),
        scheduler_type=cfg.get("scheduler_type", "cosine"),
        use_amp=cfg.get("use_amp", True),
        backend=cfg.get("backend", "nccl")  # Trainer 내부 ddp_setup은 이미 init되어 있으면 스킵
    )
    trained = trainer.fit()

    # --- 테스트 ---
    if dist.get_rank() == 0:
        evaluator = Evaluator(trained, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        evaluator.test(helper)

    ddp_cleanup()

def main():
    world_size = torch.cuda.device_count()
    assert world_size > 1, "멀티 GPU가 필요합니다. (CUDA 장치가 2개 이상인지 확인하세요)"

    cfg = dict(
        TOTAL_SIZE_METRO_ITEM=50000,
        TOTAL_SIZE_METRO_STEP=10000,
        save_dir="./checkpoints_ddp_spawn",
        max_epochs=10,
        lr=2e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        log_interval=50,
        scheduler_type="cosine",
        use_amp=True,
        backend="nccl",
    )

    mp.spawn(worker, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()