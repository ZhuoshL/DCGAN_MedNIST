from pathlib import Path
import math, random, time, os
import numpy as np
import pytorch_lightning as pl
import torch
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import MedNISTDataModule
from DCGAN_MedNIST.gan_module import GAN

def set_random_seed(random_seed=42):
    # 固定环境变量、Python 标准库、NumPy、PyTorch 以及 MONAI 的随机种子，关闭 cuDNN 动态优化以确保实验可复现
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_determinism(random_seed)

def train(args):
    # 根据传入参数设定随机种子，确保训练过程可复现
    set_random_seed(args.seed)

    # 创建检查点保存目录与结果保存目录（自动创建不存在的父级目录）
    checkpoint_dir, result_dir = Path(args.checkpoint_dir), Path(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 MedNIST 数据模块，仅加载 "Hand"（手部 X 光）分类数据
    dm = MedNISTDataModule(batch_size=args.batch_size, num_workers=args.workers, class_names=["Hand"], root_dir=args.data_dir)

    # 实例化基于 PyTorch Lightning 封装的 DCGAN 模型，解包传入数据维度及各项超参数
    model = GAN(*dm.dims, latent_dim=args.latent_dim, lr=args.lr, b1=args.beta1, b2=args.beta2, batch_size=args.batch_size, result_dir=args.result_dir)

    # 配置 ModelCheckpoint 回调函数，训练过程中自动将最新的 Lightning 权重保存为 "last.ckpt"
    checkpoint_callback = ModelCheckpoint(dirpath=str(checkpoint_dir), filename="last", save_last=True, save_top_k=0)

    # 判断计算硬件环境：若 GPU 可用且用户未指定 --cpu，则使用 "gpu" 加速，否则退回 "cpu"
    accelerator = "gpu" if torch.cuda.is_available() and not args.cpu else "cpu"

    # 配置 PyTorch Lightning 训练器（设置最大训练轮数、回调函数、硬件设备及日志记录频率）
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_callback], accelerator=accelerator, devices=1, log_every_n_steps=10)

    # 记录训练起始时间并启动模型训练流程
    start_time = time.time()
    trainer.fit(model, dm)

    # 计算并格式化输出训练总耗时（秒、小时与分钟）
    time_end = time.time() - start_time
    print(f"\nTraining time: {time_end:.2f} s", flush=True)
    minutes, hours = math.modf(time_end / 3600)
    print(f" --> {int(hours)} hours and {int(minutes * 60)} minutes.", flush=True)

    # 除了 Lightning 默认保存的 .ckpt 文件外，额外提取并单独保存生成器与判别器的原生 PyTorch state_dict（方便后续推理）
    torch.save(model.generator.state_dict(), checkpoint_dir / "generator_final.pth")
    torch.save(model.discriminator.state_dict(), checkpoint_dir / "discriminator_final.pth")

    # 打印权重与输出结果的保存目录路径
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Result directory: {result_dir}")

    return model