import os, tempfile
import pytorch_lightning as pl
from monai import transforms
from monai.apps import MedNISTDataset
from monai.data import CacheDataset, DataLoader

class MedNISTDataModule(pl.LightningDataModule):
    """基于 MONAI 和 PyTorch Lightning 的 MedNIST 数据模块，负责自动下载、预处理、缓存及构造数据加载器。"""

    def __init__(self, batch_size=64, num_workers=2, class_names=None, root_dir=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = class_names if class_names else ["Hand"]  # 默认仅筛选加载 "Hand"（手部 X 光）类别
        self.dims = (1, 64, 64)  # 定义图像的形状为：1 个通道（灰度图），分辨率为 64x64 像素

        # 设置数据集根目录；未指定时优先读取环境变量 MONAI_DATA_DIRECTORY，若也未设置则创建临时文件夹
        self.root_dir = root_dir or os.environ.get("MONAI_DATA_DIRECTORY", tempfile.mkdtemp())
        os.makedirs(self.root_dir, exist_ok=True)

        # 定义图像数据增强与预处理流水线 Pipeline
        self.transforms = transforms.Compose([
            transforms.LoadImaged(keys=["image"]),                               # 读取图像文件
            transforms.EnsureChannelFirstd(keys=["image"]),                      # 确保图像通道位于首位，符合 PyTorch 的 (C, H, W) 格式
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True), # 将像素值从 [0, 255] 线性归一化归一至 [-1, 1]，匹配生成器 Tanh 激活函数的输出区间
            transforms.Rotate90d(keys=["image"], k=3),                          # 旋转 270 度（逆时针 90 度 3 次）以纠正图像显示方向
        ])

    def prepare_data(self):
        """准备数据阶段：若本地不存在 MedNIST 数据集文件夹，则自动触发下载（该过程仅在主进程中执行一次）。"""
        if not os.path.exists(os.path.join(self.root_dir, "MedNIST")):
            MedNISTDataset(root_dir=self.root_dir, section="training", download=True)

    def setup(self, stage=None):
        """设置数据阶段：根据类别要求过滤样本，并使用 CacheDataset 预先加载并缓存预处理好的数据到内存中，以大幅加速后续训练。"""
        if stage == "fit" or stage is None:
            train_data = MedNISTDataset(root_dir=self.root_dir, section="training", download=False, seed=0)

            # 根据指定类别名称（如 "Hand"）提取并过滤样本列表
            train_datalist = [{"image": item["image"], "label": item["label"]} for item in train_data.data if item["class_name"] in self.class_names]

            # 构造带内存缓存特性的训练数据集对象，利用多进程提前处理并缓存数据
            self.train_ds = CacheDataset(data=train_datalist, transform=self.transforms, num_workers=self.num_workers)

    def train_dataloader(self):
        """创建训练集的 DataLoader 加载器（支持多进程加载与数据打乱）。"""
        # 当工作线程数 num_workers > 0 时开通 persistent_workers 参数，保持多进程通道常驻以避免每个 Epoch 重建线程开销
        persistent_workers = self.num_workers > 0
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=persistent_workers)

    def val_dataloader(self):
        # 验证集数据加载器接口（GAN 训练无须显式验证集，留空）
        pass

    def test_dataloader(self):
        # 测试集数据加载器接口（留空）
        pass