from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.generator import Generator
from models.discriminator import Discriminator
from utils.visualization import save_generated_grid, save_loss_curve, save_real_vs_synthetic

class GAN(pl.LightningModule):
    """PyTorch Lightning GAN 封装模块，管理网络结构、手动交替优化过程以及可视化监控结果的导出。"""

    def __init__(self, channels, width, height, latent_dim: int = 100, lr: float = 0.0002, b1: float = 0.5, b2: float = 0.999, batch_size: int = 128, result_dir: str = "./results", **kwargs):
        super().__init__()
        # 自动将所有传入参数保存至 self.hparams，方便后续调用
        self.save_hyperparameters()
        # 禁用 Lightning 的自动优化机制，以精细控制生成器与判别器的交替手动更新流程
        self.automatic_optimization = False

        # 初始化生成器与判别器网络模型
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        # 固定一组验证用的随机噪声，用于在各个 Epoch 中观察固定输入下的图像生成质感演变
        self.validation_z = torch.randn(64, self.hparams.latent_dim)
        # 用于模型结构可视化或计算计算图的示例输入张量
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

        # 记录每个 Epoch 均值损失的列表（用于绘制整体 Loss 曲线）
        self.g_losses, self.d_losses = [], []
        # 临时累加器（用于收集单个 Epoch 内每个 Step 的损失值）
        self._g_loss_accumulator, self._d_loss_accumulator = [], []
        self.result_dir = Path(result_dir)

    def forward(self, z):
        # 前向传播：传入随机噪声向量 z，输出生成的伪造图像
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        # 对抗损失计算：y_hat 为判别器预测概率，y 为真实/伪造目标标签（使用二元交叉熵二分类损失）
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        # 从当前批次数据中提取真实图像，并获取优化器对象（opt_g: 生成器优化器, opt_d: 判别器优化器）
        real_imgs = batch["image"]
        opt_g, opt_d = self.optimizers()

        # 采样对应当前批次大小的潜在空间随机噪声
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim, device=self.device)

        # ------------------------------------------------------------------
        # 训练生成器 (Generator)：目标是骗过判别器，使 D(G(z)) 接近 全1 标签
        # ------------------------------------------------------------------
        self.toggle_optimizer(opt_g)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        y = torch.ones(real_imgs.size(0), 1, device=self.device)  # 全1 真实目标标签
        g_loss = self.adversarial_loss(y_hat, y)

        # 生成器手动反向传播与参数更新
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # ------------------------------------------------------------------
        # 训练判别器 (Discriminator)：目标是区分真实图像（标签1）与伪造图像（标签0）
        # ------------------------------------------------------------------
        self.toggle_optimizer(opt_d)
        # 计算真实图像损失
        y_hat_real = self.discriminator(real_imgs)
        y_real = torch.ones(real_imgs.size(0), 1, device=self.device)
        real_loss = self.adversarial_loss(y_hat_real, y_real)

        # 计算生成/伪造图像损失（通过 .detach() 切断生成器梯度更新）
        y_hat_fake = self.discriminator(self(z).detach())
        y_fake = torch.zeros(real_imgs.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        # 判别器总损失为真实损失与伪造损失之和
        d_loss = real_loss + fake_loss

        # 判别器手动反向传播与参数更新
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # 记录并累加当前批次的损失值
        self._g_loss_accumulator.append(g_loss.item())
        self._d_loss_accumulator.append(d_loss.item())

        # 在进度条和日志中打印当前 Epoch 的平均损失
        self.log("g_loss", g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": d_loss + g_loss, "g_loss": g_loss.detach(), "d_loss": d_loss.detach()}

    def configure_optimizers(self):
        # 配置生成器和判别器的 Adam 优化器
        lr, b1, b2 = self.hparams.lr, self.hparams.b1, self.hparams.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        # 每一个训练 Epoch 结束后触发：计算当前 Epoch 的平均 Loss 并存入历史数据
        avg_g_loss = float(np.mean(self._g_loss_accumulator))
        avg_d_loss = float(np.mean(self._d_loss_accumulator))
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        self._g_loss_accumulator.clear()
        self._d_loss_accumulator.clear()

        print(f"[Epoch {self.current_epoch + 1}] | Generator Loss: {avg_g_loss:.6f} | Discriminator Loss: {avg_d_loss:.6f}")

        # 将固定的验证噪声向量转移到与生成器相同的设备与数据类型上
        z = self.validation_z.type_as(self.generator.main[0].weight)

        # 切换到评估模式导出当前 Epoch 固定的样本图像
        self.generator.eval()
        with torch.no_grad():
            sample_imgs = self(z).detach().cpu()
        self.generator.train()

        # 保证结果输出目录存在，将当前生成的 64 张采样图保存为拼图网格
        generated_dir = self.result_dir / "generated_images"
        generated_dir.mkdir(parents=True, exist_ok=True)
        save_generated_grid(sample_imgs, generated_dir / f"epoch_{self.current_epoch + 1:03d}.png", nrow=8)

        # 从 DataLoader 中获取真实图像，并保存 4 组“真实 vs 伪造”对比图
        real_batch = next(iter(self.trainer.datamodule.train_dataloader()))
        real_imgs = real_batch["image"][:4]
        save_real_vs_synthetic(real_imgs, sample_imgs[:4], generated_dir / f"compare_epoch_{self.current_epoch + 1:03d}.png", num_images=4)

        # 实时更新并绘制全局训练 Loss 变化曲线图
        save_loss_curve(self.g_losses, self.d_losses, self.result_dir / "loss_curve.png")