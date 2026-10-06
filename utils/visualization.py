from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

def save_real_vs_synthetic(real_imgs, fake_imgs, output_path, num_images=4):
    """保存真实样本与伪造生成样本的上下对比画板（第一行为真实 X 光图像，第二行为生成器生成的 X 光图像）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 确定实际展示的图片数量，防止索引溢出
    num_images = min(num_images, len(real_imgs), len(fake_imgs))

    fig = plt.figure(figsize=(8, 4))
    for i in range(num_images):
        # 绘制第 1 行：真实样本 (Real)
        plt.subplot(2, num_images, i + 1)
        plt.axis("off")
        plt.title("Real")
        plt.imshow(real_imgs[i, 0].detach().cpu(), cmap="gray", interpolation="none")

        # 绘制第 2 行：伪造生成样本 (Synthetic)
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.axis("off")
        plt.title("Synthetic")
        plt.imshow(fake_imgs[i, 0].detach().cpu(), cmap="gray", interpolation="none")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def save_generated_grid(fake_imgs, output_path, nrow=8):
    """将批量生成的伪造图像拼接为标准化网格矩阵并直接保存为图片文件（自动将输入数值范围 [-1, 1] 映射恢复到 [0, 1]）。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 调用 torchvision 提供的 save_image 保存排版网格图
    save_image(fake_imgs.detach().cpu(), output_path, nrow=nrow, normalize=True, value_range=(-1, 1))

def save_loss_curve(g_losses, d_losses, output_path):
    """绘制并保存生成器 (Generator) 与判别器 (Discriminator) 随训练 Epoch 变化的 Loss 损失曲线图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g_losses, label="Generator Loss")       # 绘制生成器损失折线
    ax.plot(d_losses, label="Discriminator Loss")   # 绘制判别器损失折线
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Losses")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)