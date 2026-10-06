import torch.nn as nn

class Generator(nn.Module):
    """生成器网络架构：将随机噪声向量逐步通过转置卷积（解卷积）上采样，最终生成符合真实数据分布的单通道 64x64 手部 X 光伪造图像，输出数值范围介于 [-1, 1]。"""

    def __init__(self, latent_dim, nc=1):
        super(Generator, self).__init__()
        # 定义主前向传播网络 Sequential
        self.main = nn.Sequential(
            # 第1层：输入潜在空间向量 Z [latent_dim, 1, 1]，上采样至特征图尺寸 1024 x 4 x 4
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 第2层：输入 1024 x 4 x 4，解卷积上采样至 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 第3层：输入 512 x 8 x 8，解卷积上采样至 256 x 16 x 16
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 第4层：输入 256 x 16 x 16，解卷积上采样至 128 x 32 x 32
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 第5层（输出层）：输入 128 x 32 x 32，解卷积上采样至目标尺寸 nc x 64 x 64（对于单通道图像 nc=1，尺寸为 1x64x64）
            nn.ConvTranspose2d(in_channels=128, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            # 使用 Tanh 激活函数将像素输出值压缩限定在 [-1, 1] 范围内，与数据预处理保持一致
            nn.Tanh(),
        )

    def forward(self, input):
        # 将形状为 [batch_size, latent_dim] 的二维输入张量重构为四维张量 [batch_size, latent_dim, 1, 1]，满足 2D 转置卷积的输入维度要求
        input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)