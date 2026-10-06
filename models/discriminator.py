import torch.nn as nn

class Discriminator(nn.Module):
    """判别器网络架构（侦探模型）：接收单通道 64x64 图像，通过下采样提取特征，最终输出一个介于 [0, 1] 之间的概率值，用于判断输入图像是真图（接近 1）还是伪造图（接近 0）。"""

    def __init__(self, nc=1):
        super(Discriminator, self).__init__()
        # 定义主前向传播网络 Sequential
        self.main = nn.Sequential(
            # 第1层：输入图片尺寸为 nc x 64 x 64（对于单通道图像 nc=1），通过跨步卷积下采样至 128 x 32 x 32（第一层通常不加 BatchNorm）
            nn.Conv2d(in_channels=nc, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 第2层：输入 128 x 32 x 32，下采样至 256 x 16 x 16
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第3层：输入 256 x 16 x 16，下采样至 512 x 8 x 8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 第4层：输入 512 x 8 x 8，下采样至 1024 x 4 x 4
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # 第5层（输出层）：输入 1024 x 4 x 4，卷积压缩至 1 x 1 x 1 特征标量
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            # 使用 Sigmoid 激活函数将预测输出限制在 [0, 1] 之间的概率区间
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        # 将输出形状重构展平为二维矩阵 [batch_size, 1]，方便后续与目标标签计算二元交叉熵损失
        return output.view(-1, 1)