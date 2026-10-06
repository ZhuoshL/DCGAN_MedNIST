import argparse
from train import train
from test import test


def build_parser():
    # 创建命令行参数解析器对象，并设置程序描述文本
    parser = argparse.ArgumentParser(description="MedNIST Hand X-ray DCGAN")

    # 核心运行模式设置：只能从 ['train', 'test'] 中选择，默认为测试模式
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="选择运行模式：train（训练模型）或 test（生成图像）")

    # 路径配置参数
    parser.add_argument("--data-dir", default="./data", help="训练数据集所在的目录路径")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="保存和读取模型权重检查点（Checkpoint）的目录")
    parser.add_argument("--result-dir", default="./results", help="测试时生成的图像结果保存目录")

    # 训练超参数设置
    parser.add_argument("--epochs", type=int, default=100, help="模型总共训练的轮数（Epochs）")
    parser.add_argument("--batch-size", type=int, default=128, help="单次训练/测试批次中的样本数量")
    parser.add_argument("--latent-dim", type=int, default=100, help="输入生成器的随机噪声（Latent Vector）维度")
    parser.add_argument("--lr", type=float, default=0.0002, help="优化器（如Adam）的学习率（Learning Rate）")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam优化器的一阶矩估计指数衰减率（β1）")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam优化器的二阶矩估计指数衰减率（β2）")

    # 系统与环境控制参数
    parser.add_argument("--workers", type=int, default=2, help="DataLoader 数据加载时使用的子进程数量")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子，用于固定随机状态以确保实验可复现")

    # 图像生成与排版控制（用于测试模式）
    parser.add_argument("--num-samples", type=int, default=64, help="测试阶段批量生成手部X光图像的总数量")
    parser.add_argument("--nrow", type=int, default=8, help="拼图网格中每一行显示的图像数量（64张图按8x8网格排列）")

    # 设备选择开关：不带参数时默认值为 False，加上 --cpu 标记时触发为 True
    parser.add_argument("--cpu", action="store_true", help="显式指定仅使用 CPU 运算（即使系统存在可用 GPU）")

    return parser


if __name__ == "__main__":
    # 解析终端/命令行传入的实际参数
    args = build_parser().parse_args()

    # 根据指定的 mode 参数判断执行训练逻辑还是测试/生成逻辑
    if args.mode == "train":
        train(args)  # 调用 train.py 中的训练入口函数
    else:
        test(args)  # 调用 test.py 中的测试/生成入口函数