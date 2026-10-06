from pathlib import Path
import torch
from models import Generator
from utils.visualization import save_generated_grid

def test(args):
    # 根据是否有可用 GPU 且用户未勾选 --cpu 参数，自动选择计算设备（CUDA 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 构建生成器权重文件的完整保存/读取路径
    checkpoint_path = Path(args.checkpoint_dir) / "generator_final.pth"

    # 检查权重文件是否存在，若不存在则提示错误并中断运行
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到生成器权重文件: {checkpoint_path}\n请先运行训练命令：\npython main.py --mode train")

    # 实例化生成器模型：传入潜在空间维度 latent_dim，图像通道数 nc=1（单通道灰度图），并转移到指定计算设备
    generator = Generator(latent_dim=args.latent_dim, nc=1).to(device)

    # 加载已训练好的模型权重参数，并通过 map_location 匹配当前的计算设备
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 将生成器设置为评估/推理模式（关闭 Dropout 和 Batch Normalization 的更新）
    generator.eval()

    # 初始化基于 CPU 的随机数生成器并设置随机种子，确保生成图像的过程具有可复现性
    cpu_rng = torch.Generator().manual_seed(args.seed)

    # 从标准正态分布中采样随机噪声向量 z（形状为 [生成的样本数, 潜在维度]），并送入计算设备
    z = torch.randn(args.num_samples, args.latent_dim, generator=cpu_rng).to(device)

    # 关闭梯度计算以节省内存与计算资源，传入随机噪声向量生成手部 X 光伪造图像
    with torch.no_grad():
        fake_imgs = generator(z)

    # 定义生成的图像网格（图片矩阵）的保存路径
    output_path = Path(args.result_dir) / "test_results" / f"generated_hand_xray_seed_{args.seed}.png"

    # 调用可视化辅助函数，将生成的批量图像拼成指定行数（nrow）的网格并保存为图片
    save_generated_grid(fake_imgs, output_path, nrow=args.nrow)

    # 定义潜在空间噪声向量 z 的保存路径
    latent_path = Path(args.result_dir) / "test_results" / f"latent_vectors_seed_{args.seed}.pt"

    # 递归创建保存结果所需的父级文件夹目录（若已存在则忽略）
    latent_path.parent.mkdir(parents=True, exist_ok=True)

    # 将采样的潜在向量转移至 CPU 并保存为 PyTorch 张量文件（用于后续复现或分析）
    torch.save(z.cpu(), latent_path)

    # 打印测试运行完成后的文件保存信息
    print(f"生成的图像已保存至: {output_path}")
    print(f"潜在向量已保存至 : {latent_path}")