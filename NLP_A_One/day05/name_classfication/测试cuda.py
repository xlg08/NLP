import torch

# 需要合适的驱动（NVIDIA Driver）和 CUDA 工具包才能让 cuda 正常工作。


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 查看当前系统中可用的 GPU 块（即 GPU 的数量）
gpu_count = torch.cuda.device_count()
print(f"可用的GPU数量：{gpu_count}")

# 列出每个 GPU 的名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 查看当前默认 GPU（如果有）
if torch.cuda.is_available():       # 检测当前系统是否支持并启用了 CUDA（即 NVIDIA GPU 支持）。
    print(f"当前默认GPU编号：{torch.cuda.current_device()}")
    print(f"当前默认GPU名称：{torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("没有可用的GPU")

print(torch.version.cuda)   # 输出 CUDA 版本，如 '12.1'


