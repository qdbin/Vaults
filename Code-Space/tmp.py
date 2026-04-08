import torch


# 检查 CUDA 是否可用
ans = torch.cuda.is_available()
print(f"CUDA 是否可用: {ans}")

# 如果有 CUDA，显示版本信息
if ans:
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA 不可用，将使用 CPU")
    print(f"PyTorch 版本: {torch.__version__}")
