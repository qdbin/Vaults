#!/usr/bin/env python3
"""恢复 Harbor 源码到原始状态"""

import subprocess
import sys

HARBOR_PACKAGE = "harbor"

def restore():
    print("正在重新安装 harbor 以恢复原始代码...")
    try:
        # 强制重新安装 harbor
        subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir", HARBOR_PACKAGE], check=True)
        print("✓ Harbor 已恢复到原始状态")
    except subprocess.CalledProcessError as e:
        print(f"✗ 恢复失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    restore()
