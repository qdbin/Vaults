#!/usr/bin/env python3
"""
修复 Harbor 在 Windows 上的编码问题
由于 Windows 默认使用 GBK 编码，而 Harbor 的代码没有显式指定 UTF-8，
导致读取包含非 ASCII 字符的文件时出错。
"""

import os
import sys
from pathlib import Path

# Harbor 安装路径
HARBOR_PATH = Path("C:/Users/hanbin/AppData/Roaming/uv/tools/harbor/Lib/site-packages/harbor")

def fix_file_encoding(file_path: Path, replacements: list):
    """修复文件中的编码问题"""
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return False
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"已修复: {file_path}")
        return True
    else:
        print(f"无需修复: {file_path}")
        return False

def main():
    # 1. 修复 task.py
    task_py = HARBOR_PATH / "models/task/task.py"
    fix_file_encoding(task_py, [
        ("self.paths.instruction_path.read_text()", "self.paths.instruction_path.read_text(encoding='utf-8')"),
        ("self.paths.config_path.read_text()", "self.paths.config_path.read_text(encoding='utf-8')"),
    ])
    
    # 2. 修复 publisher.py
    publisher_py = HARBOR_PATH / "publisher/publisher.py"
    fix_file_encoding(publisher_py, [
        ("paths.config_path.read_text()", "paths.config_path.read_text(encoding='utf-8')"),
        ("paths.instruction_path.read_text()", "paths.instruction_path.read_text(encoding='utf-8')"),
        ("paths.readme_path.read_text()", "paths.readme_path.read_text(encoding='utf-8')"),
    ])
    
    # 3. 修复 packager.py
    packager_py = HARBOR_PATH / "publisher/packager.py"
    fix_file_encoding(packager_py, [
        ("paths.gitignore_path.read_text()", "paths.gitignore_path.read_text(encoding='utf-8')"),
    ])
    
    # 4. 修复 mappers/terminal_bench.py
    tb_mapper_py = HARBOR_PATH / "mappers/terminal_bench.py"
    fix_file_encoding(tb_mapper_py, [
        ("dockerfile_path.read_text()", "dockerfile_path.read_text(encoding='utf-8')"),
        ("source.read_text()", "source.read_text(encoding='utf-8')"),
        ("compose_path.read_text()", "compose_path.read_text(encoding='utf-8')"),
        ("task_paths.test_path.read_text()", "task_paths.test_path.read_text(encoding='utf-8')"),
    ])
    
    # 5. 修复 templating.py
    templating_py = HARBOR_PATH / "utils/templating.py"
    fix_file_encoding(templating_py, [
        ("template_path.read_text()", "template_path.read_text(encoding='utf-8')"),
    ])
    
    # 6. 修复 tasks/client.py
    tasks_client_py = HARBOR_PATH / "tasks/client.py"
    fix_file_encoding(tasks_client_py, [
        ("gitattributes.read_text()", "gitattributes.read_text(encoding='utf-8')"),
    ])
    
    print("\n修复完成！")
    print("请重新运行 harbor 命令测试。")

if __name__ == "__main__":
    main()
