#!/usr/bin/env python3
"""
Sphinx 多语言文档构建脚本
支持构建中文和英文版本的文档
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# 支持的语言列表
LANGUAGES = ['en', 'zh_CN']
DEFAULT_LANGUAGE = 'en'

def run_command(cmd, cwd=None, env=None):
    """执行命令并返回结果"""
    print(f"执行命令: {cmd}")
    if env:
        print(f"环境变量: {env}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        return False
    return True

def update_translations():
    """更新翻译文件（提取消息并更新 PO 文件）"""
    print("正在更新翻译文件...")
    
    # 1. 提取可翻译消息
    if not run_command("sphinx-build -b gettext . build/gettext"):
        print("提取消息失败")
        return False
    
    # 2. 确保 locales 目录存在
    locales_dir = Path("locales")
    locales_dir.mkdir(exist_ok=True)
    
    # 3. 更新各语言的翻译文件
    for lang in LANGUAGES:
        if lang == DEFAULT_LANGUAGE:
            continue
            
        # 使用 sphinx-intl 更新翻译文件
        cmd = f"sphinx-intl update -p build/gettext -l {lang}"
        if not run_command(cmd):
            print(f"更新 {lang} 翻译文件失败")
            return False
    
    print("翻译文件更新完成!")
    return True

def build_language(lang):
    """构建指定语言的文档"""
    print(f"正在构建 {lang} 版本的文档...")
    
    # 获取输出目录
    output_dir = f"build/{lang}"
    
    cmd = f"sphinx-build -b html -D language='{lang}' . {output_dir}"
    success = run_command(cmd)
    
    if success:
        print(f"{lang} 版本文档已生成到: {output_dir}")
    
    return success

def build_all():
    """构建所有语言版本"""
    print("开始构建多语言文档...")
    
    # 1. 更新翻译文件
    if not update_translations():
        print("更新翻译文件失败")
        return False
    
    # 2. 构建各语言版本
    for lang in LANGUAGES:
        if not build_language(lang):
            print(f"构建 {lang} 版本失败")
            return False
    
    
    print("多语言文档构建完成!")
    print("输出结构:")
    print("  build/")
    print("  ├── en/")
    print("  ├── zh_CN/")
    print("  └── gettext/")
    return True

def clean():
    """清理构建文件"""
    print("清理构建文件...")
    dirs_to_clean = ["build", "locales"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"已删除 {dir_name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            clean()
        elif sys.argv[1] == "update":
            update_translations()
        elif sys.argv[1] in LANGUAGES:
            build_language(sys.argv[1])
        else:
            print(f"未知命令: {sys.argv[1]}")
            print("可用命令: clean, update, en, zh_CN")
    else:
        build_all()