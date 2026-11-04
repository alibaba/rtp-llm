# aios/navi_sched/tools/mangle_and_flatten.py
import os
import sys
import shutil
import zipfile
import subprocess
import argparse
import re


def validate_debug_symbols(core_path):
    sections = subprocess.check_output(
        ["readelf", "--wide", "--section-headers", core_path], text=True)
    required = (".symtab", ".debug_info", ".debug_line")
    missing = [name for name in required
               if not re.search(r"\]\s+" + re.escape(name) + r"\s", sections)]
    if missing:
        raise RuntimeError(
            "NaviSched requires perf/GDB symbols; missing %s in %s. "
            "Build with --config=release --strip=never and preserve debug sections."
            % (", ".join(missing), core_path))

def flatten_and_mangle(
        input_wheel, output_wheel, dist_name, wheel_tag):
    temp_dir = "temp_unpacked_wheel"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # 1. 解压 Input Wheel
    with zipfile.ZipFile(input_wheel, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 2. 自动定位 Python 包根目录 (aios/navi_sched 剥离后在 Wheel 里应该叫 navi_sched)
    pkg_root = os.path.join(temp_dir, dist_name)
    if not os.path.exists(pkg_root):
        # 兜底：如果没剥离干净，尝试动态找含有 __init__.py 的目录
        for root, dirs, files in os.walk(temp_dir):
            if (
                "__init__.py" in files
                and root != temp_dir
                and not root.endswith(".dist-info")
            ):
                pkg_root = root
                break

    if not pkg_root or not os.path.exists(pkg_root):
        raise RuntimeError(f"在 Wheel 中找不到包目录: {dist_name}")

    # 3. 递归找出所有的 .so / .pyd 并收集
    all_so_paths = []
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.endswith(".so") or ".so." in f:
                all_so_paths.append(os.path.join(root, f))

    # Keep ELF dependency names intact. The extension already has $ORIGIN in
    # its runpath; rewriting DT_NEEDED with the local patchelf corrupts large
    # binaries by emitting misaligned LOAD segments.
    for old_path in all_so_paths:
        old_filename = os.path.basename(old_path)
        new_path = os.path.join(pkg_root, old_filename)
        if os.path.abspath(old_path) != os.path.abspath(new_path):
            shutil.move(old_path, new_path)

    # 5. 强力清理空文件夹
    for root, dirs, files in os.walk(pkg_root, topdown=False):
        if not dirs and not files:
            os.rmdir(root)

    # Keep local function names and DWARF in the installed ELF for perf and GDB.
    core_path = os.path.join(pkg_root, "_navi_sched_core.so")
    validate_debug_symbols(core_path)

    # ============================================================
    # 7. 核心：修正 _raw 元数据，改回正轨！
    # ============================================================
    old_dist_info = None
    for d in os.listdir(temp_dir):
        if d.endswith(".dist-info"):
            old_dist_info = os.path.join(temp_dir, d)
            break

    if old_dist_info:
        # A. 重命名 .dist-info 目录
        # (例如：navi_sched_raw-0.1.0.dist-info -> navi_sched-0.1.0.dist-info)
        new_dist_info = old_dist_info.replace(dist_name + "_raw", dist_name)
        shutil.move(old_dist_info, new_dist_info)

        # B. 修正 METADATA 文件中的包名
        metadata_path = os.path.join(new_dist_info, "METADATA")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 把 Name: navi_sched_raw 替换为 Name: navi_sched
            content = content.replace(f"Name: {dist_name}_raw", f"Name: {dist_name}")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(content)

        wheel_path = os.path.join(new_dist_info, "WHEEL")
        if os.path.exists(wheel_path):
            with open(wheel_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace(
                "Root-Is-Purelib: true", "Root-Is-Purelib: false")
            content = content.replace("Tag: py3-none-any", f"Tag: {wheel_tag}")
            with open(wheel_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # C. 重新生成 RECORD 文件 (防止 pip 报哈希不匹配)
        record_path = os.path.join(new_dist_info, "RECORD")
        if os.path.exists(record_path):
            os.remove(record_path) # 先删掉旧的

    # 8. 重新打包输出到目标位置
    output_dir = os.path.dirname(output_wheel)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 记录打包进去的所有文件，用于生成完美的 RECORD 文件
    written_files = []
    with zipfile.ZipFile(output_wheel, 'w', zipfile.ZIP_DEFLATED) as zip_write:
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, temp_dir)
                zip_write.write(abs_path, rel_path)
                written_files.append(rel_path)

    # 在最终生成的包内追加 RECORD 记录 (pip 规范：不需要强制写 hash，列出即可)
    with zipfile.ZipFile(output_wheel, 'a', zipfile.ZIP_DEFLATED) as zip_write:
        record_rel_path = os.path.relpath(os.path.join(new_dist_info, "RECORD"), temp_dir)
        record_content = (
            "\n".join([f"{f},," for f in written_files])
            + f"\n{record_rel_path},,"
        )
        zip_write.writestr(record_rel_path, record_content)

    shutil.rmtree(temp_dir)
    print(f"Successfully post-processed and saved to: {output_wheel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dist-name", required=True)
    parser.add_argument("--wheel-tag", required=True)
    args = parser.parse_args()

    flatten_and_mangle(
        args.input,
        args.output,
        args.dist_name,
        args.wheel_tag)
