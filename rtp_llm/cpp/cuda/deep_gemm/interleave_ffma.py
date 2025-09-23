import argparse
import mmap
import os
import re
import subprocess

CUDA_HOME = "/usr/local/cuda"


def run_cuobjdump(file_path):
    command = [f"{CUDA_HOME}/bin/cuobjdump", "-sass", file_path]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert result.returncode == 0
    return result.stdout


def extract_ffma(sass):
    lines = sass.splitlines()
    collected = []
    current = []

    arch_name, func_name = "N/A", "N/A"
    skip_next_line = False
    for line in lines:
        if "code for" in line:
            arch_name = line.lstrip().lstrip("code for ").rstrip()
        elif "Function :" in line:
            func_name = line.lstrip().lstrip("Function :").rstrip()
        elif "FFMA" in line:
            current.append(line)
            skip_next_line = True
        elif skip_next_line:
            current.append(line)
            skip_next_line = False
        else:
            if len(current) >= 16:
                assert len(current) % 2 == 0
                collected.append((f"{arch_name}::{func_name}", current))
            current = []

    ## reserve this env
    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(f"Found {len(collected)} FFMA segments")
    return collected


def extract_hex_from_line(line):
    match = re.search(r"/\*\s*(0x[0-9a-fA-F]+)\s*\*/", line)
    assert match
    return int(match.group(1), 16)


def validate(m, offset, le_bytes, num_lines):
    assert len(le_bytes) == num_lines // 2
    assert m[offset : offset + 16] == le_bytes[0]
    for i in range(1, num_lines // 2):
        if m[offset + i * 16 : offset + i * 16 + 16] != le_bytes[i]:
            return False
    return True


def parse_registers(line):
    import re

    line = re.sub(r"/\*.*?\*/", "", line)
    line = line.replace(";", "")
    tokens = line.strip().split(",")
    registers = []
    for token in tokens:
        token = token.strip()
        words = token.split()
        for word in words:
            if word.startswith("R"):
                reg = word.split(".")[0]
                registers.append(reg)
    return registers


def modify_segment(m, name, ffma_lines):
    num_lines = len(ffma_lines)
    assert num_lines % 2 == 0

    le_bytes, new_le_bytes = [], []
    reused_list = []
    dst_reg_set = set()
    last_reused, last_dst_reg = False, ""
    num_changed = 0
    for i in range(num_lines // 2):
        dst_reg = parse_registers(ffma_lines[i * 2])[-2]
        low_line, high_line = ffma_lines[i * 2], ffma_lines[i * 2 + 1]
        low_hex, high_hex = extract_hex_from_line(low_line), extract_hex_from_line(
            high_line
        )
        le_bytes.append(low_hex.to_bytes(8, "little") + high_hex.to_bytes(8, "little"))
        reused = (high_hex & 0x0800000000000000) != 0
        if reused:
            is_first_occurred = dst_reg not in dst_reg_set
            if is_first_occurred or (last_reused and dst_reg == last_dst_reg):
                # Modify the `reuse` and `yield` bits
                assert high_hex & 0x0800200000000000, f"{hex(high_hex)}"
                high_hex ^= 0x0800200000000000
                reused = False
                num_changed += 1
            else:
                reused_list.append(i)
        dst_reg_set.add(dst_reg)
        new_le_bytes.append(
            low_hex.to_bytes(8, "little") + high_hex.to_bytes(8, "little")
        )
        last_reused, last_dst_reg = reused, dst_reg
    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(
            f" > segment `{name}` new reused list ({num_changed} changed): {reused_list}"
        )

    # Find the offset
    offsets = []
    offset = m.find(le_bytes[0])
    while offset != -1:
        offsets.append(offset)
        offset = m.find(le_bytes[0], offset + 1)
    offsets = list(filter(lambda x: validate(m, x, le_bytes, num_lines), offsets))

    # Replace with `new_le_bytes`
    for offset in offsets:
        for i in range(num_lines // 2):
            m[offset + i * 16 : offset + i * 16 + 16] = new_le_bytes[i]


def process(path):
    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(f"Processing {path}")
    output = run_cuobjdump(path)
    segments = extract_ffma(output)
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        for segment in segments:
            modify_segment(mm, *segment)
        mm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interleave FFMA reg reuse")
    parser.add_argument("--so", help="Path to the SO file")
    args = parser.parse_args()

    process(args.so)
