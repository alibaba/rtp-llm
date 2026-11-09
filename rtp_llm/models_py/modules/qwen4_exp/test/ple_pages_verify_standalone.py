"""Standalone byte-exact verification of quantize_pages after the ROWS/PADDED
runtime-args change (mirrors ple_sm90_pages_test.test_native_quant_bytes_and_scales
without the bazel-built native adapter)."""

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_page_projection


def exact(a, b):
    return bool(torch.equal(a, b))


def check(page, rows, ks_val=0.03, vs_val=0.07):
    torch.manual_seed(rows * 31 + page)
    x = (torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 7.0)
    x[::11] *= 900.0
    x[3, 5] = float("nan")
    x[7, 9] = float("inf")
    ks = torch.tensor([ks_val], device="cuda")
    vs = torch.tensor([vs_val], device="cuda")
    quantized, scales = ple_page_projection.quantize_pages(x, ks, vs, page_size=page)
    row_q, row_scales = ple_page_projection.quantize_pages(
        x, ks, vs, page_size=page, row_scales=True
    )
    failures = []
    for i, start in enumerate(range(0, rows, page)):
        end = min(start + page, rows)
        native_q, native_s = scaled_fp8_per_tensor_quant(x[start:end].contiguous())
        if not exact(quantized[start:end].view(torch.uint8), native_q.view(torch.uint8)):
            failures.append((page, rows, i, "bytes"))
        if not exact(scales[0][i : i + 1], native_s):
            failures.append((page, rows, i, "raw_scale"))
        if not exact(scales[1][i : i + 1], native_s * ks):
            failures.append((page, rows, i, "key_scale"))
        if not exact(scales[2][i : i + 1], native_s * vs):
            failures.append((page, rows, i, "value_scale"))
        if not exact(row_q[start:end].view(torch.uint8), native_q.view(torch.uint8)):
            failures.append((page, rows, i, "row_bytes"))
        pad_end = min(start + page, row_q.shape[0])
        if not exact(
            row_scales[1][start:pad_end], (native_s * ks).expand(pad_end - start, 1)
        ):
            failures.append((page, rows, i, "padded_key_scale"))
        if not exact(
            row_scales[2][start:pad_end], (native_s * vs).expand(pad_end - start, 1)
        ):
            failures.append((page, rows, i, "padded_value_scale"))
    if int(torch.count_nonzero(quantized[rows:].view(torch.uint8)).item()) != 0:
        failures.append((page, rows, -1, "padding_nonzero"))
    if int(torch.count_nonzero(row_q[rows:].view(torch.uint8)).item()) != 0:
        failures.append((page, rows, -1, "row_padding_nonzero"))
    return failures


def main():
    all_fail = []
    for page in (128, 192, 64):
        for rows in (
            129,
            min(page - 1, 513),
            min(page + 1, 1089),
            min(2 * page + 3, 2051),
            128 * 11 + 37,
        ):
            if rows < 1:
                continue
            all_fail += check(page, rows)
    # Tiny-value domain (SM90 native scale FTZ boundary).
    for value in (0.0, 1e-38, 1e-36, 5.25e-36, 5.27e-36, 1e-20):
        x = torch.full((193, 2560), value, device="cuda", dtype=torch.bfloat16)
        ks = torch.tensor([0.03], device="cuda")
        vs = torch.tensor([0.07], device="cuda")
        q, s = ple_page_projection.quantize_pages(x, ks, vs, page_size=192)
        nq, ns = scaled_fp8_per_tensor_quant(x[:192].contiguous())
        if not exact(q[:192].view(torch.uint8), nq.view(torch.uint8)):
            all_fail.append(("tiny", value, "bytes"))
        if not exact(s[0][:1], ns):
            all_fail.append(("tiny", value, "scale"))
    if all_fail:
        print("FAILURES:")
        for f in all_fail[:20]:
            print("  ", f)
        raise SystemExit(1)
    print("ALL BYTE-EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
