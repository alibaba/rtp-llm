"""Compare the fixture with pinned upstream CT eager source functions.

This opt-in audit downloads a hash-pinned wheel, evaluates selected reference
functions without compile decorators, and does not import the full package.
It is not a native kernel or original-BF16 accuracy test.
"""

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch

URL = "https://files.pythonhosted.org/packages/c9/86/b3d9204e42969832779c7e122b406ccc399e43e9e365ae8b925e0ab89e48/compressed_tensors-0.11.1a20250917-py3-none-any.whl"
SHA256 = "b0edd61c7d2beb8ec956250ddd7525d3de2f5016105c355029543e64ed11c9a1"
PACK = "compressed_tensors/compressors/quantized_compressors/nvfp4_quantized.py"
FORWARD = "compressed_tensors/quantization/lifecycle/forward.py"
BASE_COMPRESSOR = "compressed_tensors/compressors/quantized_compressors/base.py"
ROOT = Path(__file__).resolve().parents[1]


def extract(source, wanted, namespace):
    """Execute only named definitions, with unchanged bodies, eagerly."""
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        ):
            selected.append(node)
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), "pinned_CT_reference", "exec"
        ),
        namespace,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, required=True, help="Downloaded wheel/cache path"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.wheel.exists():
        args.wheel.write_bytes(urllib.request.urlopen(URL, timeout=30).read())
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Upstream wheel checksum mismatch")
    with zipfile.ZipFile(args.wheel) as archive:
        packing_source = archive.read(PACK).decode()
        forward_source = archive.read(FORWARD).decode()
        base_source = archive.read(BASE_COMPRESSOR).decode()
    namespace = dict(torch=torch, Tensor=torch.Tensor, Optional=Optional)
    extract(
        packing_source,
        {"FLOAT_TO_E2M1", "kE2M1ToFloat", "pack_fp4_to_uint8", "unpack_fp4_from_uint8"},
        namespace,
    )
    extract(forward_source, {"_dequantize"}, namespace)
    tile = json.loads(
        (ROOT / "rtp_llm/config/test/fixtures/nvfp4/q_proj_tile.json").read_text()
    )
    packed = torch.tensor(
        list(base64.b64decode(tile["weight_packed"])), dtype=torch.uint8
    ).reshape(128, 64)
    scales = (
        torch.tensor(list(base64.b64decode(tile["weight_scale"])), dtype=torch.uint8)
        .reshape(128, 8)
        .view(torch.float8_e4m3fn)
    )
    unpacked = namespace["unpack_fp4_from_uint8"](packed, 128, 128, dtype=torch.float32)
    # This grouping mirrors upstream _process_quantization's GROUP branch;
    # _dequantize itself is executed directly from the pinned wheel.
    global_scale = torch.tensor(tile["weight_global_scale"])
    reference = namespace["_dequantize"](
        unpacked.unflatten(-1, (8, 16)), scales.unsqueeze(-1), global_scale=global_scale
    ).flatten(-2)
    spec = importlib.util.spec_from_file_location(
        "adapter_scales", ROOT / "rtp_llm/config/nvfp4_utils.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    values = lut[codes & 7] * torch.where((codes & 8) != 0, -1, 1)
    actual = (
        values
        * scales.float().repeat_interleave(16, -1)
        * adapter.reciprocal_global_scale([global_scale])
    )
    torch.testing.assert_close(actual, reference, rtol=2e-7, atol=1e-10)
    all_bytes = torch.arange(256).to(torch.uint8).reshape(1, 256)
    upstream_values = namespace["unpack_fp4_from_uint8"](
        all_bytes, 1, 512, dtype=torch.float32
    )
    torch.testing.assert_close(
        namespace["pack_fp4_to_uint8"](upstream_values), all_bytes, rtol=0, atol=0
    )
    # Known nonsymmetric pair establishes which nibble means even K.
    known = torch.tensor([[0.5, 6.0]])
    assert namespace["pack_fp4_to_uint8"](known).item() == 0x71
    wrong = torch.stack((packed >> 4, packed & 15), dim=-1).flatten(-2).long()
    wrong_values = lut[wrong & 7] * torch.where((wrong & 8) != 0, -1, 1)
    assert not torch.equal(wrong_values, unpacked)
    report = dict(
        status="PASS",
        package="compressed-tensors",
        version="0.11.1a20250917",
        wheel_url=URL,
        wheel_sha256=SHA256,
        sources={
            PACK: hashlib.sha256(packing_source.encode()).hexdigest(),
            FORWARD: hashlib.sha256(forward_source.encode()).hexdigest(),
            BASE_COMPRESSOR: hashlib.sha256(base_source.encode()).hexdigest(),
        },
        mode="selected_upstream_source_functions_eager_without_compile_decorators",
        checks=dict(
            real_tile_matches_upstream_dequantization=True,
            all_256_packed_codes_roundtrip=True,
            asymmetric_pair_even_k_low_nibble=True,
            wrong_nibble_order_detected=True,
        ),
        max_abs_error=(actual - reference).abs().max().item(),
        source_contracts=dict(
            scale_layout="GROUP branch unflattens last dimension; scale.unsqueeze(-1), no swizzle",
            scale_storage="BaseQuantizationCompressor.compress preserves non-weight scale tensors",
            global_scale="_dequantize divides local scale by global_scale",
        ),
        limitation="Not full-package import, native GEMM, or original-BF16 quality validation",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
