"""SM100 fault attribution, not a replacement for the two-rank V4 smoke."""

import argparse
import os
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


@contextmanager
def stage(label, device):
    print(f"BEGIN rank={device.index} {label}", flush=True)
    yield
    torch.cuda.synchronize(device)
    print(f"PASS rank={device.index} {label}", flush=True)


def fixture():
    groups, rank, width = 8, 1024, 4096  # actual per-rank wo_a geometry (TP=1)
    rows = groups * rank
    weight_bytes = (
        torch.arange(rows).to(torch.uint8)[:, None].expand(rows, width).contiguous()
    )
    # Normalized, finite UE8M0 values with a distinct row/column phase.
    scale_bytes = (
        torch.arange(rows // 128 * (width // 128)).reshape(rows // 128, width // 128)
        % 17
        + 119
    ).to(torch.uint8)
    return {
        "wo_a.weight": weight_bytes.view(torch.float8_e4m3fn),
        "wo_a.scale": scale_bytes.view(torch.float8_e8m0fnu),
    }


def worker(world_rank, directory, mode):
    device = torch.device("cuda", world_rank)
    torch.cuda.set_device(device)
    if mode == "fenced":
        # Check the effective allocator in both workers, not just Bazel's
        # parent environment. The actual production reader is constructed in
        # exercise(); do not patch or call an alternative copier in this test.
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        if not settings["expandable_segments"]:
            raise AssertionError("V4 fenced prefill requires expandable segments")
    torch.distributed.init_process_group(
        "nccl",
        rank=world_rank,
        world_size=2,
        timeout=timedelta(minutes=3),
        init_method=(Path(directory) / "rendezvous").as_uri(),
    )
    try:
        exercise(device, directory, mode)
    finally:
        torch.distributed.destroy_process_group()


def exercise(device, directory, mode):
    from rtp_llm.models_py.model_loader import (
        NewLoaderConfig,
        NewModelLoader,
        _ExpertRangeFilter,
    )
    from rtp_llm.models_py.module_base import copy_weight_
    from rtp_llm.models_py.modules.dsv4.fp8.attention import _prepare_wo_a_stacked
    from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
    from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor

    groups, rank, width = 8, 1024, 4096
    reference = fixture()
    owner = {name: torch.empty_like(t, device=device) for name, t in reference.items()}
    paths = sorted(str(p) for p in Path(directory).glob("*.safetensors"))
    assert len(paths) == 2
    with stage("setup", device):
        loader = NewModelLoader(
            SimpleNamespace(model_type="deepseek_v4"),
            NewLoaderConfig(device=str(device)),
            model_path=directory,
        )
        expert_filter = _ExpertRangeFilter(256, 2, device.index)
    with stage("dispatch-and-close-" + mode, device):
        seen = set()
        iterator = loader._fastsafetensors_weights_iterator(
            paths, str(device), None, expert_filter, ReadSpec.full
        )
        try:
            for name, checkpoint in iterator:
                source, _ = unwrap_checkpoint_tensor(checkpoint)
                if name in seen or source.device != device:
                    raise AssertionError(
                        "Unexpected duplicate or non-device FST tensor"
                    )
                seen.add(name)
                if mode == "fenced":
                    with stage("consumer-copy-" + name, device):
                        copy_weight_(owner[name], source, name)
                else:
                    copy_weight_(owner[name], source, name)
                del checkpoint, source
        finally:
            iterator.close()
        assert seen == set(reference)

    with stage("owned-bytes-after-reader-close", device):
        for name in reference:
            assert torch.equal(
                owner[name].view(torch.uint8).cpu(), reference[name].view(torch.uint8)
            ), name
    with stage("grouped-wo-a-scale-packing", device):
        weight, scale = _prepare_wo_a_stacked(
            owner["wo_a.weight"], owner["wo_a.scale"], groups, rank, width
        )
    # Check the numerical byte layout independently of DeepGEMM's packer.
    with stage("packed-output-byte-check", device):
        expected = (
            reference["wo_a.scale"]
            .view(torch.uint8)
            .repeat_interleave(128, dim=0)
            .reshape(groups, rank, width // 128)
        )
        assert weight.data_ptr() == owner["wo_a.weight"].data_ptr()
        assert scale.dtype == torch.int32
        words = width // 512
        assert scale.shape == (groups, rank, words)
        assert scale.stride() == (rank * words, 1, rank)
        assert torch.equal(scale.cpu().contiguous().view(torch.uint8), expected)
    if mode == "fenced":
        # Independent finite synthetic input, known to run the v2.6 SM100
        # pack kernel on this isolated fixture. Compare the actual kernel's
        # result and storage contract, not just our own byte oracle.
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        with stage("native-ue8m0-layout-reference", device):
            grid = (
                owner["wo_a.scale"].float().reshape(groups, rank // 128, width // 128)
            )
            native = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                grid.repeat_interleave(128, dim=1)
            )
            assert native.dtype == scale.dtype and native.stride() == scale.stride()
            assert torch.equal(native, scale)
        with stage("native-ue8m0-exponent-edges", device):
            # The pinned CUDA packer asserts FP32 scales have a zero sign
            # and mantissa. Probe all 256 checkpoint bytes with our raw
            # layout, but compare *only* normalized powers to the native
            # packer; 0 decodes to a subnormal and 255 decodes to NaN.
            probe = torch.arange(grid.numel(), device=device).to(torch.uint8)
            probe = probe.reshape_as(owner["wo_a.scale"]).view(torch.float8_e8m0fnu)
            _, direct = _prepare_wo_a_stacked(
                owner["wo_a.weight"], probe, groups, rank, width
            )
            direct_bytes = direct.contiguous().view(torch.uint8)
            probe_bytes = (
                probe.view(torch.uint8).reshape_as(grid).repeat_interleave(128, dim=1)
            )
            assert torch.equal(direct_bytes, probe_bytes)
            finite = probe.view(torch.uint8).clamp(1, 254).view(torch.float8_e8m0fnu)
            _, direct = _prepare_wo_a_stacked(
                owner["wo_a.weight"], finite, groups, rank, width
            )
            probe_fp32 = finite.float().reshape_as(grid).repeat_interleave(128, dim=1)
            native = get_mn_major_tma_aligned_packed_ue8m0_tensor(probe_fp32)
            assert native.stride() == direct.stride() and torch.equal(native, direct)
    # Isolate the original H2D site with the actual rotary implementation.
    # A short context is enough to exercise the copy, not to test HBM limits.
    with stage("rope-table-cpu-to-device", device):
        args = (64, 4096, 4096, 10000.0, 1.0, 32, 1)
        reference_cis = precompute_freqs_cis(*args)
        device_cis = precompute_freqs_cis(*args, device=device)
    assert torch.equal(device_cis.cpu(), reference_cis)


def main(mode):
    if os.environ.get("FASTSAFETENSORS_NOGDS") != "1":
        raise RuntimeError("This fixture requires the original no-GDS reader")
    if mode != "fixture" and (
        torch.cuda.device_count() < 2
        or any(torch.cuda.get_device_capability(i)[0] != 10 for i in (0, 1))
    ):
        raise RuntimeError("Two SM100-family GPUs required")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as directory:
        for idx, (name, value) in enumerate(fixture().items()):
            path = Path(directory) / f"model-{idx + 1:05}-of-00002.safetensors"
            save_file({name: value}, str(path))
            if mode == "fixture":
                from safetensors.torch import load_file

                actual = load_file(str(path))[name]
                assert actual.dtype == value.dtype and torch.equal(
                    actual.view(torch.uint8), value.view(torch.uint8)
                )
        if mode == "fixture":
            print("PASS CPU synthetic checkpoint round trip", flush=True)
            return
        torch.multiprocessing.spawn(worker, args=(directory, mode), nprocs=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ordinary", "fenced", "fixture"))
    main(parser.parse_args().mode)
