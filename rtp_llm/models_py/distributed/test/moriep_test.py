import importlib.util
import os
import pathlib
import socket
import sys
import unittest

import mori
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 通过 rtp-llm 路径加载 moriep_wrapper 需要整体编译，这里用 importlib 按文件路径加载，能够整体编译后应该去掉这部分
# from rtp_llm.models_py.distributed.moriep_wrapper import MoriEPWrapper, MoriEPWrapperConfig, init_moriep_wrapper

_DIST_DIR = pathlib.Path(__file__).resolve().parent.parent


def _load_module(name: str, path: pathlib.Path):
    """Load a single Python file as *name* in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_wrapper_mod = _load_module(
    "moriep_wrapper_under_test", _DIST_DIR / "moriep_wrapper.py"
)
MoriEPWrapper = _wrapper_mod.MoriEPWrapper
MoriEPWrapperConfig = _wrapper_mod.MoriEPWrapperConfig
init_moriep_wrapper = _wrapper_mod.init_moriep_wrapper


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _init_dist(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )


def _cleanup_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test data generation & correctness checks
# (参考 mori/tests/python/ops/test_dispatch_combine.py 中 EpDispatchCombineTestCase)
# ---------------------------------------------------------------------------


def _gen_test_data(config: MoriEPWrapperConfig, device: torch.device):
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    num_tokens = config.max_num_inp_token_per_rank
    num_total_experts = config.num_experts_per_rank * config.world_size

    all_rank_num_token = torch.full(
        (config.world_size,),
        num_tokens,
        device=device,
        dtype=torch.int64,
    )

    all_rank_indices = []
    for _ in range(config.world_size):
        indices = torch.empty(
            num_tokens, config.num_experts_per_token, dtype=torch.int64
        )
        for i in range(num_tokens):
            perm = torch.randperm(num_total_experts, generator=rng, device=device)
            indices[i] = perm[: config.num_experts_per_token]
        all_rank_indices.append(indices.to(torch.int32).to(device))

    all_rank_weights = [
        torch.rand(
            num_tokens,
            config.num_experts_per_token,
            dtype=torch.float32,
            generator=rng,
            device=device,
        )
        for _ in range(config.world_size)
    ]

    all_rank_scales = [
        torch.empty(num_tokens, config.scale_dim, dtype=torch.float32, device=device)
        for _ in range(config.world_size)
    ]

    all_rank_input = []
    for _ in range(config.world_size):
        inp = torch.randn(
            num_tokens,
            config.hidden_dim,
            dtype=torch.float32,
            generator=rng,
            device=device,
        ).to(config.data_type)
        all_rank_input.append(inp)

    return (
        all_rank_num_token,
        all_rank_indices,
        all_rank_input,
        all_rank_weights,
        all_rank_scales,
    )


def _check_dispatch_result(
    config: MoriEPWrapperConfig,
    op,
    test_data,
    dispatch_output,
    dispatch_weights,
    dispatch_scales,
    dispatch_indices,
    dispatch_recv_num_token,
):
    torch.cuda.synchronize()
    dist.barrier()

    _, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales = test_data
    src_token_pos = op.get_dispatch_src_token_pos()

    for i, pos in enumerate(src_token_pos):
        src_rank = int(pos) // config.max_num_inp_token_per_rank
        src_id = int(pos) % config.max_num_inp_token_per_rank

        assert torch.equal(
            all_rank_input[src_rank][src_id], dispatch_output[i]
        ), f"token {i}: data mismatch (src_rank={src_rank}, src_id={src_id})"

        if dispatch_weights is not None:
            assert torch.equal(
                all_rank_weights[src_rank][src_id], dispatch_weights[i]
            ), f"token {i}: weight mismatch"

        if dispatch_scales is not None:
            assert torch.equal(
                all_rank_scales[src_rank][src_id], dispatch_scales[i]
            ), f"token {i}: scale mismatch"

        assert torch.equal(
            all_rank_indices[src_rank][src_id], dispatch_indices[i]
        ), f"token {i}: index mismatch"

    assert len(torch.unique(src_token_pos)) == len(
        src_token_pos
    ), "duplicate src_token_pos"
    assert len(src_token_pos) == dispatch_recv_num_token[0], "recv count mismatch"


def _check_combine_result(
    config: MoriEPWrapperConfig,
    test_data,
    combine_output,
):
    torch.cuda.synchronize()
    dist.barrier()

    all_rank_num_token, all_rank_indices, all_rank_input, _, _ = test_data
    rank = config.rank

    for i in range(all_rank_num_token[rank]):
        pes = [
            int(idx) // config.num_experts_per_rank
            for idx in all_rank_indices[rank][i].cpu().tolist()
        ]
        unique_pes = len(set(pes))

        expected = (all_rank_input[rank][i].to(torch.float32) * unique_pes).to(
            config.data_type
        )
        got = combine_output[i]

        atol, rtol = 1e-2, 1e-2
        assert torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol), (
            f"Rank[{rank}] token {i}: combine mismatch, "
            f"pes={pes}, unique_pes={unique_pes}, got={got}, expected={expected}"
        )


# ---------------------------------------------------------------------------
# Worker: single-rank correctness test
# ---------------------------------------------------------------------------


def _run_moriep_correctness(
    rank: int,
    world_size: int,
    port: int,
    data_type: torch.dtype,
    max_inp_token_per_rank: int,
    use_external_inp_buf: bool,
) -> None:

    try:
        _init_dist(rank, world_size, port)

        config = MoriEPWrapperConfig(
            data_type=data_type,
            rank=rank,
            world_size=world_size,
            hidden_dim=1024,
            scale_dim=0,
            scale_type_size=0,
            max_token_type_size=torch.tensor([], dtype=data_type).element_size(),
            max_num_inp_token_per_rank=max_inp_token_per_rank,
            num_experts_per_rank=4,
            num_experts_per_token=2,
            use_external_inp_buf=use_external_inp_buf,
            gpu_per_node=world_size,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            rdma_block_num=0,
            num_qp_per_pe=1,
            quant_type="none",
        )
        init_moriep_wrapper(config, shmem_group_name="default")
        wrapper = MoriEPWrapper.get_instance()
        device = torch.device("cuda", rank)

        test_data = _gen_test_data(config, device)
        _, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales = (
            test_data
        )

        # ---- dispatch ----
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = wrapper.dispatch(
            all_rank_input[rank],
            all_rank_weights[rank],
            all_rank_scales[rank],
            all_rank_indices[rank],
            block_num=64,
            warp_per_block=16,
        )

        _check_dispatch_result(
            config,
            wrapper.op,
            test_data,
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        )

        # ---- combine ----
        total_recv_num_token = int(dispatch_recv_num_token[0].item())

        if not config.use_external_inp_buf:
            combine_input = wrapper.op.get_registered_combine_input_buffer(
                config.data_type,
                hidden_dim=dispatch_output.size(1),
            )
            combine_input[:total_recv_num_token, :].copy_(
                dispatch_output[:total_recv_num_token, :]
            )
            combine_source = combine_input
        else:
            combine_source = dispatch_output

        combine_output, _ = wrapper.combine(
            combine_source,
            None,
            dispatch_indices,
            block_num=64,
            warp_per_block=16,
        )

        _check_combine_result(config, test_data, combine_output)

        if rank == 0:
            print(
                f"[PASS] dtype={data_type}, world_size={world_size}, "
                f"use_external_inp_buf={use_external_inp_buf}, "
                f"recv_tokens={total_recv_num_token}"
            )

        wrapper.reset_op()
        MoriEPWrapper.reset()
        if hasattr(mori.shmem, "shmem_finalize"):
            mori.shmem.shmem_finalize()
    finally:
        _cleanup_dist()


# ---------------------------------------------------------------------------
# Test cases: world_size = 2, 4, 8, data_type = bf16, fp8_e4m3fn, num_tokens = 128, 4096, 8192, use_external_inp_buf = True, False
# ---------------------------------------------------------------------------


class MoriEPWrapperIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def _require_mori(self):
        try:
            import mori  # noqa: F401
        except ImportError:
            self.skipTest("mori is not available")

    def _require_gpus(self, n: int):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if torch.cuda.device_count() < n:
            self.skipTest(f"Need at least {n} CUDA devices")

    def _run(
        self,
        world_size: int,
        data_type: torch.dtype,
        max_inp_token_per_rank: int,
        use_external_inp_buf: bool,
    ):
        self._require_gpus(world_size)
        self._require_mori()
        port = _get_free_port()
        mp.spawn(
            _run_moriep_correctness,
            args=(
                world_size,
                port,
                data_type,
                max_inp_token_per_rank,
                use_external_inp_buf,
            ),
            nprocs=world_size,
            join=True,
        )

    _world_size_list = [2, 4, 8]
    _dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    _max_inp_token_per_rank_list = [128]

    def test_dispatch_combine_correctness(self):
        for dtype in self._dtype_list:
            for ws in self._world_size_list:
                for max_inp_token_per_rank in self._max_inp_token_per_rank_list:
                    for use_external_inp_buf in [True, False]:
                        with self.subTest(
                            dtype=dtype,
                            world_size=ws,
                            max_inp_token_per_rank=max_inp_token_per_rank,
                            use_external_inp_buf=use_external_inp_buf,
                        ):
                            self._run(
                                ws, dtype, max_inp_token_per_rank, use_external_inp_buf
                            )


if __name__ == "__main__":
    unittest.main()
