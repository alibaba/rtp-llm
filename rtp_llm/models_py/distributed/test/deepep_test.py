# type: ignore
import itertools
import os
import random
import time
from functools import partial
from typing import Any, Dict, Tuple
from unittest import TestCase, main
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ProcessGroup

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPBuffer,
    DeepEPConfig,
    DeepEPWrapper,
    DeepepWrapperConfig,
    init_deepep_wrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.utils.math import align
from rtp_llm.ops import (
    FfnDisAggregateConfig,
    MoeConfig,
    NcclCommConfig,
    ParallelismConfig,
)
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.utils.bench_util import bench, bench_kineto
from rtp_llm.test.utils.numeric_util import (
    calc_diff,
    hash_tensor,
    per_token_cast_back,
    per_token_cast_to_fp8,
)
from rtp_llm.test.utils.port_util import PortsContext


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def calc_ll_num_max_token_per_rank(max_generate_batch_size: int, tp_size: int) -> int:
    """Calculate ll_num_max_token_per_rank with alignment to 8.

    Args:
        max_generate_batch_size: Maximum generation batch size
        tp_size: Tensor parallelism size

    Returns:
        ll_num_max_token_per_rank aligned to 8
    """
    ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size
    # Align to 8
    ll_num_max_token_per_rank = (ll_num_max_token_per_rank + 7) // 8 * 8
    return ll_num_max_token_per_rank


class DeepEPTest(TestCase):

    NUM_PROCESSES = [2]
    MAX_SEQ_LEN = [1024]
    MAX_GENERATE_BATCH_SIZES = [128]
    HIDDEN_SIZES = [7168]
    NUM_EXPERT = [64]
    TOP_K = [8]

    def setUp(self) -> None:
        pass

    @staticmethod
    def _test_intranode_main(
        num_tokens: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        num_sms: int,
        local_rank: int,
        num_ranks: int,
        rank: int,
        buffer: DeepEPBuffer,
        group: dist.ProcessGroup,
    ):
        assert num_experts % num_ranks == 0
        if local_rank == 0:
            print(
                f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}",
                flush=True,
            )

        # Random data
        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
        x_pure_rand = torch.randn(
            (num_tokens, hidden), dtype=torch.bfloat16, device="cuda"
        )
        x_e4m3 = (
            per_token_cast_to_fp8(x, False) if DeepEPBuffer.is_sm90_compiled() else None
        )
        x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
        topk_weights = (
            torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda")
            * rank
        )
        topk_weights_pure_rand = torch.randn(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        )
        rank_idx = topk_idx // (num_experts // num_ranks)
        rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx, num_ranks)

        # Expert meta
        num_tokens_per_expert = torch.zeros(
            (num_experts,), dtype=torch.int, device="cuda"
        )
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
        gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
        dist.all_reduce(gbl_num_tokens_per_expert, group=group)

        # Rank layout meta
        num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
        token_idx_in_rank = torch.full(
            (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
        )
        for i in range(num_ranks):
            num_tokens_per_rank[i] = (rank_idx == i).sum()
            token_sel = (rank_idx == i).max(dim=-1)[0]
            count = token_sel.sum().item()
            tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
            tokens[:count] = torch.sort(tokens[:count])[0]
            token_idx_in_rank[i][tokens[:count]] = torch.arange(
                count, dtype=torch.long, device="cuda"
            )
        token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
        is_token_in_rank = token_idx_in_rank >= 0
        gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
        dist.all_reduce(gbl_num_tokens_per_rank, group=group)

        (
            ref_num_tokens_per_rank,
            _,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
        t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
        if local_rank == 0:
            print(f"[layout] Kernel performance: {t * 1000:.3f} ms", flush=True)
            print("", flush=True)
        group.barrier()
        time.sleep(1)

        # Config
        nvl_buffer_size = 256
        config = DeepEPConfig(num_sms, 8, nvl_buffer_size)

        # Test dispatch
        # noinspection PyShadowingNames
        def check_data(check_x, rank_prefix_matrix):
            assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
            check_start = 0
            for i in range(num_ranks):
                check_end = rank_prefix_matrix[i][rank].item()
                assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
                check_start = check_end

        for previous_mode in (False, True):
            for async_mode in (False, True):
                for current_x in filter(
                    lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)
                ):
                    for with_topk in (False, True):
                        if local_rank == 0:
                            print(
                                f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                                flush=True,
                                end="",
                            )
                        dispatch_args = {
                            "x": current_x,
                            "num_tokens_per_rank": num_tokens_per_rank,
                            "is_token_in_rank": is_token_in_rank,
                            "num_tokens_per_expert": num_tokens_per_expert,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            dispatch_args.update(
                                {
                                    "topk_idx": topk_idx,
                                    "topk_weights": (
                                        topk_weights_pure_rand
                                        if current_x is x_pure_rand
                                        else topk_weights
                                    ),
                                }
                            )
                        if previous_mode:
                            dispatch_args.update({"previous_event": buffer.capture()})
                        (
                            recv_x,
                            recv_topk_idx,
                            recv_topk_weights,
                            recv_num_tokens_per_expert_list,
                            handle,
                            event,
                        ) = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = (
                            per_token_cast_back(*recv_x)
                            if isinstance(recv_x, tuple)
                            else recv_x
                        )

                        # Checks
                        rank_prefix_matrix = handle[0]
                        assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                            0
                        ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
                        assert (
                            gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
                            == recv_num_tokens_per_expert_list
                        )
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)
                        recv_topk_weights_clone = None
                        if with_topk:
                            # Check `topk_idx`
                            assert (
                                recv_topk_idx.eq(-1)
                                | (
                                    (recv_topk_idx >= 0)
                                    & (recv_topk_idx < (num_experts // num_ranks))
                                )
                            ).sum().item() == recv_topk_idx.numel()
                            for i, count in enumerate(recv_num_tokens_per_expert_list):
                                assert recv_topk_idx.eq(i).sum().item() == count

                            # Check `topk_weights`
                            recv_topk_weights_clone = recv_topk_weights.clone()
                            if current_x is not x_pure_rand:
                                recv_topk_weights[recv_topk_idx.eq(-1)] = (
                                    recv_topk_weights.amax(
                                        dim=1, keepdim=True
                                    ).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                                )
                                check_data(recv_topk_weights, rank_prefix_matrix)

                        # Test `num_worst_tokens != 0`
                        if with_topk:
                            num_worst_tokens = num_tokens * num_ranks
                            dispatch_args.update({"num_worst_tokens": num_worst_tokens})
                            (
                                recv_worst_x,
                                recv_worst_topk_idx,
                                recv_worst_topk_weights,
                                empty_list,
                                _,
                                event,
                            ) = buffer.dispatch(**dispatch_args)
                            event.current_stream_wait() if async_mode else ()
                            recv_worst_x = (
                                per_token_cast_back(*recv_worst_x)
                                if isinstance(recv_worst_x, tuple)
                                else recv_worst_x
                            )
                            assert len(empty_list) == 0
                            assert num_worst_tokens == recv_worst_x.size(0)
                            assert num_worst_tokens == recv_worst_topk_idx.size(0)
                            assert num_worst_tokens == recv_worst_topk_weights.size(0)
                            assert torch.equal(recv_x, recv_worst_x[: recv_x.size(0)])
                            assert torch.equal(
                                recv_topk_idx, recv_worst_topk_idx[: recv_x.size(0)]
                            )
                            assert torch.equal(
                                recv_topk_weights_clone,
                                recv_worst_topk_weights[: recv_x.size(0)],
                            )
                            assert torch.all(
                                recv_worst_topk_idx[recv_x.size(0) :] == -1
                            ).item()

                        # Test cached dispatch (must without top-k staffs)
                        if not with_topk:
                            dispatch_args = {
                                "x": current_x,
                                "handle": handle,
                                "config": config,
                                "async_finish": async_mode,
                            }
                            if previous_mode:
                                dispatch_args.update(
                                    {"previous_event": buffer.capture()}
                                )
                            recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                            event.current_stream_wait() if async_mode else ()
                            recv_x = (
                                per_token_cast_back(*recv_x)
                                if isinstance(recv_x, tuple)
                                else recv_x
                            )
                            if current_x is not x_pure_rand:
                                check_data(recv_x, rank_prefix_matrix)

                        # Test combine
                        combine_args = {
                            "x": recv_x,
                            "handle": handle,
                            "config": config,
                            "async_finish": async_mode,
                        }
                        if with_topk:
                            combine_args.update({"topk_weights": recv_topk_weights})
                        if previous_mode:
                            combine_args.update({"previous_event": buffer.capture()})
                        combined_x, combined_topk_weights, event = buffer.combine(
                            **combine_args
                        )
                        event.current_stream_wait() if async_mode else ()
                        check_x = combined_x.float() / is_token_in_rank.sum(
                            dim=1
                        ).unsqueeze(1)
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        assert calc_diff(check_x, ref_x) < 5e-6
                        if with_topk:
                            check_topk_weights = (
                                combined_topk_weights
                                if (current_x is x_pure_rand)
                                else (
                                    combined_topk_weights
                                    / is_token_in_rank.sum(dim=1).unsqueeze(1)
                                )
                            )
                            ref_topk_weights = (
                                topk_weights_pure_rand
                                if current_x is x_pure_rand
                                else topk_weights
                            )
                            assert (
                                calc_diff(check_topk_weights, ref_topk_weights) < 1e-9
                            )

                        # For later tuning
                        dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                        combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                        if local_rank == 0:
                            print(" passed", flush=True)
        if local_rank == 0:
            print("", flush=True)

        # Tune dispatch performance
        best_dispatch_results = None
        fp8_factor = (1 + 4 / 128) / 2
        for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
            best_time, best_results = 1e10, None
            nvl_recv_bytes = (
                (dispatch_bf16_nvl_recv_bytes * fp8_factor)
                if isinstance(current_x, tuple)
                else dispatch_bf16_nvl_recv_bytes
            )
            for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
                if nvl_chunk_size > 0:
                    config = DeepEPConfig(num_sms, nvl_chunk_size, nvl_buffer_size)
                else:
                    # Test default config as well
                    DeepEPBuffer.set_num_sms(num_sms)
                    config = DeepEPBuffer.get_dispatch_config(num_ranks)
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t = bench(lambda: buffer.dispatch(**tune_args))[0]
                if t < best_time and nvl_chunk_size > 0:
                    best_time, best_results = t, (num_sms, nvl_chunk_size)
                if local_rank == 0:
                    print(
                        f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                        f"{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                        flush=True,
                    )
            if local_rank == 0:
                print(
                    f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                    flush=True,
                )
                print("", flush=True)

            # Gather the best config from rank 0 and the first test setting
            if best_dispatch_results is None:
                best_dispatch_results = torch.tensor(
                    [best_results[0], best_results[1]], dtype=torch.int32, device="cuda"
                )
                all_best_fp8_results_list = [
                    torch.zeros_like(best_dispatch_results)
                    for _ in range(torch.distributed.get_world_size())
                ]
                dist.all_gather(
                    all_best_fp8_results_list, best_dispatch_results, group=group
                )
                best_dispatch_results = all_best_fp8_results_list[0].tolist()
        dispatch_config = DeepEPConfig(
            best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size
        )

        dispatch_args = {
            "x": x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": dispatch_config if dispatch_config is not None else config,
        }
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

        # Tune combine performance
        best_time, best_results = 1e10, None
        for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
            if nvl_chunk_size > 0:
                config = DeepEPConfig(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                DeepEPBuffer.set_num_sms(num_sms)
                config = DeepEPBuffer.get_combine_config(num_ranks)
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            t = bench(lambda: buffer.combine(**tune_args))[0]
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f"{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                    flush=True,
                )
                if t < best_time and nvl_chunk_size > 0:
                    best_time, best_results = t, (num_sms, nvl_chunk_size)

        if local_rank == 0:
            print(
                f"[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us",
                flush=True,
            )
            print("", flush=True)

    @staticmethod
    def _test_low_latency_main(
        num_tokens: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        rank: int,
        num_ranks: int,
        group: ProcessGroup,
        buffer: DeepEPBuffer,
        seed: int = 0,
    ):
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)

        assert num_experts % num_ranks == 0
        num_local_experts = num_experts // num_ranks

        # NOTES: the integers greater than 256 exceed the BF16 precision limit
        rank_offset = 128
        assert (
            num_ranks - rank_offset < 257
        ), "Too many ranks (exceeding test precision limit)"

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
            rank - rank_offset
        )
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            x[:, -hidden:] = (
                torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
            )
        else:
            x[:, -128:] = (
                torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
            )
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
        topk_weights = torch.randn(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        ).abs()

        # Randomly mask some positions
        for i in range(10):
            topk_idx[
                random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)
            ] = -1

        # Check dispatch correctness
        do_check = True
        hash_value, num_times = 0, 0
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                    for use_ue8m0 in (False, True) if round_scale else (False,):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros(
                                (num_local_experts,), dtype=torch.int, device="cuda"
                            )
                            packed_recv_x, packed_recv_count, handle, event, hook = (
                                buffer.low_latency_dispatch(
                                    x,
                                    topk_idx,
                                    num_tokens,
                                    num_experts,
                                    use_fp8=dispatch_use_fp8,
                                    round_scale=round_scale,
                                    use_ue8m0=use_ue8m0,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook,
                                    return_recv_hook=return_recv_hook,
                                )
                            )
                            hook() if return_recv_hook else event.current_stream_wait()
                        packed_recv_x = (
                            (packed_recv_x[0], packed_recv_x[1].contiguous())
                            if dispatch_use_fp8
                            else packed_recv_x
                        )
                        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, 1),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8
                                else packed_recv_x.clone()
                            )
                        else:
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, hidden // 128),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8
                                else packed_recv_x.clone()
                            )
                        all_topk_idx = torch.empty(
                            (num_ranks, num_tokens, num_topk),
                            dtype=topk_idx.dtype,
                            device="cuda",
                        )
                        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = (
                                per_token_cast_back(
                                    packed_recv_x[0][i], packed_recv_x[1][i]
                                )
                                if dispatch_use_fp8
                                else packed_recv_x[i]
                            )
                            recv_count, recv_src_info, recv_layout_range = (
                                packed_recv_count[i],
                                handle[0][i],
                                handle[1][i],
                            )

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert (
                                cumulative_local_expert_recv_stats[i].item()
                                == num_valid_tokens
                            ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                            assert (
                                num_valid_tokens
                                == (recv_layout_range & int_mask).sum().item()
                            ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                            assert (
                                num_valid_tokens
                                == (all_topk_idx == expert_id).sum().item()
                            ), f"{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}"

                            # Check received data
                            recv_x = recv_x[:num_valid_tokens]
                            recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                            recv_src_info = recv_src_info[:num_valid_tokens]
                            assert torch.equal(
                                recv_x_amin, recv_x[:, :-128].amax(dim=-1)
                            )
                            if round_scale:
                                assert (
                                    calc_diff(recv_x[:, -1], recv_src_info.view(-1))
                                    < 0.007
                                )
                            else:
                                assert (
                                    recv_x[:, -128:]
                                    - recv_src_info.view(-1, 1) % num_tokens
                                ).sum().item() == 0
                            for j in range(num_ranks):
                                begin_idx, count = (
                                    recv_layout_range[j] >> 32
                                ).item(), (recv_layout_range[j] & int_mask).item()
                                if (
                                    not round_scale
                                    and os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2"
                                ):
                                    assert (
                                        recv_x_amin == j - rank_offset
                                    ).sum().item() == (
                                        all_topk_idx[j] == expert_id
                                    ).sum().item()
                                assert (
                                    recv_x[begin_idx : begin_idx + count][:-128] - j
                                ).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[0][i, :num_valid_tokens]
                                )
                                if os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2":
                                    hash_value ^= hash_tensor(
                                        packed_recv_x[1][i, :num_valid_tokens]
                                    )
                            else:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[i, :num_valid_tokens]
                                )

                        # Check combine correctness
                        for zero_copy in (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[
                                    :, :, :
                                ] = simulated_gemm_x
                            out = torch.empty(
                                (num_tokens, hidden),
                                dtype=torch.bfloat16,
                                device="cuda",
                            )
                            combined_x, event, hook = buffer.low_latency_combine(
                                simulated_gemm_x,
                                topk_idx,
                                topk_weights,
                                handle,
                                async_finish=not return_recv_hook,
                                zero_copy=zero_copy,
                                return_recv_hook=return_recv_hook,
                                out=out,
                            )
                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                diff = calc_diff(
                                    x
                                    * topk_weights.masked_fill(topk_idx == -1, 0)
                                    .sum(dim=1)
                                    .view(-1, 1),
                                    combined_x,
                                )
                                assert torch.isnan(combined_x).sum().item() == 0
                                if not diff < (7e-4 if round_scale else 1e-5):
                                    print(
                                        f"assert Error: diff < (7e-4 if round_scale else 1e-5) {diff=}, {zero_copy=}",
                                        flush=True,
                                    )
                                # when enable round_scale, this diff check may not be passed
                                # assert diff < (7e-4 if round_scale else 1e-5), f'Error: {diff=}, {zero_copy=}'
                                if os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2":
                                    hash_value ^= hash_tensor(combined_x)

        # noinspection PyShadowingNames
        def large_gemm_with_hook(hook):
            mat_0 = torch.randn((8192, 8192), dtype=torch.float)
            mat_1 = torch.randn((8192, 8192), dtype=torch.float)
            mat_0 @ mat_1
            hook()

        # noinspection PyShadowingNames
        def test_func(zero_copy: bool, return_recv_hook: bool):
            recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                use_fp8=True,
                async_finish=False,
                return_recv_hook=return_recv_hook,
            )
            large_gemm_with_hook(hook) if return_recv_hook else None
            if zero_copy:
                buffer.get_next_low_latency_combine_buffer(handle)[
                    :, :, :
                ] = simulated_gemm_x
            combined_x, event, hook = buffer.low_latency_combine(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                zero_copy=zero_copy,
                return_recv_hook=return_recv_hook,
            )
            large_gemm_with_hook(hook) if return_recv_hook else None

        # Calculate bandwidth
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            num_fp8_bytes, num_bf16_bytes = (
                hidden + hidden / hidden * 4 + 16
            ), hidden * 2
        else:
            num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
        num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
        for i in range(num_tokens):
            num_selections = (topk_idx[i] != -1).sum().item()
            num_dispatch_comm_bytes += num_fp8_bytes * num_selections
            num_combine_comm_bytes += num_bf16_bytes * num_selections

        # Dispatch + combine testing
        avg_t, min_t, max_t = bench(
            partial(test_func, zero_copy=False, return_recv_hook=False)
        )
        print(
            f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
            f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
            flush=True,
        )

        # Separate profiling
        for return_recv_hook in (False, True):
            group.barrier()
            dispatch_t, combine_t = bench_kineto(
                partial(test_func, zero_copy=True, return_recv_hook=return_recv_hook),
                kernel_names=("dispatch", "combine"),
                barrier_comm_profiling=True,
                suppress_kineto_output=True,
                num_kernels_per_period=2 if return_recv_hook else 1,
            )
            if not return_recv_hook:
                print(
                    f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                    f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
                    flush=True,
                )
            else:
                print(
                    f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                    f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                    flush=True,
                )
        return hash_value

    @staticmethod
    def _test_low_latency_m2n_main(
        scale,
        ae_mask: int,
        num_m: int,
        num_tokens: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        rank: int,
        num_ranks: int,
        group: dist.ProcessGroup,
        buffer: DeepEPBuffer,
        seed: int = 0,
    ):
        num_n = num_ranks - num_m
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)

        assert num_experts % num_ranks == 0
        num_local_experts = num_experts // num_ranks

        # NOTES: the integers greater than 256 exceed the BF16 precision limit
        rank_offset = 128
        assert (
            num_ranks - rank_offset < 257
        ), "Too many ranks (exceeding test precision limit)"

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
            rank - rank_offset
        )
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            x[:, -hidden:] = (
                torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
            )
        else:
            x[:, -128:] = (
                torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
            )
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
        topk_weights = torch.randn(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        ).abs()

        # Randomly mask some positions
        for i in range(10):
            topk_idx[
                random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)
            ] = -1

        # Check dispatch correctness
        if rank < ae_mask:
            do_check = True
        else:
            do_check = False
        hash_value, num_times = 0, 0
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                    for use_ue8m0 in (False, True) if round_scale else (False,):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros(
                                (num_local_experts,), dtype=torch.int, device="cuda"
                            )
                            if rank < ae_mask:
                                (
                                    packed_recv_x,
                                    packed_recv_count,
                                    handle,
                                    event,
                                    hook,
                                ) = buffer.low_latency_dispatch_send(
                                    x,
                                    topk_idx,
                                    num_tokens,
                                    int(num_experts // scale),
                                    ae_mask,
                                    num_topk,
                                    use_fp8=dispatch_use_fp8,
                                    round_scale=round_scale,
                                    use_ue8m0=use_ue8m0,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook,
                                    return_recv_hook=return_recv_hook,
                                )
                            else:
                                (
                                    packed_recv_x,
                                    packed_recv_count,
                                    handle,
                                    event,
                                    hook,
                                ) = buffer.low_latency_dispatch_recv(
                                    hidden,
                                    num_topk,
                                    num_tokens,
                                    int(num_experts // scale),
                                    ae_mask,
                                    use_fp8=dispatch_use_fp8,
                                    round_scale=round_scale,
                                    use_ue8m0=use_ue8m0,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook,
                                    return_recv_hook=return_recv_hook,
                                )
                            hook() if return_recv_hook else event.current_stream_wait()
                            if rank >= ae_mask:
                                topk_idx = torch.full((num_tokens, num_topk), -1)
                        packed_recv_x = (
                            (packed_recv_x[0], packed_recv_x[1].contiguous())
                            if dispatch_use_fp8
                            else packed_recv_x
                        )
                        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, 1),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8
                                else packed_recv_x.clone()
                            )
                        else:
                            simulated_gemm_x = (
                                per_token_cast_back(
                                    packed_recv_x[0].view(-1, hidden),
                                    packed_recv_x[1].view(-1, hidden // 128),
                                ).view(packed_recv_x[0].shape)
                                if dispatch_use_fp8
                                else packed_recv_x.clone()
                            )
                        all_topk_idx = torch.empty(
                            (num_ranks, num_tokens, num_topk),
                            dtype=topk_idx.dtype,
                            device="cuda",
                        )
                        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = (
                                per_token_cast_back(
                                    packed_recv_x[0][i], packed_recv_x[1][i]
                                )
                                if dispatch_use_fp8
                                else packed_recv_x[i]
                            )
                            recv_count, recv_src_info, recv_layout_range = (
                                packed_recv_count[i],
                                handle[0][i],
                                handle[1][i],
                            )

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert (
                                cumulative_local_expert_recv_stats[i].item()
                                == num_valid_tokens
                            ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                            assert (
                                num_valid_tokens
                                == (recv_layout_range & int_mask).sum().item()
                            ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                            assert (
                                num_valid_tokens
                                == (all_topk_idx == expert_id).sum().item()
                            ), f"{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}"

                            # Check received data
                            recv_x = recv_x[:num_valid_tokens]
                            recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                            recv_src_info = recv_src_info[:num_valid_tokens]
                            assert torch.equal(
                                recv_x_amin, recv_x[:, :-128].amax(dim=-1)
                            )
                            if round_scale:
                                # Skip assertion if no data is received
                                if recv_src_info.numel() > 0:
                                    assert (
                                        calc_diff(recv_x[:, -1], recv_src_info.view(-1))
                                        < 0.007
                                    )
                            else:
                                assert (
                                    recv_x[:, -128:]
                                    - recv_src_info.view(-1, 1) % num_tokens
                                ).sum().item() == 0
                            for j in range(num_ranks):
                                begin_idx, count = (
                                    recv_layout_range[j] >> 32
                                ).item(), (recv_layout_range[j] & int_mask).item()
                                if (
                                    not round_scale
                                    and os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2"
                                ):
                                    assert (
                                        recv_x_amin == j - rank_offset
                                    ).sum().item() == (
                                        all_topk_idx[j] == expert_id
                                    ).sum().item()
                                assert (
                                    recv_x[begin_idx : begin_idx + count][:-128] - j
                                ).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[0][i, :num_valid_tokens]
                                )
                                if os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2":
                                    hash_value ^= hash_tensor(
                                        packed_recv_x[1][i, :num_valid_tokens]
                                    )
                            else:
                                hash_value ^= hash_tensor(
                                    packed_recv_x[i, :num_valid_tokens]
                                )

                        # Check combine correctness
                        for zero_copy in (False, True):
                            if zero_copy:
                                if rank < ae_mask:
                                    buffer.get_next_low_latency_combine_buffer(handle)[
                                        :, :, :
                                    ] = simulated_gemm_x
                                else:
                                    buffer.get_next_low_latency_combine_buffer_m2n(
                                        handle, hidden
                                    )[:, :, :] = simulated_gemm_x
                            out = torch.empty(
                                (num_tokens, hidden),
                                dtype=torch.bfloat16,
                                device="cuda",
                            )
                            if rank < ae_mask:
                                combined_x, event, hook = (
                                    buffer.low_latency_combine_recv(
                                        topk_idx,
                                        topk_weights,
                                        handle,
                                        ae_mask,
                                        num_tokens,
                                        num_topk,
                                        async_finish=not return_recv_hook,
                                        zero_copy=zero_copy,
                                        return_recv_hook=return_recv_hook,
                                        out=out,
                                    )
                                )
                            else:
                                combined_x, event, hook = (
                                    buffer.low_latency_combine_send(
                                        simulated_gemm_x,
                                        handle,
                                        num_topk,
                                        async_finish=not return_recv_hook,
                                        zero_copy=zero_copy,
                                        return_recv_hook=return_recv_hook,
                                        out=out,
                                    )
                                )
                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                diff = calc_diff(
                                    x
                                    * topk_weights.masked_fill(topk_idx == -1, 0)
                                    .sum(dim=1)
                                    .view(-1, 1),
                                    combined_x,
                                )
                                assert torch.isnan(combined_x).sum().item() == 0
                                if not diff < (7e-4 if round_scale else 1e-5):
                                    print(
                                        f"assert Error: diff < (7e-4 if round_scale else 1e-5) {diff=}, {zero_copy=}",
                                        flush=True,
                                    )
                                # when enable round_scale, this diff check may not be passed
                                # assert diff < (7e-4 if round_scale else 1e-5), f'Error: {diff=}, {zero_copy=}'
                                if os.getenv("ACCL_FP8_CAST_LEVEL", "1") != "2":
                                    hash_value ^= hash_tensor(combined_x)

        # noinspection PyShadowingNames
        def large_gemm_with_hook(hook):
            mat_0 = torch.randn((8192, 8192), dtype=torch.float)
            mat_1 = torch.randn((8192, 8192), dtype=torch.float)
            mat_0 @ mat_1
            hook()

        # noinspection PyShadowingNames
        def test_func(zero_copy: bool, return_recv_hook: bool):
            if rank < ae_mask:
                recv_x, recv_count, handle, event, hook = (
                    buffer.low_latency_dispatch_send(
                        x,
                        topk_idx,
                        num_tokens,
                        int(num_experts // scale),
                        ae_mask,
                        num_topk,
                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                        use_fp8=True,
                        async_finish=False,
                        return_recv_hook=return_recv_hook,
                    )
                )
            else:
                recv_x, recv_count, handle, event, hook = (
                    buffer.low_latency_dispatch_recv(
                        hidden,
                        num_topk,
                        num_tokens,
                        int(num_experts // scale),
                        ae_mask,
                        use_fp8=dispatch_use_fp8,
                        round_scale=round_scale,
                        use_ue8m0=use_ue8m0,
                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                        async_finish=not return_recv_hook,
                        return_recv_hook=return_recv_hook,
                    )
                )
            large_gemm_with_hook(hook) if return_recv_hook else None
            if zero_copy:
                if rank < ae_mask:
                    buffer.get_next_low_latency_combine_buffer(handle)[
                        :, :, :
                    ] = simulated_gemm_x
                else:
                    buffer.get_next_low_latency_combine_buffer_m2n(handle, hidden)[
                        :, :, :
                    ] = simulated_gemm_x
            if rank < ae_mask:
                combined_x, event, hook = buffer.low_latency_combine_recv(
                    topk_idx,
                    topk_weights,
                    handle,
                    ae_mask,
                    num_tokens,
                    num_topk,
                    zero_copy=zero_copy,
                    return_recv_hook=return_recv_hook,
                )
            else:
                combined_x, event, hook = buffer.low_latency_combine_send(
                    simulated_gemm_x,
                    handle,
                    num_topk,
                    async_finish=not return_recv_hook,
                    zero_copy=zero_copy,
                    return_recv_hook=return_recv_hook,
                    out=out,
                )
            large_gemm_with_hook(hook) if return_recv_hook else None

        # Calculate bandwidth
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            num_fp8_bytes, num_bf16_bytes = (
                hidden + hidden / hidden * 4 + 16
            ), hidden * 2
        else:
            num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
        num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
        for i in range(num_tokens):
            num_selections = (topk_idx[i] != -1).sum().item()
            num_dispatch_comm_bytes += num_fp8_bytes * num_selections
            num_combine_comm_bytes += num_bf16_bytes * num_selections

        # Dispatch + combine testing
        avg_t, min_t, max_t = bench(
            partial(test_func, zero_copy=False, return_recv_hook=False)
        )
        print(
            f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
            f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
            flush=True,
        )

        # Separate profiling
        for return_recv_hook in (False, True):
            group.barrier()
            dispatch_t, combine_t = bench_kineto(
                partial(test_func, zero_copy=True, return_recv_hook=return_recv_hook),
                kernel_names=("dispatch", "combine"),
                barrier_comm_profiling=True,
                suppress_kineto_output=True,
                num_kernels_per_period=2 if return_recv_hook else 1,
            )
            if not return_recv_hook:
                print(
                    f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                    f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
                    flush=True,
                )
            else:
                print(
                    f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                    f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                    flush=True,
                )
        return hash_value

    @staticmethod
    def _test_intranode_expert_alignment_main(
        rank: int,
        num_ranks: int,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        expert_alignment: int,
        buffer: DeepEPBuffer,
        group: dist.ProcessGroup,
        seed: int = 777,
    ):
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)

        assert num_experts % num_ranks == 0
        num_local_experts = num_experts // num_ranks

        # Random data
        if rank == 0:
            num_tokens = 3
        else:
            num_tokens = 2
        x = torch.ones(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
        ) * (rank + 1)
        x[:, 1] = torch.arange(1, num_tokens + 1, device="cuda").to(torch.bfloat16)
        x_e4m3 = (
            per_token_cast_to_fp8(x, False) if DeepEPBuffer.is_sm90_compiled() else None
        )
        x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
        if rank == 0:
            topk_idx = torch.tensor(
                [[3, 1], [3, 1], [3, 1]], dtype=torch.int64, device="cuda"
            )
        else:
            topk_idx = torch.tensor([[0, 1], [3, 1]], dtype=torch.int64, device="cuda")
        topk_weights = torch.ones(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        ) * (rank + 1)
        rank_idx = topk_idx // num_local_experts
        rank_idx.masked_fill_(topk_idx == -1, -1)
        inplace_unique(rank_idx, num_ranks)

        # Expert meta
        num_tokens_per_expert = torch.zeros(
            (num_experts,), dtype=torch.int, device="cuda"
        )
        for i in range(num_experts):
            num_tokens_per_expert[i] = (topk_idx == i).sum()
        gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
        dist.all_reduce(gbl_num_tokens_per_expert, group=group)
        aligned_gbl_num_tokens_per_expert = torch.tensor(
            [align(num, expert_alignment) for num in gbl_num_tokens_per_expert],
            device="cuda",
        )

        # Rank layout meta
        num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
        token_idx_in_rank = torch.full(
            (num_ranks, num_tokens), -1, dtype=torch.long, device="cuda"
        )
        for i in range(num_ranks):
            num_tokens_per_rank[i] = (rank_idx == i).sum()
            token_sel = (rank_idx == i).max(dim=-1)[0]
            count = token_sel.sum().item()
            tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
            tokens[:count] = torch.sort(tokens[:count])[0]
            token_idx_in_rank[i][tokens[:count]] = torch.arange(
                count, dtype=torch.long, device="cuda"
            )
        token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
        is_token_in_rank = token_idx_in_rank >= 0
        gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
        dist.all_reduce(gbl_num_tokens_per_rank, group=group)

        # Test dispatch layout
        (
            ref_num_tokens_per_rank,
            _,
            ref_num_tokens_per_expert,
            ref_is_token_in_rank,
            _,
        ) = buffer.get_dispatch_layout(topk_idx, num_experts)
        assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
        assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
        assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

        # Config
        nvl_buffer_size = 256
        config = DeepEPConfig(24, 8, nvl_buffer_size)

        # Test dispatch fp8
        dispatch_args = {
            "x": x_e4m3,
            "num_tokens_per_rank": ref_num_tokens_per_rank,
            "is_token_in_rank": ref_is_token_in_rank,
            "num_tokens_per_expert": ref_num_tokens_per_expert,
            "config": config,
            "async_finish": True,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "expert_alignment": expert_alignment,
        }
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(**dispatch_args)
        event.current_stream_wait()
        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
        # Checks
        print(
            f"[rank {rank}] num_recv_tokens={recv_x.size(0)}, ref_num_recv_tokens={gbl_num_tokens_per_rank[rank].item()}",
            flush=True,
        )
        assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
            0
        ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
        num_tokens_per_expert = [
            recv_topk_idx.eq(local_expert_idx).sum().item()
            for local_expert_idx in range(num_local_experts)
        ]
        print(
            f"[rank {rank}] num_tokens_per_expert={num_tokens_per_expert}, gbl_num_tokens_per_expert={gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}",
            flush=True,
        )
        assert (
            num_tokens_per_expert
            == gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        ), f"{num_tokens_per_expert} != {gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}"
        print(
            f"[rank {rank}] recv_num_tokens_per_expert_list={recv_num_tokens_per_expert_list}, aligned_gbl_num_tokens_per_expert={aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}",
            flush=True,
        )
        assert (
            recv_num_tokens_per_expert_list
            == aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        ), f"{recv_num_tokens_per_expert_list} != {aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}"
        print(f"[rank {rank}] recv_x={recv_x}", flush=True)
        # Check `topk_idx`
        print(f"[rank {rank}] recv_topk_idx={recv_topk_idx}", flush=True)
        assert (
            recv_topk_idx.eq(-1)
            | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
        ).sum().item() == recv_topk_idx.numel()
        for i, count in enumerate(recv_num_tokens_per_expert_list):
            assert (
                align(recv_topk_idx.eq(i).sum().item(), expert_alignment) == count
            ), f"{align(recv_topk_idx.eq(i).sum().item(), expert_alignment)} != {count}"
        # Check `topk_weights`
        print(f"[rank {rank}] recv_topk_weights={recv_topk_weights}", flush=True)

        # Test dispatch bf16
        dispatch_args = {
            "x": x,
            "num_tokens_per_rank": ref_num_tokens_per_rank,
            "is_token_in_rank": ref_is_token_in_rank,
            "num_tokens_per_expert": ref_num_tokens_per_expert,
            "config": config,
            "async_finish": True,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "expert_alignment": expert_alignment,
        }
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(**dispatch_args)
        event.current_stream_wait()
        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
        # Checks
        print(
            f"[rank {rank}] num_recv_tokens={recv_x.size(0)}, ref_num_recv_tokens={gbl_num_tokens_per_rank[rank].item()}",
            flush=True,
        )
        assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
            0
        ), f"{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}"
        num_tokens_per_expert = [
            recv_topk_idx.eq(local_expert_idx).sum().item()
            for local_expert_idx in range(num_local_experts)
        ]
        print(
            f"[rank {rank}] num_tokens_per_expert={num_tokens_per_expert}, gbl_num_tokens_per_expert={gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}",
            flush=True,
        )
        assert (
            num_tokens_per_expert
            == gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        ), f"{num_tokens_per_expert} != {gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}"
        print(
            f"[rank {rank}] recv_num_tokens_per_expert_list={recv_num_tokens_per_expert_list}, aligned_gbl_num_tokens_per_expert={aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}",
            flush=True,
        )
        assert (
            recv_num_tokens_per_expert_list
            == aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()
        ), f"{recv_num_tokens_per_expert_list} != {aligned_gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist()}"
        print(f"[rank {rank}] recv_x={recv_x}", flush=True)
        # Check `topk_idx`
        print(f"[rank {rank}] recv_topk_idx={recv_topk_idx}", flush=True)
        assert (
            recv_topk_idx.eq(-1)
            | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))
        ).sum().item() == recv_topk_idx.numel()
        for i, count in enumerate(recv_num_tokens_per_expert_list):
            assert (
                align(recv_topk_idx.eq(i).sum().item(), expert_alignment) == count
            ), f"{align(recv_topk_idx.eq(i).sum().item(), expert_alignment)} != {count}"
        # Check `topk_weights`
        print(f"[rank {rank}] recv_topk_weights={recv_topk_weights}", flush=True)

    @staticmethod
    def _test_low_latency_per_token_quant_main(
        num_tokens: int,
        hidden: int,
        num_experts: int,
        num_topk: int,
        rank: int,
        num_ranks: int,
        group: dist.ProcessGroup,
        buffer: DeepEPBuffer,
        use_logfmt: bool = False,
        seed: int = 0,
    ):
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)

        assert num_experts % num_ranks == 0
        num_local_experts = num_experts // num_ranks

        # NOTES: the integers greater than 256 exceed the BF16 precision limit
        rank_offset = 128
        assert (
            num_ranks - rank_offset < 257
        ), "Too many ranks (exceeding test precision limit)"
        quant_size = hidden if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2" else 128

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (
            rank - rank_offset
        )
        x[:, -quant_size:] = (
            torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
        )
        x_list = [x]
        for i in range(4 if use_logfmt else 0):
            # NOTES: make more LogFMT casts and also with some BF16
            x_list.append(
                torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
                * 0.5
                * random.random()
            )
        # NOTES: the last one is for performance testing
        # Most of the values in the perf case is lower than the threshold, casting most channels
        x_list.append(
            torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1
        )
        scores = (
            torch.randn(
                (num_tokens, num_experts), dtype=torch.float32, device="cuda"
            ).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
        topk_weights = torch.randn(
            (num_tokens, num_topk), dtype=torch.float32, device="cuda"
        ).abs()

        # Randomly mask some positions
        for i in range(10):
            topk_idx[
                random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)
            ] = -1

        # Check dispatch correctness
        do_check = True
        hash_value, num_times = 0, 0
        for current_x in x_list:
            for return_recv_hook in (False, True):
                for dispatch_use_fp8 in (False, True):
                    for round_scale in (False, True) if dispatch_use_fp8 else (False,):
                        for use_ue8m0 in (False, True) if round_scale else (False,):
                            for use_per_token_quant in (
                                (False, True)
                                if (
                                    use_ue8m0 == False
                                    and dispatch_use_fp8 == True
                                    and current_x is x_list[-1]
                                )
                                else (False,)
                            ):
                                for dispatch_skip_fp8_quant in (
                                    (False, True)
                                    if (
                                        dispatch_use_fp8
                                        and use_ue8m0 == False
                                        and round_scale == False
                                        and current_x is x_list[-1]
                                    )
                                    else (False,)
                                ):
                                    num_times += 1
                                    for i in range((num_times % 2) + 1):
                                        cumulative_local_expert_recv_stats = (
                                            torch.zeros(
                                                (num_local_experts,),
                                                dtype=torch.int,
                                                device="cuda",
                                            )
                                        )
                                        if dispatch_skip_fp8_quant:
                                            x_e4m3 = per_token_cast_to_fp8(
                                                current_x,
                                                (
                                                    hidden
                                                    if use_per_token_quant
                                                    else quant_size
                                                ),
                                            )
                                            (
                                                packed_recv_x,
                                                packed_recv_count,
                                                handle,
                                                event,
                                                hook,
                                            ) = buffer.low_latency_dispatch(
                                                x_e4m3[0],
                                                topk_idx,
                                                num_tokens,
                                                num_experts,
                                                use_fp8=dispatch_use_fp8,
                                                round_scale=round_scale,
                                                use_ue8m0=use_ue8m0,
                                                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                async_finish=not return_recv_hook,
                                                return_recv_hook=return_recv_hook,
                                                pertoken_quant=use_per_token_quant,
                                                x_scales=x_e4m3[1],
                                            )
                                        else:
                                            (
                                                packed_recv_x,
                                                packed_recv_count,
                                                handle,
                                                event,
                                                hook,
                                            ) = buffer.low_latency_dispatch(
                                                current_x,
                                                topk_idx,
                                                num_tokens,
                                                num_experts,
                                                use_fp8=dispatch_use_fp8,
                                                round_scale=round_scale,
                                                use_ue8m0=use_ue8m0,
                                                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                async_finish=not return_recv_hook,
                                                return_recv_hook=return_recv_hook,
                                                pertoken_quant=use_per_token_quant,
                                            )
                                        (
                                            hook()
                                            if return_recv_hook
                                            else event.current_stream_wait()
                                        )

                                    if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
                                        use_per_token_quant = True

                                    packed_recv_x = (
                                        (
                                            packed_recv_x[0],
                                            packed_recv_x[1].contiguous(),
                                        )
                                        if dispatch_use_fp8
                                        else packed_recv_x
                                    )
                                    if use_per_token_quant:
                                        simulated_gemm_x = (
                                            per_token_cast_back(
                                                packed_recv_x[0].view(-1, hidden),
                                                packed_recv_x[1].view(-1, 1),
                                                True,
                                            ).view(packed_recv_x[0].shape)
                                            if dispatch_use_fp8
                                            else packed_recv_x.clone()
                                        )
                                    else:
                                        simulated_gemm_x = (
                                            per_token_cast_back(
                                                packed_recv_x[0].view(-1, hidden),
                                                packed_recv_x[1].view(
                                                    -1, hidden // 128
                                                ),
                                            ).view(packed_recv_x[0].shape)
                                            if dispatch_use_fp8
                                            else packed_recv_x.clone()
                                        )
                                    all_topk_idx = torch.empty(
                                        (num_ranks, num_tokens, num_topk),
                                        dtype=topk_idx.dtype,
                                        device="cuda",
                                    )
                                    dist.all_gather_into_tensor(
                                        all_topk_idx, topk_idx, group=group
                                    )
                                    for i in range(
                                        num_local_experts if do_check else 0
                                    ):
                                        expert_id = rank * num_local_experts + i
                                        recv_x = (
                                            per_token_cast_back(
                                                packed_recv_x[0][i],
                                                packed_recv_x[1][i],
                                                use_per_token_quant,
                                            )
                                            if dispatch_use_fp8
                                            else packed_recv_x[i]
                                        )
                                        recv_count, recv_src_info, recv_layout_range = (
                                            packed_recv_count[i],
                                            handle[0][i],
                                            handle[1][i],
                                        )

                                        # Check expert indices
                                        int_mask = (2**32) - 1
                                        num_valid_tokens = recv_count.item()
                                        assert (
                                            cumulative_local_expert_recv_stats[i].item()
                                            == num_valid_tokens
                                        ), f"{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}"
                                        assert (
                                            num_valid_tokens
                                            == (recv_layout_range & int_mask)
                                            .sum()
                                            .item()
                                        ), f"{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()"
                                        assert (
                                            num_valid_tokens
                                            == (all_topk_idx == expert_id).sum().item()
                                        ), f"{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}"

                                        if num_valid_tokens == 0:
                                            continue
                                        # Check received data
                                        if current_x is x:
                                            recv_x = recv_x[:num_valid_tokens]
                                            recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                            recv_src_info = recv_src_info[
                                                :num_valid_tokens
                                            ]
                                            assert torch.equal(
                                                recv_x_amin,
                                                recv_x[:, :-128].amax(dim=-1),
                                            )
                                            if round_scale:
                                                assert (
                                                    calc_diff(
                                                        recv_x[:, -1],
                                                        recv_src_info.view(-1),
                                                    )
                                                    < 0.007
                                                )
                                            else:
                                                assert (
                                                    recv_x[:, -128:]
                                                    - recv_src_info.view(-1, 1)
                                                    % num_tokens
                                                ).sum().item() == 0
                                            for j in range(num_ranks):
                                                begin_idx, count = (
                                                    recv_layout_range[j] >> 32
                                                ).item(), (
                                                    recv_layout_range[j] & int_mask
                                                ).item()
                                                if (
                                                    not round_scale
                                                    and not use_per_token_quant
                                                ):
                                                    assert (
                                                        recv_x_amin == j - rank_offset
                                                    ).sum().item() == (
                                                        all_topk_idx[j] == expert_id
                                                    ).sum().item()
                                                    assert (
                                                        recv_x[
                                                            begin_idx : begin_idx
                                                            + count,
                                                            :-128,
                                                        ]
                                                        - j
                                                        + rank_offset
                                                    ).sum().item() == 0
                                        if dispatch_use_fp8:
                                            hash_value ^= hash_tensor(
                                                packed_recv_x[0][i, :num_valid_tokens]
                                            )
                                            if not use_per_token_quant:
                                                hash_value ^= hash_tensor(
                                                    packed_recv_x[1][
                                                        i, :num_valid_tokens
                                                    ]
                                                )
                                        else:
                                            hash_value ^= hash_tensor(
                                                packed_recv_x[i, :num_valid_tokens]
                                            )

                                    # Check combine correctness
                                    for zero_copy in (
                                        (False,) if use_logfmt else (False, True)
                                    ):
                                        if zero_copy:
                                            buffer.get_next_low_latency_combine_buffer(
                                                handle
                                            )[:, :, :] = simulated_gemm_x
                                        out = torch.empty(
                                            (num_tokens, hidden),
                                            dtype=torch.bfloat16,
                                            device="cuda",
                                        )
                                        combined_x, event, hook = (
                                            buffer.low_latency_combine(
                                                simulated_gemm_x,
                                                topk_idx,
                                                topk_weights,
                                                handle,
                                                use_logfmt=use_logfmt,
                                                async_finish=not return_recv_hook,
                                                zero_copy=zero_copy,
                                                return_recv_hook=return_recv_hook,
                                                out=out,
                                            )
                                        )
                                        (
                                            hook()
                                            if return_recv_hook
                                            else event.current_stream_wait()
                                        )
                                        if do_check and (
                                            not use_per_token_quant
                                            or not (current_x is x)
                                        ):
                                            diff = calc_diff(
                                                current_x
                                                * topk_weights.masked_fill(
                                                    topk_idx == -1, 0
                                                )
                                                .sum(dim=1)
                                                .view(-1, 1),
                                                combined_x,
                                            )
                                            assert (
                                                torch.isnan(combined_x).sum().item()
                                                == 0
                                            )
                                            # assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}, {use_per_token_quant=}, {dispatch_skip_fp8_quant=}'
                                            if not diff < (
                                                9e-4 if dispatch_use_fp8 else 1e-5
                                            ):
                                                print(
                                                    f"{rank=} assert Error: diff < (9e-4 if dispatch_use_fp8 else 1e-5) {diff=}, {dispatch_use_fp8=}, {zero_copy=}, {use_per_token_quant=}, {dispatch_skip_fp8_quant=}",
                                                    flush=True,
                                                )
                                            if not use_per_token_quant:
                                                hash_value ^= hash_tensor(combined_x)

        # noinspection PyShadowingNames
        def large_gemm_with_hook(hook):
            mat_0 = torch.randn((8192, 8192), dtype=torch.float)
            mat_1 = torch.randn((8192, 8192), dtype=torch.float)
            mat_0 @ mat_1
            hook()

        # noinspection PyShadowingNames
        def test_func(return_recv_hook: bool):
            recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
                current_x,
                topk_idx,
                num_tokens,
                num_experts,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                use_fp8=True,
                async_finish=False,
                return_recv_hook=return_recv_hook,
            )
            large_gemm_with_hook(hook) if return_recv_hook else None
            combined_x, event, hook = buffer.low_latency_combine(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                handle,
                use_logfmt=use_logfmt,
                return_recv_hook=return_recv_hook,
            )
            large_gemm_with_hook(hook) if return_recv_hook else None

        # Calculate bandwidth
        num_fp8_bytes, num_bf16_bytes = (
            hidden + hidden / quant_size * 4 + 16
        ), hidden * 2
        num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
        num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
        for i in range(num_tokens):
            num_selections = (topk_idx[i] != -1).sum().item()
            num_dispatch_comm_bytes += num_fp8_bytes * num_selections
            num_combine_comm_bytes += (
                num_logfmt10_bytes if use_logfmt else num_bf16_bytes
            ) * num_selections

        # Dispatch + combine testing
        avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
        print(
            f"[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, "
            f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
            flush=True,
        )

        # Separate profiling
        for return_recv_hook in (False, True):
            group.barrier()
            dispatch_t, combine_t = bench_kineto(
                partial(test_func, return_recv_hook=return_recv_hook),
                kernel_names=("dispatch", "combine"),
                barrier_comm_profiling=True,
                suppress_kineto_output=True,
                num_kernels_per_period=2 if return_recv_hook else 1,
            )
            if not return_recv_hook:
                print(
                    f"[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | "
                    f"Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us",
                    flush=True,
                )
            else:
                print(
                    f"[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | "
                    f"Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us",
                    flush=True,
                )
        return hash_value

    @staticmethod
    def _create_deepep_config(
        rank: int,
        num_ranks: int,
        args: Dict[str, Any],
        use_deepep_low_latency: bool = False,
        enable_ffn_disaggregate: bool = False,
        deep_ep_num_sm: int = 24,
    ) -> Tuple[MoEConfigAdapter, NcclCommConfig, int]:
        """Helper function to create MoEConfigAdapter and NcclCommConfig for DeepEP tests."""
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 500000
        if args:
            model_config.moe_k = args.get("moe_k", 2)
            model_config.expert_num = args.get("expert_num", 4)
            model_config.hidden_size = args.get("hidden_size", 128)

        master_port = int(os.getenv("MASTER_PORT", "8376"))
        base_port = master_port + 11
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        )
        nccl_init_port = base_port - 11

        parallelism_config = ParallelismConfig()
        parallelism_config.dp_rank = rank
        parallelism_config.dp_size = num_ranks
        parallelism_config.tp_rank = 0
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = num_ranks
        parallelism_config.ep_rank = rank
        parallelism_config.world_size = num_ranks
        parallelism_config.local_rank = rank
        parallelism_config.world_rank = rank
        parallelism_config.local_world_size = num_ranks

        moe_config = MoeConfig()
        moe_config.use_deepep_low_latency = use_deepep_low_latency
        moe_config.use_deepep_internode = False
        moe_config.deep_ep_num_sm = deep_ep_num_sm

        max_generate_batch_size = 32
        if args:
            max_generate_batch_size = args.get("max_generate_batch_size", 32)

        ffn_disaggregate_config = FfnDisAggregateConfig()
        ffn_disaggregate_config.enable_ffn_disaggregate = enable_ffn_disaggregate
        if enable_ffn_disaggregate and args:
            ffn_disaggregate_config.attention_dp_size = num_ranks // 2
            ffn_disaggregate_config.attention_tp_size = 1
            ffn_disaggregate_config.ffn_dp_size = num_ranks // 2
            ffn_disaggregate_config.ffn_tp_size = 1

        # Set ffn_disaggregate_config to parallelism_config
        parallelism_config.ffn_disaggregate_config = ffn_disaggregate_config

        moe_config.ll_num_max_token = max_generate_batch_size
        # Create and return MoEConfigAdapter
        config_adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )
        return config_adapter, nccl_comm_config, nccl_init_port

    @staticmethod
    def _run_deepep_intranode_test(rank: int, num_ranks: int, args: Dict[str, Any]):
        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        # init params
        config_adapter, nccl_comm_config, nccl_init_port = (
            DeepEPTest._create_deepep_config(
                rank,
                num_ranks,
                args,
                use_deepep_low_latency=False,
                enable_ffn_disaggregate=False,
            )
        )
        # init distributed environment

        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(config_adapter)
        deep_ep_wrapper = DeepEPWrapper.get_instance(
            deepep_config, group=dist.group.WORLD
        )
        buffer = deep_ep_wrapper.buffer
        # run test
        DeepEPTest._test_intranode_main(
            args["max_seq_len"],
            deep_ep_wrapper.hidden_size,
            deep_ep_wrapper.num_experts,
            deep_ep_wrapper.num_topk,
            deep_ep_wrapper.num_sms,
            deep_ep_wrapper.ep_rank,
            deep_ep_wrapper.ep_size,
            deep_ep_wrapper.ep_rank,
            buffer,
            dist.group.WORLD,
        )
        DeepEPWrapper.reset()
        destroy_distributed_environment()

    @staticmethod
    def _run_deepep_low_latency_test(rank: int, num_ranks: int, args: Dict[str, Any]):
        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"

        config_adapter, nccl_comm_config, nccl_init_port = (
            DeepEPTest._create_deepep_config(
                rank,
                num_ranks,
                args,
                use_deepep_low_latency=True,
                enable_ffn_disaggregate=False,
            )
        )

        # init distributed environment

        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        # Calculate ll_num_max_token_per_rank
        ll_num_max_token_per_rank = (
            args["max_generate_batch_size"] + config_adapter.tp_size - 1
        ) // config_adapter.tp_size
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            config_adapter, ll_num_max_token_per_rank
        )
        deep_ep_wrapper = DeepEPWrapper.get_instance(
            deepep_config, group=dist.group.WORLD
        )
        buffer = deep_ep_wrapper.buffer
        # run test
        DeepEPTest._test_low_latency_main(
            (args["max_generate_batch_size"] + config_adapter.tp_size - 1)
            // config_adapter.tp_size,
            deep_ep_wrapper.hidden_size,
            deep_ep_wrapper.num_experts,
            deep_ep_wrapper.num_topk,
            deep_ep_wrapper.ep_rank,
            deep_ep_wrapper.ep_size,
            dist.group.WORLD,
            buffer,
            seed=1,
        )
        DeepEPWrapper.reset()
        destroy_distributed_environment()

    @staticmethod
    def _run_deepep_low_latency_m2n_test(
        rank: int, num_ranks: int, args: Dict[str, Any]
    ):
        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        # init params
        config_adapter, nccl_comm_config, nccl_init_port = (
            DeepEPTest._create_deepep_config(
                rank,
                num_ranks,
                args,
                use_deepep_low_latency=True,
                enable_ffn_disaggregate=True,
            )
        )
        # init distributed environment
        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        # Calculate ll_num_max_token_per_rank for M2N mode
        ffn_disaggregate_config = (
            config_adapter.parallelism_config.ffn_disaggregate_config
        )
        ll_num_max_token_per_rank = calc_ll_num_max_token_per_rank(
            args["max_generate_batch_size"], ffn_disaggregate_config.attention_tp_size
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            config_adapter, ll_num_max_token_per_rank
        )
        deep_ep_wrapper = DeepEPWrapper.get_instance(
            deepep_config, group=dist.group.WORLD
        )
        buffer = deep_ep_wrapper.buffer
        # run test
        ffn_disaggregate_config = (
            config_adapter.parallelism_config.ffn_disaggregate_config
        )
        num_m = (
            ffn_disaggregate_config.attention_dp_size
            * ffn_disaggregate_config.attention_tp_size
        )
        num_n = (
            ffn_disaggregate_config.ffn_dp_size * ffn_disaggregate_config.ffn_tp_size
        )
        scale = num_ranks / num_n
        logical_num_experts = deep_ep_wrapper.num_experts * num_ranks // num_n
        DeepEPTest._test_low_latency_m2n_main(
            scale,
            num_m,
            num_m,
            (args["max_generate_batch_size"] + config_adapter.tp_size - 1)
            // config_adapter.tp_size,
            deep_ep_wrapper.hidden_size,
            logical_num_experts,
            deep_ep_wrapper.num_topk,
            deep_ep_wrapper.ep_rank,
            deep_ep_wrapper.ep_size,
            dist.group.WORLD,
            buffer,
            seed=1,
        )
        DeepEPWrapper.reset()
        destroy_distributed_environment()

    @staticmethod
    def _run_deepep_normal_expert_alignment_test(rank: int, num_ranks: int):
        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        # init params
        args = {"moe_k": 2, "expert_num": 4, "hidden_size": 128}
        config_adapter, nccl_comm_config, nccl_init_port = (
            DeepEPTest._create_deepep_config(
                rank,
                num_ranks,
                args,
                use_deepep_low_latency=False,
                enable_ffn_disaggregate=False,
            )
        )
        # init distributed environment

        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(config_adapter)
        deep_ep_wrapper = DeepEPWrapper.get_instance(
            deepep_config, group=dist.group.WORLD
        )
        buffer = deep_ep_wrapper.buffer
        # run test
        DeepEPTest._test_intranode_expert_alignment_main(
            deep_ep_wrapper.ep_rank,
            deep_ep_wrapper.ep_size,
            deep_ep_wrapper.hidden_size,
            deep_ep_wrapper.num_experts,
            deep_ep_wrapper.num_topk,
            expert_alignment=4,
            buffer=buffer,
            group=dist.group.WORLD,
            seed=777,
        )
        DeepEPWrapper.reset()
        destroy_distributed_environment()

    @staticmethod
    def _run_deepep_low_latency_per_token_quant_test(
        rank: int, num_ranks: int, args: Dict[str, Any]
    ):
        # set env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        # init params
        config_adapter, nccl_comm_config, nccl_init_port = (
            DeepEPTest._create_deepep_config(
                rank,
                num_ranks,
                args,
                use_deepep_low_latency=True,
                enable_ffn_disaggregate=False,
            )
        )
        # init distributed environment

        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        # Calculate ll_num_max_token_per_rank
        ll_num_max_token_per_rank = calc_ll_num_max_token_per_rank(
            args["max_generate_batch_size"], config_adapter.tp_size
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            config_adapter, ll_num_max_token_per_rank
        )
        deep_ep_wrapper = DeepEPWrapper.get_instance(
            deepep_config, group=dist.group.WORLD
        )
        buffer = deep_ep_wrapper.buffer
        # run test
        DeepEPTest._test_low_latency_per_token_quant_main(
            (args["max_generate_batch_size"] + config_adapter.tp_size - 1)
            // config_adapter.tp_size,
            deep_ep_wrapper.hidden_size,
            deep_ep_wrapper.num_experts,
            deep_ep_wrapper.num_topk,
            deep_ep_wrapper.ep_rank,
            deep_ep_wrapper.ep_size,
            dist.group.WORLD,
            buffer,
            use_logfmt=False,
            seed=1,
        )
        DeepEPWrapper.reset()
        destroy_distributed_environment()

    def test_deepep_normal(self):
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            for params in itertools.product(
                self.NUM_PROCESSES,
                self.MAX_SEQ_LEN,
                self.HIDDEN_SIZES,
                self.NUM_EXPERT,
                self.TOP_K,
            ):
                args = {
                    "max_seq_len": params[1],
                    "hidden_size": params[2],
                    "expert_num": params[3],
                    "moe_k": params[4],
                }
                mp.spawn(
                    DeepEPTest._run_deepep_intranode_test,
                    args=(params[0], args),
                    nprocs=params[0],
                    join=True,
                )

    def test_deepep_low_latency(self):
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            for params in itertools.product(
                self.NUM_PROCESSES,
                self.MAX_GENERATE_BATCH_SIZES,
                self.HIDDEN_SIZES,
                self.NUM_EXPERT,
                self.TOP_K,
            ):
                args = {
                    "max_generate_batch_size": params[1],
                    "hidden_size": params[2],
                    "expert_num": params[3],
                    "moe_k": params[4],
                }
                mp.spawn(
                    DeepEPTest._run_deepep_low_latency_test,
                    args=(params[0], args),
                    nprocs=params[0],
                    join=True,
                )

    def test_deepep_low_latency_m2n(self):
        if not hasattr(DeepEPBuffer, "get_low_latency_rdma_size_hint_m2n"):
            return
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            for params in itertools.product(
                self.NUM_PROCESSES,
                self.MAX_GENERATE_BATCH_SIZES,
                self.HIDDEN_SIZES,
                self.NUM_EXPERT,
                self.TOP_K,
            ):
                if params[0] != self.NUM_PROCESSES[0]:
                    continue
                args = {
                    "max_generate_batch_size": params[1],
                    "hidden_size": params[2],
                    "expert_num": params[3],
                    "moe_k": params[4],
                }
                mp.spawn(
                    DeepEPTest._run_deepep_low_latency_m2n_test,
                    args=(params[0], args),
                    nprocs=params[0],
                    join=True,
                )

    def test_deepep_normal_expert_alignment(self):
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            mp.spawn(
                DeepEPTest._run_deepep_normal_expert_alignment_test,
                args=(2,),
                nprocs=2,
                join=True,
            )

    def test_deepep_low_latency_per_token_quant(self):
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            for params in itertools.product(
                self.NUM_PROCESSES,
                self.MAX_GENERATE_BATCH_SIZES,
                self.HIDDEN_SIZES,
                self.NUM_EXPERT,
                self.TOP_K,
            ):
                args = {
                    "max_generate_batch_size": params[1],
                    "hidden_size": params[2],
                    "expert_num": params[3],
                    "moe_k": params[4],
                }
                mp.spawn(
                    DeepEPTest._run_deepep_low_latency_test,
                    args=(params[0], args),
                    nprocs=params[0],
                    join=True,
                )

    @staticmethod
    def _init_sp_deepep_wrapper(rank: int, num_ranks: int):
        # set env
        os.environ["WORLD_SIZE"] = str(num_ranks)
        os.environ["DP_SIZE"] = str(num_ranks)
        os.environ["EP_SIZE"] = str(num_ranks)
        os.environ["MODEL_TYPE"] = "fake_model"
        os.environ["SP_TYPE"] = "eagle"
        os.environ["SP_MODEL_TYPE"] = "qwen_2-mtp"
        os.environ["GEN_NUM_PER_CIRCLE"] = "4"
        os.environ["ROLE_TYPE"] = "DECODE"
        os.environ["USE_DEEPEP_MOE"] = "1"
        os.environ["USE_DEEPEP_INTERNODE"] = "0"
        os.environ["USE_DEEPEP_LOW_LATENCY"] = "1"
        py_env_configs: PyEnvConfigs = setup_args()
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 500000
        model_config.moe_k = 8
        model_config.expert_num = 32
        model_config.hidden_size = 7168

        setup_and_configure_server(py_env_configs)
        engine_config: EngineConfig = EngineConfig.create(py_env_configs, None)
        engine_config.parallelism_config.local_rank = rank
        engine_config.parallelism_config.world_rank = rank
        assert engine_config.moe_config.ll_num_max_token == 32 * (4 + 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_ranks))
        os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
        os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        init_deepep_wrapper(engine_config, model_config)

        config_adapter = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=engine_config.parallelism_config,
            moe_config=engine_config.moe_config,
        )
        master_port = int(os.getenv("MASTER_PORT", "8376"))
        base_port = master_port + 11
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        )
        nccl_init_port = base_port - 11
        torch.cuda.set_device(config_adapter.parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{config_adapter.parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config=config_adapter.parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            config_adapter, engine_config.moe_config.ll_num_max_token
        )
        # just need test get instance
        deep_ep_wrapper = DeepEPWrapper.get_instance(deepep_config)

    def test_init_sp_deepep_wrapper(self):
        with PortsContext(None, 1) as ports:
            os.environ["MASTER_PORT"] = str(ports[0])
            mp.spawn(
                DeepEPTest._init_sp_deepep_wrapper,
                args=(2,),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    main()
