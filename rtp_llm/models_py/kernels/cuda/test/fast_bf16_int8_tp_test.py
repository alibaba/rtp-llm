"""Real NCCL TP2 coverage of the production INT8 all-reduce helper, without mocks."""

import faulthandler
import itertools
import time
import unittest

import torch
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.kernels.cuda.fast_bf16_int8 import (
    dequantize_reduce,
    quantize,
    support,
)
from rtp_llm.models_py.kernels.cuda.test.fast_bf16_int8_test_utils import (
    assert_bits_equal,
    rank_inputs,
    reference_reduce,
)
from rtp_llm.models_py.model_desc.quantized_all_reduce import _quantized_all_reduce
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager


def _worker(rank, port):
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_rank = parallel.local_rank = parallel.tp_rank = rank
    parallel.world_size = parallel.tp_size = 2
    parallel.dp_size = 1
    comm = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=port + 1,
        dp_tp_nccl_port=port + 2,
        ffn_tp_nccl_port=port + 3,
    )
    init_distributed_environment(parallel, comm, port, timeout=60)
    print(f"TP2 rank {rank}: initialized NCCL", flush=True)
    try:
        for m, factor, dtype in itertools.product(
            (1, 3, 8, 83), (0.01, 1.0, 100.0), (torch.bfloat16, torch.float16)
        ):
            inputs = rank_inputs(m, 5120, factor)
            expected_q, expected_s, expected_out = reference_reduce(inputs, dtype)
            x = inputs[rank].cuda()
            assert support(x), "test must exercise the INT8 path"
            gathered = {}

            def decode(q, scales, out, *, group_size):
                assert q.dtype == torch.int8
                assert scales.dtype == dtype
                gathered.update(q=q, scales=scales)
                dequantize_reduce(q, scales, out, group_size=group_size)

            # Both all_gather_into_tensor calls execute the actual NCCL backend.
            kwargs = {} if dtype == torch.bfloat16 else {"scale_dtype": dtype}
            result = _quantized_all_reduce(x, quantize, decode, **kwargs)
            assert result is x
            assert_bits_equal(gathered["q"], expected_q)
            assert_bits_equal(gathered["scales"], expected_s)
            assert_bits_equal(result, expected_out)
            if rank == 0:
                print(f"TP2 eager: m={m}, factor={factor}, scale={dtype}", flush=True)

        # Capture the actual helper, including both collectives and its allocations.
        # Callbacks only retain buffers; no host reads occur during capture.
        for dtype in (torch.bfloat16, torch.float16):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            gathered = {}
            with torch.cuda.stream(stream):
                x = rank_inputs(3, 5120, 1.0)[rank].cuda()

                def decode(q, scales, out, *, group_size):
                    gathered.update(q=q, scales=scales)
                    dequantize_reduce(q, scales, out, group_size=group_size)

                def run():
                    _quantized_all_reduce(x, quantize, decode, scale_dtype=dtype)

                for _ in range(3):
                    x.copy_(rank_inputs(3, 5120, 1.0)[rank])
                    run()
                stream.synchronize()
                torch.distributed.barrier()
                print(f"TP2 rank {rank}: capture {dtype}", flush=True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run()
                for factor in (0.01, 100.0):
                    inputs = rank_inputs(3, 5120, factor)
                    expected_q, expected_s, expected_out = reference_reduce(
                        inputs, dtype
                    )
                    x.copy_(inputs[rank])
                    graph.replay()
                    stream.synchronize()
                    assert_bits_equal(gathered["q"], expected_q)
                    assert_bits_equal(gathered["scales"], expected_s)
                    assert_bits_equal(x, expected_out)
                    if rank == 0:
                        print(f"TP2 graph: factor={factor}, scale={dtype}", flush=True)
            stream.synchronize()
            # A captured NCCL graph retains its communicator. Release the graph
            # before destroying the process group, otherwise teardown can hang.
            graph.reset()
            del graph
        torch.cuda.synchronize()
        torch.distributed.barrier()
    finally:
        destroy_distributed_environment()
        faulthandler.cancel_dump_traceback_later()


class FastBf16Int8TpTest(unittest.TestCase):
    def test_real_tp2_eager_and_cuda_graph(self):
        # This target explicitly reserves two SM120 GPUs. Missing resources must
        # fail, rather than silently pass with the communication tests skipped.
        self.assertIsNone(torch.version.hip)
        self.assertTrue(torch.cuda.is_available(), "CUDA is required")
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "TP2 requires two GPUs")
        for rank in range(2):
            self.assertIn("RTX PRO 5000", torch.cuda.get_device_name(rank).upper())
        ports, locks = PortManager().get_consecutive_ports(4)
        context = None
        try:
            context = mp.spawn(_worker, args=(ports[0],), nprocs=2, join=False)
            deadline = time.monotonic() + 240
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    self.fail("TP2 workers exceeded 240 seconds")
        finally:
            if context is not None:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
