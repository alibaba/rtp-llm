from aiter.test_common import checkAllclose
import multiprocessing as mp
import os
import torch
import unittest


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/rtp_llm/tests/librocm_test_ops.so")


def custom_ag(tp_rank, tp_size, all_gather_input, all2all_input, inplace=False):
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
    os.environ["ROCM_DISABLE_CUSTOM_AG"] = "0"

    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)

    all_gather_input  = all_gather_input.to(device)
    all_gather_output = torch.zeros([tp_size, *all_gather_input.shape], dtype=all_gather_input.dtype, device=all_gather_input.device)

    all2all_input  = all2all_input.to(device)
    all2all_output = torch.zeros_like(all2all_input)

    rocm_custom_ag_op = torch.classes.unittest.ROCmCustomAGOp(tp_rank, tp_size)

    rocm_custom_ag_op.forward(all_gather_input, all_gather_output, inplace, all2all_input, all2all_output)

    return [all_gather_output.cpu().clone(), all2all_output.cpu().clone()] # FIXME(liyangcheng.lyc) return gpu tensor may get wrong tensor


class TestROCmCustomAG(unittest.TestCase):

    def _test_custom_ag(self, tp_size, dtype, shape, dim=0, inplace=False):
        # NOTE(liyangcheng.lyc): shape here denotes per-rank shape
        print(f"[custom_ag]: {tp_size=}, {dtype=}, {shape=}, {dim=}, {inplace=}")

        pool = mp.Pool(processes=tp_size)

        torch_all_gather_inputs = []
        rtp_all_gather_outputs = []

        torch_all2all_inputs = []
        rtp_all2all_outputs = []

        rtp_outputs = []

        for tp_rank in range(tp_size):
            torch_all_gather_input = torch.randn(shape, dtype=dtype)
            print(f"{torch_all_gather_input}")

            torch_all2all_input = torch.arange(tp_size, dtype=dtype)
            print(f"{torch_all2all_input=}")

            torch_all_gather_inputs.append(torch_all_gather_input)
            torch_all2all_inputs.append(torch_all2all_input)

            rtp_outputs.append(
                pool.apply_async(custom_ag, args=(tp_rank, tp_size, torch_all_gather_input, torch_all2all_input, inplace))
            )

        pool.close()
        pool.join()

        torch_all_gather_ref = torch_all_gather_inputs[0]
        torch_all2all_refs = []
        for i in range(1, tp_size):
            torch_all_gather_ref = torch.concat((torch_all_gather_ref, torch_all_gather_inputs[i]), dim=dim)

        for i in range(tp_size):
            torch_all2all_refs.append(torch.ones(tp_size, dtype=dtype) * i)


        rtp_outputs = [el.get() for el in rtp_outputs]
        for idx, rtp_output in enumerate(rtp_outputs):
            rtp_all_gather_output = rtp_output[0]
            rtp_all2all_output = rtp_output[1]

            rtp_all_gather_output = rtp_all_gather_output.movedim(0, dim)
            rtp_all_gather_output = rtp_all_gather_output.reshape(torch_all_gather_ref.shape)

            print(f"{torch_all_gather_ref=}, {rtp_all_gather_output=}")
            msg1 = f"test_custom_ar: {tp_size=}, {dtype=}, {shape=}, {dim=}, {inplace=}"
            checkAllclose(torch_all_gather_ref, rtp_all_gather_output, msg=msg1)

            print(f"{torch_all2all_refs[idx]=}, {rtp_all2all_output=}")
            msg2 = f"test_all2all: {tp_size=}, {dtype=}, {shape=}"
            checkAllclose(torch_all2all_refs[idx], rtp_all2all_output, msg=msg2)


    def test_custom_ag(self):
        for tp_size in [2, 4, 8]:
            for dtype in [torch.bfloat16]:
                for shape in [[8, 8], [1, 8]]:
                    for dim in [0, 1]:
                        for inplace in [True, False]:
                            tp_shape = shape[:dim] + [(shape[dim] + tp_size - 1) // tp_size] + shape[(dim + 1):]
                            self._test_custom_ag(tp_size, dtype, tp_shape, dim, inplace)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    unittest.main()
