from aiter.test_common import checkAllclose
import multiprocessing as mp
import os
import torch
import unittest


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/rtp_llm/tests/librocm_test_ops.so")


def quick_ar(tp_rank, tp_size, x):
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"

    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)

    x = x.to(device)
    rtp_ret = torch.zeros_like(x)

    rocm_custom_ar_op = torch.classes.unittest.ROCmCustomAROp(tp_rank, tp_size)

    rocm_custom_ar_op.forward(x, rtp_ret)

    return rtp_ret.cpu().clone() # FIXME(liyangcheng.lyc) return gpu tensor may get wrong tensor


def custom_ar(tp_rank, tp_size, x):
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"

    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)

    x = x.to(device)
    rtp_ret = torch.zeros_like(x)

    rocm_custom_ar_op = torch.classes.unittest.ROCmCustomAROp(tp_rank, tp_size)

    rocm_custom_ar_op.forward(x, rtp_ret)

    return rtp_ret.cpu().clone() # FIXME(liyangcheng.lyc) return gpu tensor may get wrong tensor


class TestROCmCustomAR(unittest.TestCase):

    def _test_quick_ar(self, tp_size, dtype, shape):
        if tp_size == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        elif tp_size == 4:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        elif tp_size == 8:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

        print(f"[quick_ar]: {tp_size=}, {dtype=}, {shape=}")

        pool = mp.Pool(processes=tp_size)

        torch_ref = torch.zeros(shape, dtype=dtype)
        rtp_rets = []

        for tp_rank in range(tp_size):
            x = torch.randn(shape, dtype=dtype)

            torch_ref += x
            rtp_rets.append(
                pool.apply_async(quick_ar, args=(tp_rank, tp_size, x))
            )

        pool.close()
        pool.join()

        rtp_rets = [el.get() for el in rtp_rets]
        # NOTE(liyangcheng.lyc) quick ar use more relaxed thresholds
        atol = 1.25 * tp_size
        rtol = 0.5 * tp_size
        for rtp_ret in rtp_rets:
            msg = f"test_quick_ar: {tp_size=}, {dtype=}, {shape=}"
            checkAllclose(torch_ref, rtp_ret, msg=msg, atol=atol, rtol=rtol)


    def _test_custom_ar(self, tp_size, dtype, shape):
        if tp_size == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        elif tp_size == 4:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        elif tp_size == 8:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

        print(f"[custom_ar]: {tp_size=}, {dtype=}, {shape=}")

        pool = mp.Pool(processes=tp_size)

        torch_ref = torch.zeros(shape, dtype=dtype)
        rtp_rets = []

        for tp_rank in range(tp_size):
            x = torch.randn(shape, dtype=dtype)

            torch_ref += x
            rtp_rets.append(
                pool.apply_async(custom_ar, args=(tp_rank, tp_size, x))
            )

        pool.close()
        pool.join()

        rtp_rets = [el.get() for el in rtp_rets]
        for rtp_ret in rtp_rets:
            msg = f"test_custom_ar: {tp_size=}, {dtype=}, {shape=}"
            checkAllclose(torch_ref, rtp_ret, msg=msg)


    def test_custom_ar(self):
        for tp_size in [2, 4, 8]:
            for dtype in [torch.bfloat16]:
                for shape in [(4096, 4096), (1, 4096)]:
                    self._test_custom_ar(tp_size, dtype, shape)


    def test_quick_ar(self):
        for tp_size in [2, 4, 8]:
            for dtype in [torch.bfloat16]:
                for shape in [(4096, 4096), (1, 4096)]:
                    self._test_quick_ar(tp_size, dtype, shape)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    unittest.main()
