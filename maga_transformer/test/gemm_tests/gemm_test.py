
import torch
import sys
import os
import argparse
import numpy as np
import os
import logging

from maga_transformer.utils.gemm_utils.device_map import DeviceMap
from maga_transformer.utils.gemm_utils.device_map import get_device

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class GemmConfigSelect(object):
    def setUp(self) -> None:
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/maga_transformer/libs/libth_transformer.so")
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")
        self.gemm_config_select_ = torch.ops.gemm_dq_unit_ops.gemm_config_select
        self.int8_gemm_config_select_ = torch.ops.int8_gemm_ops.int8_gemm_config_select
        torch.manual_seed(734876213)

    def fp16_int8_gemm_select(self, compute_type, gemm_m, gemm_k, gemm_n, group_size, iterations):
        gemm_kn = list(zip(gemm_k, gemm_n))
        for k,n in gemm_kn:
            max_m = gemm_m
            weights = torch.randn((k, n)).normal_(0, 128).to(torch.int8).cuda()
            zeros = torch.randn((k // group_size, n), dtype=compute_type).cuda()
            scales = torch.randn((n), dtype=compute_type).cuda()
            for m in range(1, max_m+1):
                activations = torch.randn((m, k), dtype=compute_type).cuda()

                self.gemm_config_select_(activations, weights, scales, zeros, group_size, iterations)

    def fp16_int4_gemm_select(self, compute_type, gemm_m, gemm_k, gemm_n, group_size, iterations):
        gemm_kn = list(zip(gemm_k, gemm_n))
        for k,n in gemm_kn:
            max_m = gemm_m
            weights = torch.randn((k, n // 2)).normal_(0, 128).to(torch.int8).cuda()
            zeros = torch.randn((k // group_size, n), dtype=compute_type).cuda()
            scales = torch.randn((k // group_size, n), dtype=compute_type).cuda()
            for m in range(1, max_m+1):
                activations = torch.randn((m, k), dtype=compute_type).cuda()

                self.gemm_config_select_(activations, weights, scales, zeros, group_size, iterations)

    def w8a8_gemm_select(self, gemm_m, gemm_k, gemm_n, iterations):
        gemm_kn = list(zip(gemm_k, gemm_n))
        for k,n in gemm_kn:
            max_m = gemm_m
            mat2 = torch.randint(-128, 128, (n, k), dtype=torch.int8).cuda()
            shape_scale_b = (1, n)
            scale_b_torch = torch.ones(shape_scale_b, dtype=torch.float32) * 1e-2
            scale_b_torch *= torch.randint(1,
                                           10,
                                           shape_scale_b,
                                           dtype=torch.float32)
            scale_b_torch = scale_b_torch.cuda()
            for m in range(1, max_m+1):
                mat1 = torch.randint(-128, 128, (m, k), dtype=torch.int8).cuda()

                shape_scale_a = (m, 1)
                scale_a_torch = torch.ones(shape_scale_a, dtype=torch.float32) * 1e-2
                scale_a_torch *= torch.randint(1,
                                               10,
                                               shape_scale_a,
                                               dtype=torch.float32)
                scale_a_torch = scale_a_torch.cuda()

                self.int8_gemm_config_select_(mat1, mat2, scale_b_torch, scale_a_torch, iterations)

    def gemm_config_select_helper(self, prec, gemm_m, gemm_k, gemm_n, group_size = 128, iterations = 20):
        compute_type = torch.float16
        if prec == 'int8':
            self.fp16_int8_gemm_select(compute_type, gemm_m, gemm_k, gemm_n, group_size, iterations)
        elif prec == 'int4':
            self.fp16_int4_gemm_select(compute_type, gemm_m, gemm_k, gemm_n, group_size, iterations)
        elif prec == 'w8a8':
            self.w8a8_gemm_select(gemm_m, gemm_k, gemm_n, iterations)
        else:
            raise Exception('undefined prec')

def rename_file(prec, model_size):
    device_name = torch.cuda.get_device_name(0)
    device = get_device(device_name)

    file_name = "_".join([device, prec, model_size, "config.ini"])

    return file_name

def upload_to_oss(file_name):
    oss_id = os.environ.get("OSS_ID")
    oss_key = os.environ.get("OSS_KEY")
    oss_host = os.environ.get("OSS_HOST")
    oss_out_path = os.environ.get("OSS_OUT_PATH")

    assert oss_host is not None
    assert oss_id is not None
    assert oss_key is not None
    os.system(f"osscmd config  --host={oss_host} --id={oss_id} --key={oss_key}")

    dest_path = oss_out_path + "/" + file_name

    logging.info("upload config.ini to oss: " + dest_path)
    os.system(f"osscmd put config.ini {dest_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_m', type=str, required=True)
    parser.add_argument('--prec', type=str, required=True)
    parser.add_argument('--k', type=str, required=True)
    parser.add_argument('--n', type=str, required=True)
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--upload_to_oss', type=str, required=True)

    args, _ = parser.parse_known_args()

    t_case = GemmConfigSelect()
    t_case.setUp()
    t_case.gemm_config_select_helper(args.prec, int(args.max_m), [int(x) for x in args.k.split(',')], [int(x) for x in args.n.split(',')])

    config_file = rename_file(args.prec, args.model_size)
    if args.upload_to_oss == "True":
        upload_to_oss(config_file)
    else:
        os.rename("config.ini", config_file)