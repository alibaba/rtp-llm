import torch
import glob
import re
import logging
from enum import Enum
from maga_transformer.utils.gemm_utils.device_map import get_device, get_lut_info

def split_config(s):
    pattern = r'\{\{(\d+),(\d+),(\d+)\}, \{CutlassGemmConfig\((CutlassTileConfig::CtaShape[^,)]+),(\d+),(\d+)\),.*?\}\}'
    match = re.search(pattern, s)

    if match:
        items = match.groups()
        return items
    else:
        raise Exception("invalid gemm config file")


def get_quant_info(quant_algo):
    if quant_algo.isWeightOnlyPerCol() and quant_algo.getWeightBits() == 8:
        return "int8"
    elif quant_algo.isGroupwise() and quant_algo.getWeightBits() == 4:
        return "int4"
    elif quant_algo.isSmoothQuant() or quant_algo.isOmniQuant():
        return "w8a8"
    raise Exception("quant info not supported")

def concat_config_file_name(quant_info):
    device_name = torch.cuda.get_device_name(0)
    device = get_device(device_name)
    pattern = "_".join([device, quant_info, "*", "config.ini"])
    return pattern

def load_cutlass_gemm_config(quant_algo):
    try:
        quant_info = get_quant_info(quant_algo=quant_algo)
        device_name = torch.cuda.get_device_name(0)
        lut_info = get_lut_info(device_name, quant_info)
        for config_file in lut_info.get_files():
            with open(config_file) as reader:
                logging.info("load cutlass gemm config: " + str(config_file))
                contents = reader.read().rstrip("\n").split("\n")
                for s in contents:
                    configs = split_config(s)
                    if quant_info == "int8":
                        torch.ops.rtp_llm.insert_fp16_int8_gemm_config(int(configs[0]), int(configs[1]), int(configs[2]), configs[3], int(configs[4]), int(configs[5]))
                    elif quant_info == "int4":
                        torch.ops.rtp_llm.insert_fp16_int4_gemm_config(int(configs[0]), int(configs[1]), int(configs[2]), configs[3], int(configs[4]), int(configs[5]))
                    elif quant_info == "w8a8":
                        torch.ops.rtp_llm.insert_w8a8_gemm_config(int(configs[0]), int(configs[1]), int(configs[2]), configs[3], int(configs[4]), int(configs[5]))
    except Exception as e:
        logging.warn("load cutlass gemm config failed: " + str(e))

