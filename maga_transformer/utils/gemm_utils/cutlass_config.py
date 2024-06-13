import torch
import os
import glob
import re
import logging
from enum import Enum

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

def split_config(s):
    pattern = r'{\{(\d+),(\d+),(\d+)\}, \{CutlassGemmConfig\((CutlassTileConfig::CtaShape[^,)]+),(\d+),(\d+)\),.*?\}\}'
    match = re.search(pattern, s)
    
    if match:
        items = match.groups()
        return items
    else:
        raise Exception("invalid gemm config file")



class DeviceMap(Enum):
    A10 = ["NVIDIA A10"]
    A100 = ["NVIDIA A800-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]
    V100 = ["Tesla V100S-PCIE-32GB"]
    @classmethod
    def from_str(cls, value: str):
        for member in cls:
            if value in member.value:
                return member
        raise Exception("Device not supported")
        
    def to_str(self):
        return str(self).replace("DeviceMap.", "").lower()

def get_quant_info(quant_algo):
    if quant_algo.isWeightOnlyPerCol() and quant_algo.getWeightBits() == 8:
        return "int8"
    elif quant_algo.isGroupwise() and quant_algo.getWeightBits() == 4:
        return "int4"
    raise Exception("quant info not supported")

def concat_config_file_name(quant_algo):
    device_name = torch.cuda.get_device_name(0)
    device = DeviceMap.from_str(device_name).to_str()
    quant_info = get_quant_info(quant_algo=quant_algo)
    pattern = "_".join([device, quant_info, "*", "config.ini"])
    return pattern
     
def load_cutlass_gemm_config(quant_algo):
    try:
        config_pattern = concat_config_file_name(quant_algo=quant_algo)
        config_pattern = os.path.join(CUR_PATH, "luts", config_pattern)
        config_files = glob.glob(config_pattern)
        for config_file in config_files:
            with open(config_file) as reader:
                logging.info("load cutlass gemm config: " + str(config_file))
                contents = reader.read().rstrip("\n").split("\n")
                for s in contents:
                    configs = split_config(s)
                    torch.ops.fastertransformer.insert_fp16_int8_gemm_config(int(configs[0]), int(configs[1]), int(configs[2]), configs[3], int(configs[4]), int(configs[5]))
    except Exception as e: 
        logging.warn("load cutlass gemm config failed: " + str(e))
 
