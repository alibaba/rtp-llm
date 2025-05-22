import glob
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class LutInfo(object):
    def __init__(self, glob: str):
        self._glob = glob
        self._cur_path = self._get_files_from_glob(glob)

    def get_files(self):
        return self._cur_path

    def _get_files_from_glob(self, file_pattern: str):
        config_files = glob.glob(file_pattern)
        return config_files

class DeviceMap(object):
    ALL_QUANT_INFOS = ["int4", "int8", "w8a8"]
    def __init__(self):
        self._device_map: Dict[str, List[str]] = {}
        self._lut_map: Dict[str, Dict[str, LutInfo]] = defaultdict(dict)

    def register(self, key: str, value: List[str], lut_dir_path: Optional[str] = None):
        if key in self._device_map:
            raise Exception(f"device {key} registered multi times")
        self._device_map[key] = value
        if not lut_dir_path:
            return
        for quant_info in self.ALL_QUANT_INFOS:
            file_pattern = self._generate_file_pattern(key, quant_info, lut_dir_path)
            self._lut_map[key][quant_info] = LutInfo(file_pattern)

    def _generate_file_pattern(self, device: str, quant_info: str, lut_dir_path: str):
        pattern = "_".join([device.lower(), quant_info, "*", "config.ini"])
        return os.path.join(lut_dir_path, pattern)

    def from_str(self, device_name: str):
        for k, v in self._device_map.items():
            if device_name in v:
                return k
        raise Exception(f"Device {device_name} not supported yet")

    def get_lut_info(self, device_name: str, quant_info: str):
        device = self.from_str(device_name)
        if device not in self._lut_map:
            raise Exception(f"device: {device} not in lut_map")
        if quant_info not in self._lut_map[device]:
            raise Exception(f"quant_info {quant_info} not in lut_map of device: {device_name}")
        return self._lut_map[device][quant_info]

_DEVICE_MAP = DeviceMap()


def register_device(key: str, value: List[str], path: Optional[str] = None):
    global _DEVICE_MAP
    _DEVICE_MAP.register(key, value, path)


def get_device(name: str):
    global _DEVICE_MAP
    return _DEVICE_MAP.from_str(name).lower()

def get_lut_info(name: str, quant_info: str) -> LutInfo:
    return _DEVICE_MAP.get_lut_info(name, quant_info)

_LUT_PATH = os.path.join(CUR_PATH, "luts")
register_device("A10", ["NVIDIA A10"], _LUT_PATH)
register_device("A100", ["NVIDIA A800-SXM4-80GB", "NVIDIA A100-SXM4-80GB"], _LUT_PATH)
register_device("V100", ["Tesla V100S-PCIE-32GB"], _LUT_PATH)
register_device("H20", ["NVIDIA H20"], None)
register_device("L40S", ["NVIDIA L40S"],None)