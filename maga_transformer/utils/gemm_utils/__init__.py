from maga_transformer.utils.gemm_utils.device_map import DeviceMap
import logging
try:
    from internal_source.maga_transformer.utils.device_map import *
except:
    logging.info("internal devices not found")
