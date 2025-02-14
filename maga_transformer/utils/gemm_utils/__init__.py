from maga_transformer.utils.gemm_utils.device_map import DeviceMap
import logging
try:
    import internal_source.maga_transformer.utils.device_map
except:
    logging.info("internal devices not found")
