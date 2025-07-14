import logging

from rtp_llm.utils.gemm_utils.device_map import DeviceMap

try:
    import internal_source.rtp_llm.utils.device_map
except:
    logging.info("internal devices not found")
