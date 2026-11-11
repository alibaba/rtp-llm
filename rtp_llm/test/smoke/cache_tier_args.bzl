# Store through DEVICE, then demote idle entries so later queries reuse HOST.
# Keep the model's physical pools unchanged. Block counts fit in uint32, so
# these watermarks yield high=ceil(N*2e-12)=1 and low=floor(N*1e-12)=0.
HOST_REUSE_AFTER_DEMOTION_ARGS = (
    " --enable_device_cache 1 --enable_memory_cache 1" +
    " --block_tree_device_evict_low_watermark_ratio 1e-12" +
    " --block_tree_device_evict_high_watermark_ratio 2e-12"
)
