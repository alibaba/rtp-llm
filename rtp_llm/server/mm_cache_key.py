"""Linux serving cache identity, matching MultimodalInputClass.h sentinel values."""

_MASK = (1 << 64) - 1


def linux_string_hash(value):
    data = value.encode("utf-8")
    multiplier = 0xC6A4A7935BD1E995
    result = 0xC70F6907 ^ (len(data) * multiplier & _MASK)
    full = len(data) // 8 * 8
    for offset in range(0, full, 8):
        word = int.from_bytes(data[offset : offset + 8], "little")
        word = word * multiplier & _MASK
        word ^= word >> 47
        word = word * multiplier & _MASK
        result = ((result ^ word) * multiplier) & _MASK
    if full < len(data):
        result = ((result ^ int.from_bytes(data[full:], "little")) * multiplier) & _MASK
    result ^= result >> 47
    result = result * multiplier & _MASK
    return result ^ (result >> 47)


def multimodal_key(image):
    config = image.mm_preprocess_config
    crop = ":".join(f"{x:.6f}" for x in config.crop_positions)
    long_side = getattr(config, "max_long_side_pixel", -1)
    values = [
        linux_string_hash(image.multimodal_url), image.multimodal_type,
        config.width, config.height, config.min_pixels, config.max_pixels,
        config.fps, config.min_frames, config.max_frames, crop,
        long_side if long_side > 0 else -1,
    ]
    return "_".join(str(value) for value in values)
