from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm


GLM54_MOCK_IMAGE_TOKEN = "<|begin_of_image|><|image|><|end_of_image|>"

# The HTTP API names width and height differently from the native config.
UNSUPPORTED_IMAGE_OPTIONS = (
    ("resized_width", "width"),
    ("resized_height", "height"),
    ("max_long_side_pixel", "max_long_side_pixel"),
    ("fps", "fps"),
    ("min_frames", "min_frames"),
    ("max_frames", "max_frames"),
)


def validate_image_url(url: str) -> None:
    if not url.strip():
        raise_mm(MMErr.URL_INVALID)


def reject_image_option(name: str) -> None:
    raise_mm(f"GLM5.4 mock ViT does not support the {name} preprocessing option")


def validate_pixel_bounds(min_pixels: int, max_pixels: int) -> None:
    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if value != -1 and not 0 < value < 2**31:
            raise_mm(f"{name} must be a positive 32-bit integer or -1")
    if max_pixels != -1 and min_pixels > max_pixels:
        raise_mm("min_pixels must not exceed max_pixels")
