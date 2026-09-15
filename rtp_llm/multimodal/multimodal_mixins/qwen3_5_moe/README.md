# Qwen3.5 video preprocessing

Qwen3.5 inherits Qwen3-VL image preprocessing. Videos use NVDEC by default;
set QWEN35_VIDEO_BACKEND=cpu to use the CPU decoder. NVDEC requires
PyNvVideoCodec and matching NVIDIA video driver libraries.

Workers fetch bytes and plan sampling/grid metadata on CPU. The embedding
scheduler decodes with NVDEC, uses CPU uint8 resize to match the reference
rounding, then normalizes and patchifies on GPU.

Sampling, resize and timestamp rules share qwen3_vl_video.py:
- FPS defaults to 2, with 4–768 frames, bounded by source length.
- Odd frame counts are retained; the final frame pads the temporal patch.
- Request min_pixels/max_pixels are per-frame limits. Service options
  mm_video_total_min_pixels/mm_video_total_max_pixels override total-video
  budgets without changing processor defaults.
- Explicit height/width must both be positive and align to patch size × merge size.
- Each temporal patch's timestamp is the mean of its first and last frame time.

The mixin loads the language word embedding on CPU during weight loading.
Each scheduler batch looks up all timestamp and frame delimiter tokens together.
It interleaves them with ViT features and returns one complete embedding and
relative 3D MRoPE tensor per video. The prompt's outer vision delimiters remain
text tokens. RPC carries the usual embeddings, positions and feature hashes.

Images and videos still share packed ViT forwards. Scheduling output-token
estimates include timestamp and delimiter rows; splitting raw ViT output uses
only visual-token counts.
