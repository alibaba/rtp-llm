# Qwen3.5 video preprocessing

Qwen3.5 inherits Qwen3-VL's image preprocessing and always uses NVDEC for videos.
Preprocessing workers fetch bytes and plan frame/grid metadata on CPU; the
embedding worker decodes frames with NVDEC and resizes/normalizes them on GPU.
Video decoding requires PyNvVideoCodec and the NVIDIA video driver libraries.

Configure each video through its request content item's preprocess_config:

```json
{
  "type": "video_url",
  "video_url": {"url": "https://example.com/video.mp4"},
  "preprocess_config": {
    "fps": 6,
    "max_frames": 180,
    "min_pixels": 2500000,
    "max_pixels": 73728000
  }
}
```

- The loader uses fps and frame limits (defaults: 2 fps, 4–768 frames). It samples
  uniformly across the source video, including endpoints, like vLLM's qwen3_vl
  backend.
- Request min_pixels/max_pixels bound total video pixels (T × H × W), overriding
  the model's video_processor.size.shortest_edge/longest_edge respectively.
  Unspecified pixel limits inherit the model's video processor configuration.
- Explicit resized_height/resized_width are still supported, subject to that
  total video budget.
- The resize formula follows Qwen3-VL in Transformers 5.15, with spatial
  alignment to patch_size × merge_size. Odd sampled frame counts are retained;
  the processor repeats the final frame to align temporal patches.
- Frames are already sampled and resized when passed to the processor, so it
  runs with do_resize=False and do_sample_frames=False.

For a 1920×1080 video sampled to 180 frames, the above pixel budget produces
832×480 frames. No video backend selection is required.

The current NVDEC path accepts indexed videos with a known frame count and
average frame rate, using limited-range 8-bit YUV420 and BT.601/709 color.
