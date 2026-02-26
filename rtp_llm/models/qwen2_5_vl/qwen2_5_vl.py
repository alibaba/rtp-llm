from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from rtp_llm.config.model_config import VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen2_vl.qwen2_vl import (
    QWen2_VL,
    QWen2VLWeightInfo,
    QwenVL2VitWeight,
)

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

import torch.library as tl

from rtp_llm.models.qwen2_vl.qwen2_vl_vit import (
    FPS,
    FPS_MAX_FRAMES,
    FPS_MIN_FRAMES,
    FRAME_FACTOR,
    IMAGE_FACTOR,
    VIDEO_MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    VIDEO_TOTAL_PIXELS,
    Qwen2VLImageEmbedding,
    Qwen2VLImageProcessor,
    ceil_by_factor,
    floor_by_factor,
    smart_resize,
)

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton


def smart_nframes(configs, total_frames, video_fps) -> int:
    fps = configs.fps if configs.fps != -1 else FPS
    min_frames = ceil_by_factor(
        configs.min_frames if configs.min_frames != -1 else FPS_MIN_FRAMES, FRAME_FACTOR
    )
    max_frames = floor_by_factor(
        (
            configs.max_frames
            if configs.max_frames != -1
            else min(FPS_MAX_FRAMES, total_frames)
        ),
        FRAME_FACTOR,
    )
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


class Qwen2_5_VLImageEmbedding(Qwen2VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters, model_config=None):
        super().__init__(mm_related_params, model_config=model_config)
        self.mm_related_params = mm_related_params
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        from rtp_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        self.visual = Qwen2_5_VisionTransformerPretrainedModel(mm_related_params.config)

    def load_video(self, data, configs, **kwargs):
        vr = VideoReader(data, ctx=cpu(0), num_threads=1)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        nframes = smart_nframes(configs, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        height, width = vr[0].shape[:2]

        video = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        del vr

        image_factor = IMAGE_FACTOR

        nframes, _, height, width = video.shape
        min_pixels = (
            configs.min_pixels if configs.min_pixels != -1 else VIDEO_MIN_PIXELS
        )
        total_pixels = VIDEO_TOTAL_PIXELS
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = (
            configs.max_pixels if configs.max_pixels != -1 else max_pixels
        )
        max_pixels = min(max_pixels_supposed, max_pixels)
        if configs.height != -1 and configs.width != -1:
            resized_height, resized_width = smart_resize(
                configs.height,
                configs.width,
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video


class QWen2_5_VLWeightInfo(QWen2VLWeightInfo):
    def _get_vit_info(self, llm_weights: "ModelWeightInfo") -> "ModelWeightInfo":
        from rtp_llm.model_loader.weight_module import MMAtomicWeight
        from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id

        if self.vit_weights is not None:
            weight_names = self.vit_weights.weight_names
            ckpt_prefix = self.vit_weights.ckpt_prefix

            for w in weight_names:
                if ".gate_proj." in w:
                    up_proj_name = w.replace(".gate_proj.", ".up_proj.")
                    assert (
                        up_proj_name in weight_names
                    ), f"up_proj {up_proj_name} not found for gate_proj {w}"

                    up_gate_proj_name = w.replace(".gate_proj.", ".up_gate_proj.")
                    gate_proj_ckpt_name = ckpt_prefix + w
                    up_proj_ckpt_name = ckpt_prefix + up_proj_name

                    llm_weights.weights.append(
                        MMAtomicWeight(
                            up_gate_proj_name,
                            [
                                CkptWeightInfo(gate_proj_ckpt_name, identity),
                                CkptWeightInfo(up_proj_ckpt_name, identity),
                            ],
                            lambda ts: torch.cat(ts, dim=0).contiguous(),
                            split_func=sp_id,
                        )
                    )
                elif ".up_gate_proj." in w:
                    continue
                w_name = ckpt_prefix + w
                llm_weights.weights.append(
                    MMAtomicWeight(
                        w,
                        [CkptWeightInfo(w_name, identity)],
                        identity,
                        split_func=sp_id,
                    )
                )
        return llm_weights


class QWen2_5_VL(QWen2_VL):
    def _init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = self.model_config.mm_related_params
        self.mm_part = Qwen2_5_VLImageEmbedding(
            mm_related_params, model_config=self.model_config
        )
        self.model_config.mm_related_params.vit_weights = QwenVL2VitWeight(
            {"vit": self.mm_part.visual}
        )

    @staticmethod
    def get_weight_cls():
        return QWen2_5_VLWeightInfo


register_model("qwen2_5_vl", QWen2_5_VL, ["Qwen2_5_VLForConditionalGeneration"])
