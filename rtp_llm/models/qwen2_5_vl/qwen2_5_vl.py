from typing import Any
import os

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from rtp_llm.config.model_config import VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL, QwenVL2VitWeight, QWen2VLWeightInfo

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

import torch.library as tl
from rtp_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
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
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(
            mm_related_params.config
        )

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
                    assert up_proj_name in weight_names, f"up_proj {up_proj_name} not found for gate_proj {w}"

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
                elif '.up_gate_proj.' in w:
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
        self.mm_part = Qwen2_5_VLImageEmbedding(mm_related_params, model_config=self.model_config)
        self.model_config.mm_related_params.vit_weights = QwenVL2VitWeight(
            {"vit": self.mm_part.visual}
        )

    @staticmethod
    def get_weight_cls():
        return QWen2_5_VLWeightInfo

    def _load_mm_weight(self, vit_params: VitParameters, ctype, device: str):
        """
        重写权重加载方法，在权重加载时对 attention 层的权重进行 swizzle 变换
        """
        import re
        from rtp_llm.utils.util import to_torch_dtype
        
        vit_weight = vit_params.vit_weights
        ft_prefix = vit_weight.ft_prefix
        weight_names = vit_weight.weight_names
        
        # 检查是否启用 swizzle（AMD GPU 优化）
        use_swizzle = os.environ.get("USE_SWIZZLEA", None) == "1"
        
        def _can_shuffle(n: int, k: int, layout: tuple[int, int]) -> bool:
            """检查张量维度是否可以进行 swizzle 操作"""
            IN, IK = layout
            BK = IK * 2
            return (n % IN == 0) and (k % BK == 0)
        
        def _get_next_multiple(x: int, base: int = 32) -> int:
            """计算大于等于 x 的下一个 base 的倍数"""
            return ((x + base - 1) // base) * base
        
        def _swizzle_weight(t: torch.Tensor) -> torch.Tensor:
            """对权重张量进行 swizzle 变换"""
            from rtp_llm.utils.swizzle_utils import swizzle_tensor
            from rtp_llm.utils.model_weight import pad
            
            if t.dim() != 2:
                raise ValueError(f"Expected 2D tensor for swizzle, got shape {t.shape}")
            
            # 检查维度是否可以直接 swizzle，t shape (N,K)
            if _can_shuffle(t.shape[0], t.shape[1], (16, 16)):
                t_swizzled = swizzle_tensor(t, False, MiM=16).t()  # t_swizzled shape (K,N)
                return t_swizzled
            else:
                # 无法直接 shuffle，需要对 K 维度进行 padding 到下一个 32 的倍数
                target_k = _get_next_multiple(t.shape[1], base=32)
                t_padded = pad([t], inter_padding_size=target_k, dim=1)
                t_swizzled = swizzle_tensor(t_padded, False, MiM=16).t()  # t_swizzled shape (K,N)
                return t_swizzled
        
        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype):
            t = self.weight.get_global_weight_or_none(fname)
            if t is None:
                raise Exception(f"failed to get tensor from name {fname}")
            
            # Convert ctype (which may be DataType enum or string) to torch.dtype
            torch_dtype = to_torch_dtype(ctype)
            
            # 如果启用了 swizzle 并且是 attention 层的权重，进行 swizzle 变换
            if use_swizzle and "attn" in fname and ("qkv" in fname or "proj" in fname) and "weight" in fname and "bias" not in fname:
                t = _swizzle_weight(t)
                param.data = t.to(torch_dtype).to(device)
            else:
                # 普通权重按原来的方式加载
                param.data = t.reshape(param.data.shape).to(torch_dtype).to(device)
        
        for w in weight_names:
            w_name = ft_prefix + w
            w_name = re.sub(r"\.\d+\.", lambda x: "[" + x.group(0)[1:-1] + "].", w_name)
            param = eval(w_name)
            _safe_load_from_module(param, w, ctype)
        
        # 权重加载完成后，将 attention 层的 nn.Linear 替换为 RocmF16Linear
        if use_swizzle and hasattr(self.mm_part, 'visual') and hasattr(self.mm_part.visual, 'blocks'):
            for block in self.mm_part.visual.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, '_replace_with_rocm_linear'):
                    block.attn._replace_with_rocm_linear(hw_kernel_config=self.hw_kernel_config)


register_model("qwen2_5_vl", QWen2_5_VL, ["Qwen2_5_VLForConditionalGeneration"])
