import copy
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.downstream_modules.embedding.api_datatype import (
    ContentPart,
    ContentPartTypeEnum,
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    SimilarityRequest,
)
from rtp_llm.downstream_modules.embedding.misc import EmbeddingRendererBase
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.utils.multimodal_util import (
    MMUrlType,
    MultimodalInput,
    get_bytes_io_from_url,
)
from rtp_llm.utils.time_util import current_time_ms


def slice_image(
    image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(original_size, scale_resolution, patch_size)

        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)

        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
    original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


class MiniCPMVInputGenerator(object):

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: BaseTokenizer,
        vit_config: Optional[VitConfig] = None,
    ):
        self.tokenizer_ = tokenizer
        self.config_ = config
        self.vit_config = config.mm_related_params.config
        self.vit_config_obj = vit_config  # Store VitConfig object for download_headers and url_cache_item_num
        self.im_start = self.tokenizer_.im_start
        self.im_end = self.tokenizer_.im_end
        self.slice_start = self.tokenizer_.slice_start
        self.slice_end = self.tokenizer_.slice_end
        self.unk_token = self.tokenizer_.unk_token

        self.query_num = self.vit_config["query_num"]
        self.max_slice_nums = self.vit_config["max_slice_nums"]
        self.scale_resolution = self.vit_config["scale_resolution"]
        self.patch_size = self.vit_config["patch_size"]
        self.slice_mode = self.vit_config["slice_mode"]

    def get_grid_placeholder(self, grid, query_num):
        image_placeholder = self.im_start + self.unk_token * query_num + self.im_end

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(image_placeholder)
            slices.append("".join(lines))
        slice_placeholder = self.slice_start + "\n".join(slices) + self.slice_end
        return slice_placeholder

    # def slice_image(self, image):
    #     return slice_image(
    #         image,
    #         self.max_slice_nums,
    #         self.scale_resolution,
    #         self.patch_size,
    #     )

    def get_slice_image_placeholder(self, image):
        image_placeholder = (
            self.im_start + self.unk_token * self.query_num + self.im_end
        )

        slice_images = []

        source_image, patches, best_grid = slice_image(
            image,
            self.max_slice_nums,
            self.scale_resolution,
            self.patch_size,
        )

        slice_images.append(source_image)
        final_placeholder = image_placeholder

        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])

            final_placeholder += self.get_grid_placeholder(best_grid, self.query_num)

        return slice_images, final_placeholder

    def _render_image(self, url: str):
        content = ""
        download_headers = self.vit_config_obj.download_headers
        image = get_bytes_io_from_url(url, download_headers=download_headers)
        image = Image.open(image).convert("RGB")
        if self.slice_mode:
            _, final_placeholder = self.get_slice_image_placeholder(
                image
            )  # crop one image into multiple sub images -> List[Image]
            content = final_placeholder + "\n" + content
        else:
            content = (
                self.im_start
                + self.unk_token * self.query_num
                + self.im_end
                + "\n"
                + content
            )
        return content

    @torch.inference_mode()
    def generate(  # type: ignore
        self,
        inputs: Union[ContentPart, List[ContentPart]],
        truncate: bool = True,
        tokenizer_config: Dict[str, Any] = {},
    ) -> EngineInputs:
        if isinstance(inputs, ContentPart):
            inputs = [inputs]
        assert isinstance(inputs, list) and all(
            [isinstance(i, ContentPart) for i in inputs]
        )
        msgs: List[str] = []
        urls: List[str] = []
        types: List[MMUrlType] = []
        for content in inputs:
            if content.type == ContentPartTypeEnum.text:
                msgs.append(content.text)
            elif content.type == ContentPartTypeEnum.image_url:
                assert content.image_url != None
                msgs.append(self._render_image(content.image_url.url))
                urls.append(content.image_url.url)
                types.append(MMUrlType.IMAGE)
        begin_time = current_time_ms()
        # align images and prompts
        # do batch encode and split into embedding input per batch
        assert self.tokenizer_ is not None, "tokenizer should not be None"
        # truncate with tokenizer max_seq_len
        truncate_length = self.config_.max_seq_len
        if self.config_.position_ids_style == 1:
            truncate_length = self.config_.max_seq_len - (
                self.config_.special_tokens.pad_token_id + 1
            )
        encoded = self.tokenizer_(
            msgs,
            max_length=truncate_length,
            return_attention_mask=False,
            padding=False,
            return_length=True,
            truncation=truncate,
            return_tensors="np",
            **tokenizer_config,
        )
        combo_tokens = torch.from_numpy(np.concatenate(encoded["input_ids"])).to(
            torch.int32
        )
        if "token_type_ids" in encoded:
            combo_token_types = torch.from_numpy(
                np.concatenate(encoded["token_type_ids"])
            ).to(torch.int32)
        else:
            combo_token_types = torch.zeros_like(combo_tokens, dtype=torch.int32)
        input_lengths = torch.from_numpy(encoded["length"]).to(torch.int32)

        for length in input_lengths:
            if length > self.config_.max_seq_len:
                raise FtRuntimeException(
                    ExceptionType.LONG_PROMPT_ERROR,
                    f"one of prompt length: {length} > max_length: {self.config_.max_seq_len}",
                )

        kmonitor.report(
            GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
        )
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(combo_tokens))
        return EngineInputs(
            token_ids=combo_tokens,
            token_type_ids=combo_token_types,
            input_lengths=input_lengths,
            multimodal_inputs=[
                MultimodalInput(url=url, mm_type=mm_type)
                for url, mm_type in zip(urls, types)
            ],
        )


class MiniCPMVHandler(CustomHandler):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        input_lens = input_lengths.tolist()
        token_ids = 0
        reps = []
        print(f"input_lengths: {input_lengths}")
        print(f"token_ids: {token_ids}")

        for length in input_lens:
            hidden_state = hidden_states[token_ids : token_ids + length]
            attention_mask = torch.range(1, length).float().cuda()
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=0)
            d = attention_mask.sum(dim=0, keepdim=True)
            reps.append(s / d)
            token_ids += length
        reps_normalized = F.normalize(torch.stack(reps), dim=1)
        return reps_normalized


class MiniCPMVRenderer(EmbeddingRendererBase):

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: BaseTokenizer,
        vit_config: Optional[VitConfig] = None,
    ):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.DENSE
        self.generator = MiniCPMVInputGenerator(
            config, tokenizer, vit_config=vit_config
        )

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        return float(torch.tensor(left.embedding) @ torch.tensor(right.embedding).T)

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res.tolist()

    def create_input(self, request: Union[OpenAIEmbeddingRequest, SimilarityRequest]):
        if isinstance(request, OpenAIEmbeddingRequest):
            engine_inputs = self.generator.generate(
                request.input, tokenizer_config=request.extra_configs.tokenizer_config
            )
        else:
            engine_inputs = self.generator.generate(request.left + request.right)
        return engine_inputs

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if "data" in log_response:
            del log_response["data"]
        return log_response


class MiniCPMVModule(CustomModule):

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: BaseTokenizer,
        vit_config: Optional[VitConfig] = None,
    ):
        super().__init__(config, tokenizer)
        self.renderer = MiniCPMVRenderer(config, tokenizer, vit_config=vit_config)
        self.handler = MiniCPMVHandler(config)
