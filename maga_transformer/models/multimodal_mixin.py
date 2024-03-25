import json
import torch
import re
from typing import Any, Dict, List, Union, Tuple, Optional
from PIL import Image

from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import RequestFormat
from maga_transformer.utils.model_weight import ModelDeployWeightInfo, CkptWeightInfo, WeightInfo, sp_id, identity
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.distribute.worker_info import g_parallel_info

class BaseImageEmbedding:
    def image_embedding(self, images, device):
        raise NotImplementedError()

class BaseVitWeights:
    def __init__(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        self.weight_names: List[str] = []
        self._get_vit_params(vit_part, with_prefix)
    
    @property
    def ckpt_prefix(self) -> str:
        return "model."
    
    @property
    def ft_prefix(self) -> str:
        return "self.visual."
    
    def _get_vit_params(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        if len(vit_part) >= 2 or with_prefix:
            for vit_name, vit in vit_part.items():
                self.weight_names.extend([vit_name + '.' + w for w in vit.state_dict().keys()])
        else:
            for _, vit in vit_part.items():
                self.weight_names.extend(list(vit.state_dict().keys()))

class BaseMultiModalWeightInfo:
    def __init__(self, config: GptInitModelParameters):
        self.vit_weights: Optional[BaseVitWeights] = config.vit_related_params.vit_weights

    def _get_vit_info(self, llm_weights: ModelDeployWeightInfo):
        if self.vit_weights is not None:
            weight_names = self.vit_weights.weight_names
            ckpt_prefix = self.vit_weights.ckpt_prefix

            for w in weight_names:
                w_name = ckpt_prefix + w
                llm_weights.weights.append(WeightInfo(w_name, [CkptWeightInfo(w_name, identity)], identity))
                llm_weights.tp_strategy[w_name] = sp_id

class MultiModalMixin:
    visual: BaseImageEmbedding
    image_expand_token: int
    nccl_op_: NcclOp

    @staticmethod
    def process_encode_plugin(prompt: str, generate_config: Dict[str, Any], tokenizer: Any, **kwargs: Any) -> List[int]:
        if len(prompt) == 0:
            raise FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "prompt should have at least one token!")
        if type(prompt) is not str:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "expect string prompt, actual: " + str(prompt))
        return tokenizer.encode(prompt)

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str], 
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[str]]:
        # should delete after chatapi interface update
        if kwargs.get('generate_config', {})['request_format'] == RequestFormat.CHAT_API:
            if isinstance(prompt, str):
                messages = json.loads(prompt, strict=False)
            else:
                messages = prompt
            new_prompt: str = ""
            new_images: List[str] = []
            for message in messages:
                new_prompt += message['role'].upper() + ' :'
                if isinstance(message['content'], str):
                    new_prompt += message['content'] + '\n'
                elif isinstance(message['content'], List):
                    for x in message['content']:
                        if x['type'] == 'text':
                            new_prompt += x['text']
                        elif x['type'] == 'image_url':
                            now_images = x['image_url']
                            if isinstance(now_images, List):
                                new_images.extend(now_images)
                                new_prompt += (img_token + '\n') * len(now_images)
                            else:
                                new_images.append(now_images)
                                new_prompt += img_token + '\n'
                        else:
                            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "content type can only be text or image_url, but get: " + x['type'])
                    new_prompt += '\n'
            return new_prompt + 'ASSISTANT :', new_images
        elif isinstance(prompt, List):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "raw request format cannot accept dict prompt")
        return prompt, images
    
    def expand_token_id(self, token_ids: List[int], images: List[Image.Image]) -> Tuple[List[int], Union[torch.Tensor, List[torch.Tensor]]]:
        raise NotImplementedError()

    def load_vit_weight(self, ctype: str):
        vit_weight = self.config.vit_related_params.vit_weights
        ckpt_prefix = vit_weight.ckpt_prefix
        ft_prefix = vit_weight.ft_prefix
        weight_names = vit_weight.weight_names

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype: torch.dtype):
            param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to(ctype).to('cuda:0')

        for w in weight_names:
            w_name = ft_prefix + w
            w_name = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', w_name)
            param = eval(w_name)
            _safe_load_from_module(param, ckpt_prefix + w, ctype)

    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[Union[torch.Tensor, List[torch.Tensor]]]):
        inputs = inputs.reshape(1, -1)
        if g_parallel_info.tp_size <= 1:
            return self.multimodal_embedding(inputs, images).squeeze(0)

        if g_parallel_info.tp_rank == 0:
            embedding_tensor = self.multimodal_embedding(inputs, images).squeeze(0)
        else:
            embedding_tensor = torch.zeros((inputs.shape[1], self.config.head_num * self.config.size_per_head), dtype=torch.float16, device=self.device)
        self.nccl_op_.broadcast_tp([embedding_tensor])
        torch.cuda.current_stream().synchronize()
        return embedding_tensor
        
    def input_word_embedding(self, inputs: torch.Tensor, images: List[Union[torch.Tensor, List[torch.Tensor]]]):
        if g_parallel_info.tp_size <= 1:
            return self.multimodal_embedding(inputs, images)

        if g_parallel_info.tp_rank == 0:
            embedding_tensor = self.multimodal_embedding(inputs, images)
        else:
            embedding_tensor = torch.zeros((inputs.shape[1], self.config.head_num * self.config.size_per_head), dtype=torch.float16, device=self.device)
        self.nccl_op_.broadcast_tp([embedding_tensor])
        torch.cuda.current_stream().synchronize()
        return embedding_tensor