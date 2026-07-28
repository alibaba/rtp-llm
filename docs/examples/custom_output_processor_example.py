"""CUSTOM_OUTPUT_PROCESSOR 业务接入示例。

场景：LANGUAGE_MODEL 部署，prefill 阶段取每条请求 prompt 最后一个 token 的
last-layer hidden states，过自定义 MLP 打分，打分随生成结果一起返回，
不影响生成过程。

部署方式（部署级门控，设了即整个部署开启）:
    TASK_TYPE=LANGUAGE_MODEL
    CUSTOM_OUTPUT_PROCESSOR=/path/to/this_file.py   # 或 python 模块 dotted path
    CUSTOM_PROCESSOR_MODE=eager                     # v1 仅支持 eager，可省略

打分在响应里的位置:
    OpenAI /v1/chat/completions: choices[i].extra_outputs.custom_output
    原生 /:                      custom_output

约定与约束:
  * MLP 权重放在模型 ckpt 里，通过 custom_weight_info() 走现有权重加载链，
    不要在业务代码里自行 torch.load。
  * extend_forward 运行在引擎 forward 热路径上（每个 prefill step 一次，
    batched），禁止任何同步操作: .item() / .cpu() / .tolist() /
    torch.cuda.synchronize() / print / logging。输出留在 GPU 上，
    引擎负责异步搬运。
  * 输出第 0 维必须等于输入的第 0 维（每条 context 请求一行）。
  * 启动时引擎会用 dummy 输入预跑本 handler（吃掉懒初始化），跑不过则启动失败。
"""

from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import (
    CustomHandler,
    CustomModule,
    Trigger,
)
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.util import to_torch_dtype


class ScoreHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        hidden_size = self.config_.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
        )

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [
            "score_head.dense.weight",
            "score_head.dense.bias",
            "score_head.out.weight",
            "score_head.out.bias",
        ]
        return [
            CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            for k in w_list
        ]

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        self.mlp[0].weight.data = tensor_map["score_head.dense.weight"]
        self.mlp[0].bias.data = tensor_map["score_head.dense.bias"]
        self.mlp[2].weight.data = tensor_map["score_head.out.weight"]
        self.mlp[2].bias.data = tensor_map["score_head.out.bias"]
        data_type = to_torch_dtype(self.config_.data_type)
        self.mlp = self.mlp.to(data_type).eval().to(self.device)

    def extend_forward_args(self) -> List[str]:
        # v1 生成路径仅支持 last_hidden_states；声明其他参数会启动失败
        return ["last_hidden_states"]

    def trigger_mode(self) -> Trigger:
        return Trigger.CONTEXT  # 基类默认值，显式写出便于阅读

    def extend_forward(self, **kwargs: Any) -> torch.Tensor:
        # last_hidden_states: [context_batch, hidden]，每条请求 prompt 最后
        # 一个 token 的 hidden（与引擎喂 lm_head 的行完全一致，零额外采集）
        last_hidden = kwargs["last_hidden_states"]
        with torch.no_grad():
            return self.mlp(last_hidden)  # [context_batch, 1]


class ScoreModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: Any):
        super().__init__(config, tokenizer)
        self.handler = ScoreHandler(config)


def create_custom_module(config: ModelConfig, tokenizer: Any) -> CustomModule:
    return ScoreModule(config, tokenizer)
