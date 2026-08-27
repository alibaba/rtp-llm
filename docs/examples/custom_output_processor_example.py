"""CUSTOM_OUTPUT_PROCESSOR 业务接入示例。

场景：LANGUAGE_MODEL 部署，prefill 阶段取每条请求 prompt 倒数第二个 token 的
last-layer hidden states，过自定义 MLP 打分，打分随生成结果一起返回，
不影响生成过程。

部署方式（部署级门控，设了即整个部署开启）:
    TASK_TYPE=LANGUAGE_MODEL
    # 相对 .py 路径从 CHECKPOINT_PATH 解析，可随 ckpt 一起发布
    CUSTOM_OUTPUT_PROCESSOR=custom_output_processor.py
    # 也支持绝对 .py 路径或 Python 模块 dotted path
    # 选择模式一：固定位置。支持正数绝对位置和负数尾部相对位置。
    CUSTOM_OUTPUT_TOKEN_POSITION=-2
    # 可选：O(1) 校验该位置确实是预期特殊 token。
    CUSTOM_OUTPUT_EXPECTED_TOKEN_ID=151644

    # 选择模式二：按 token ID 反向寻找最后一次出现（与位置模式互斥）。
    # CUSTOM_OUTPUT_TRACKED_TOKEN_ID=151644

打分在响应里的位置:
    RTP-LLM OpenAI /v1/chat/completions: response.extra_outputs.custom_output
    原生 / Raw:                        response.custom_output

    extra_outputs 是 RTP-LLM 的非标准 OpenAI 扩展字段；严格过滤 OpenAI
    标准字段的接入层可能丢弃它，此时应使用直连或 Raw 接口。

约定与约束:
  * handler 声明 selected_hidden_states 时，必须且只能配置
    CUSTOM_OUTPUT_TOKEN_POSITION / CUSTOM_OUTPUT_TRACKED_TOKEN_ID 之一；
    handler 声明 last_hidden_states 时不得配置 selector，收到的是每条
    context 请求最后一个 token 的 hidden state。
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
        return ["selected_hidden_states"]

    def trigger_mode(self) -> Trigger:
        return Trigger.CONTEXT  # 基类默认值，显式写出便于阅读

    def extend_forward(self, **kwargs: Any) -> torch.Tensor:
        # selected_hidden_states: [context_batch, hidden]，每条请求倒数
        # 第二个 token 的 hidden；如果配了 expected token ID，请求入口已校验。
        selected_hidden = kwargs["selected_hidden_states"]
        with torch.no_grad():
            return self.mlp(selected_hidden)  # [context_batch, 1]


class ScoreModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: Any):
        super().__init__(config, tokenizer)
        self.handler = ScoreHandler(config)


def create_custom_module(config: ModelConfig, tokenizer: Any) -> CustomModule:
    return ScoreModule(config, tokenizer)
