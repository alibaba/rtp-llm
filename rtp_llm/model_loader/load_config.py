import json
import logging
from typing import Any, List, Optional, Union

import torch
from pydantic import BaseModel, field_validator, model_validator

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.device.device_base import DeviceBase
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.fuser import fetch_remote_file_to_local
from rtp_llm.utils.util import check_with_info


class LoadConfig(BaseModel):
    database: Any
    num_layers: int
    hidden_size: int
    head_num: int
    head_num_kv: int
    size_per_head: int
    inter_size: int
    inter_padding_size: int
    use_stack_weight: bool
    moe_inter_padding_size: int
    moe_layer_index: List[int]
    moe_n_group: int
    expert_num: int
    enable_eplb: bool
    phy_exp_num: int

    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    dp_size: int
    dp_rank: int
    ffn_tp_size: int
    ffn_tp_rank: int
    num_nodes: int
    tp_split_emb_and_lm_head: bool = False
    bit: int = 16
    merge_lora: bool = False

    vit_separation: int = 0
    compute_dtype: Any = torch.float16

    quant_algo: Any = None

    is_ft_style_weight: bool = False

    exported_device: Optional[Any] = None

    phy2log: Optional[List[List[int]]] = None
    use_swizzleA: bool = False

    @field_validator("database", "compute_dtype", "quant_algo", "exported_device")
    @classmethod
    def validate_custom_types(cls, value: Any, info) -> Any:
        field_name = info.field_name
        expected_types = {
            "database": BaseDatabase,
            "compute_dtype": torch.dtype,
            "quant_algo": (type(None), object),
            "exported_device": (type(None), DeviceBase),
        }
        expected = expected_types[field_name]

        if not isinstance(value, expected):
            raise TypeError(
                f"Field '{field_name}' expects type {expected}, got {type(value)}"
            )
        return value

    @model_validator(mode="after")
    def _set_default_phy2log(self) -> "LoadConfig":
        if self.phy2log is None and self.expert_num > 0:
            phy2log = self.create_redundant_expert(
                layer_num=self.num_layers,
                expert_num=self.expert_num,
                phy_exp_num=self.phy_exp_num,
                ep_size=self.ep_size,
                num_nodes=self.num_nodes,
                PHY2LOG_PATH_KEY="PHY2LOG_PATH",
            )
            return self.model_copy(update={"phy2log": phy2log})
        return self

    def get_selected_experts(self, layer_id: int, expert_num):
        selected_experts = range(expert_num)
        if self.phy2log:
            selected_experts = self.phy2log[layer_id]
        expert_per_ep = len(selected_experts) // self.ep_size
        ep_rank = self.ep_rank
        selected_experts = selected_experts[
            expert_per_ep * ep_rank : expert_per_ep * (ep_rank + 1)
        ]
        return selected_experts

    def udpate_layer_experts(
        self,
        layer_id_tensor: Union[int, torch.Tensor],
        layer_phy2log: Union[List[int], torch.Tensor],
    ):
        layer_id: int = (
            int(layer_id_tensor.item())
            if isinstance(layer_id_tensor, torch.Tensor)
            else layer_id_tensor
        )
        experts: List[int] = (
            layer_phy2log.tolist()
            if isinstance(layer_phy2log, torch.Tensor)
            else layer_phy2log
        )
        check_with_info(
            layer_id < self.num_layers
            and self.phy2log
            and len(self.phy2log[layer_id]) == self.phy_exp_num,
            f"layer_id:{layer_id} muse less than num_layers:{self.num_layers} and phy2log len(self.phy2log[layer_id]) must equal to {self.phy_exp_num}",
        )
        self.phy2log[layer_id] = experts
        logging.debug("update layer %s phy2log %s", layer_id, layer_phy2log)

    @staticmethod
    def create_redundant_expert(
        layer_num: int,
        expert_num: int,
        phy_exp_num: int,
        ep_size: int,
        num_nodes: int,
    ):
        if expert_num == 0:
            return None

        expert_num = expert_num
        redundant_expert = phy_exp_num - expert_num
        expert_num_per_ep = expert_num // ep_size
        rank_per_node = ep_size // num_nodes

        check_with_info(
            redundant_expert <= expert_num,
            f"redundant_expert:{redundant_expert} must less or equal than expert_num:{expert_num}",
        )
        check_with_info(
            phy_exp_num % ep_size == 0,
            f"phy_exp_num:{phy_exp_num} must be divisible by ep_size:{ep_size}",
        )

        layer_num = layer_num

        phy2log: List[List[int]] = []
        phy2log_path = fetch_remote_file_to_local(StaticConfig.load_config.phy2log_path)

        if phy2log_path:
            with open(phy2log_path, "r") as f:
                phy2log = json.load(f)
                check_with_info(
                    len(phy2log) == layer_num,
                    f"phy2log len {len(phy2log)} != layer_num {layer_num}",
                )
                check_with_info(
                    len(phy2log[0]) == phy_exp_num,
                    f"phy2log[0] len {len(phy2log[0])} != phy_exp_num {phy_exp_num}",
                )
        elif redundant_expert % ep_size == 0:
            redundant_expert_per_ep = redundant_expert // ep_size
            for _ in range(layer_num):
                layer_phy2log: List[int] = []
                for ep_rank in range(ep_size):
                    node_id = ep_rank // rank_per_node
                    layer_phy2log.extend(
                        range(
                            ep_rank * expert_num_per_ep,
                            (ep_rank + 1) * expert_num_per_ep,
                        )
                    )
                    for i in range(redundant_expert_per_ep):
                        redundant_ep_id = (
                            ep_rank + 1
                        ) % rank_per_node + node_id * rank_per_node
                        expert_id = redundant_ep_id * expert_num_per_ep + i
                        layer_phy2log.append(expert_id)
                phy2log.append(layer_phy2log)
        else:
            for _ in range(layer_num):
                layer_phy2log: List[int] = list(range(expert_num))
                layer_phy2log.extend(list(range(redundant_expert)))
                phy2log.append(layer_phy2log)

        logging.debug("phy2log: %s", phy2log)
        return phy2log
