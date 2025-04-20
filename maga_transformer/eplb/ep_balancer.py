import datetime
import os
import json
import random
import torch
import logging

from queue import Queue
from enum import Enum
from collections import deque
from typing import Dict, List, Optional, Deque, Sequence
from dataclasses import dataclass
from safetensors import safe_open

from maga_transformer.eplb.eplb import rebalance_experts
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.model_weight import CkptWeightInfo


@dataclass
class MoeWeightInfo:
    gate: CkptWeightInfo
    up: CkptWeightInfo
    down: CkptWeightInfo

    # note: only support deepseek2 fp8
    use_fp8: bool = False
    gate_s: Optional[CkptWeightInfo] = None
    up_s: Optional[CkptWeightInfo] = None
    down_s: Optional[CkptWeightInfo] = None

class HistoryStats:
    update_time: int
    stats_holder: Deque[torch.Tensor]
    stats_tensor: torch.Tensor

    def __init__(self, window_size: int, shape: Sequence[int]):
        self.update_cnt = window_size
        self.stats_holder = deque()
        self.stats_tensor = torch.zeros(shape, dtype=torch.int64)

    def add_stats(self, stats: torch.Tensor):
        if len(self.stats_holder) == self.update_cnt:
            self.stats_tensor -= self.stats_holder.popleft()
        self.stats_holder.append(stats.clone())
        self.stats_tensor += stats
        return self.stats_tensor

class SelectLayerMethod(Enum):
    ROUND = "round"
    RANDOM = "random"
    MOST_UNBALANCED_LAYER = "most_unbalanced_layer"
    MIX = "mix"

class ExpertBalancer:
    def __init__(
        self,
        num_experts: int,
        num_replicas: int,
        num_layers: int,
        num_groups: int,
        num_node: int,
        num_gpu: int,
        moe_layer_index: List[int],
        moe_weight_info: MoeWeightInfo,
        phy2log: List[List[int]],
        model_path: Optional[str] = None,
    ):
        self.num_experts = num_experts
        self.num_replicas = num_replicas
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.num_node = num_node
        self.num_gpu = num_gpu
        self.moe_layer_index = moe_layer_index
        self.moe_weight_info = moe_weight_info
        self.time_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.last_std = torch.zeros((num_layers, num_experts), dtype=torch.float32)
        self.cur_std = torch.zeros((num_layers, num_experts), dtype=torch.float32)
        self.queue: Queue[int] = Queue()
        select_layer_method = os.environ.get("BALANCE_METHOD", "mix")
        self.select_layer_method = SelectLayerMethod(select_layer_method)
        self.phy2log = phy2log
        self.update_cnt = 0

        window_size = int(os.environ.get("EPLB_STATS_WINDOW_SIZE", 10))
        self.history_log_stats = HistoryStats(window_size=window_size, shape=(num_layers, num_experts))
        self.history_gpu_loads = HistoryStats(window_size=window_size, shape=(num_layers, num_gpu))

        logging.info(f"Balance method: {self.select_layer_method}")

        # record the expert count, used for dump stats
        self.log_exp_cnt = torch.zeros((num_layers, num_experts), dtype=torch.int32)

        if model_path:
            moe_ckpt_path = model_path
        else:
            moe_ckpt_path = fetch_remote_file_to_local(
                os.environ.get("ORIGINAL_CHECKPOINT_PATH", os.environ["CHECKPOINT_PATH"])
            )

        assert moe_ckpt_path is not None
        self.moe_ckpt_path = moe_ckpt_path

    @torch.inference_mode()
    def get_balanced_layer(self, gpu_loads: torch.Tensor) -> int:
        if self.select_layer_method == SelectLayerMethod.ROUND:
            return self.round_robin()
        elif self.select_layer_method == SelectLayerMethod.RANDOM:
            return self.random_plan()
        elif self.select_layer_method == SelectLayerMethod.MOST_UNBALANCED_LAYER:
            return self.find_most_unbalanced_layer(gpu_loads)
        elif self.select_layer_method == SelectLayerMethod.MIX:
            if self.update_cnt % 2 == 0:
                return self.round_robin()
            else:
                return self.find_most_unbalanced_layer(gpu_loads)
        else:
            raise ValueError(f"Invalid balance method: {self.select_layer_method}")

    def round_robin(self) -> int:
        # init queue
        if self.queue.empty():
            for layer_id in self.moe_layer_index:
                self.queue.put(layer_id)
        layer_id = self.queue.get()
        self.queue.put(layer_id)
        return layer_id

    def random_plan(self) -> int:
        layer_id = random.choice(self.moe_layer_index)
        return layer_id

    @torch.inference_mode()
    def find_most_unbalanced_layer(self, gpu_loads: torch.Tensor) -> int:
        std_per_layer = gpu_loads.float().std(dim=1, unbiased=False)
        self.cur_std = std_per_layer

        # note: select idx must in moe_layer_index
        most_unbalanced_idx = -1
        for idx in self.moe_layer_index:
            if most_unbalanced_idx == -1 or std_per_layer[idx] > std_per_layer[most_unbalanced_idx]:
                most_unbalanced_idx = idx

        return most_unbalanced_idx

    @torch.inference_mode()
    def create_balance_plan(
        self, log_stats: torch.Tensor, gpu_loads: torch.Tensor
    ):
        self.update_cnt += 1
        log_stats = self.history_log_stats.add_stats(log_stats)
        gpu_loads = self.history_gpu_loads.add_stats(gpu_loads)
        logging.info(log_stats)
        self.log_exp_cnt.copy_(log_stats)

        file_name = f"gpu_load_{self.time_prefix}.json"

        layer_id = self.get_balanced_layer(gpu_loads)

        phy2log, log2phy, logcnt = rebalance_experts(
            log_stats[layer_id : layer_id + 1],
            self.num_replicas,
            self.num_groups,
            self.num_node,
            self.num_gpu,
        )

        self.phy2log[layer_id] = phy2log.tolist()
        log_data = {
            "update_cnt": self.update_cnt,
            "gpu_loads": gpu_loads.tolist(),
            "log_stats": log_stats.tolist(),
            "layer": layer_id,
            "plan": phy2log.tolist(),
            "phy2log": self.phy2log,
        }
        with open(file_name, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        self.last_std = self.cur_std

        # note log2phy need to padding
        pad_k = self.num_replicas - self.num_experts + 1
        log2phy_pad = torch.empty(
            (self.num_experts, pad_k), dtype=torch.int32, device="cpu"
        )
        log2phy_pad.fill_(-1)

        k = log2phy.shape[-1]
        log2phy_pad[:, :k] = log2phy[0]

        logging.info(f"[EPLB_py PLAN] phy2log for layer {layer_id}: {phy2log[0]}")

        dtype = torch.int32
        return (
            torch.tensor([layer_id], dtype=torch.int32).contiguous(),
            logcnt[0].to(dtype).contiguous(),
            log2phy_pad.to(dtype).contiguous(),
            phy2log[0].to(dtype).contiguous(),
        )

    @torch.inference_mode()
    def load_single_moe_weight(
        self,
        weight_info: CkptWeightInfo,
        weight_map: Dict[str, str],
        layer_id: int,
        expert_id: int,
    ):
        tensor_name = weight_info.name.format(layer_id, expert_id)
        tensor_path = os.path.join(self.moe_ckpt_path, weight_map[tensor_name])
        with safe_open(tensor_path, framework="pt") as f:
            tensor = weight_info.merge_fun([f.get_tensor(tensor_name)])
        return tensor

    @torch.inference_mode()
    def load_moe_weight(
        self,
        layer_id_tensor: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        phy2log: torch.Tensor,
    ):
        layer_id = int(layer_id_tensor.item())
        expert_per_ep = self.num_replicas // ep_size
        choose_expert_id: List[int] = phy2log[
            expert_per_ep * ep_rank : expert_per_ep * (ep_rank + 1)
        ].tolist()

        logging.info(f"[EPLB_py][RANK {ep_rank}] Load MOE weight layer {layer_id} for {choose_expert_id}")

        moe_w1_tensors: List[torch.Tensor] = []
        moe_w2_tensors: List[torch.Tensor] = []
        moe_s1_tensors: List[torch.Tensor] = []
        moe_s2_tensors: List[torch.Tensor] = []

        use_fp8 = self.moe_weight_info.use_fp8

        index_path = os.path.join(self.moe_ckpt_path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            weight_map: Dict[str, str] = json.loads(f.read())["weight_map"]

        for expert_id in choose_expert_id:
            moe_gate = self.load_single_moe_weight(
                self.moe_weight_info.gate, weight_map, layer_id, expert_id
            )
            moe_up = self.load_single_moe_weight(
                self.moe_weight_info.up, weight_map, layer_id, expert_id
            )
            moe_down = self.load_single_moe_weight(
                self.moe_weight_info.down, weight_map, layer_id, expert_id
            )

            if use_fp8:
                assert self.moe_weight_info.gate_s is not None
                assert self.moe_weight_info.up_s is not None
                assert self.moe_weight_info.down_s is not None

                moe_gate_s = self.load_single_moe_weight(
                    self.moe_weight_info.gate_s, weight_map, layer_id, expert_id
                )
                moe_up_s = self.load_single_moe_weight(
                    self.moe_weight_info.up_s, weight_map, layer_id, expert_id
                )
                moe_down_s = self.load_single_moe_weight(
                    self.moe_weight_info.down_s, weight_map, layer_id, expert_id
                )
                moe_s1_tensors.append(torch.concat([moe_up_s, moe_gate_s], dim=0))
                moe_s2_tensors.append(moe_down_s)

            moe_w1_tensors.append(torch.concat([moe_up, moe_gate], dim=0))
            moe_w2_tensors.append(moe_down)

        moe_w1 = torch.stack(moe_w1_tensors, dim=0).contiguous()
        moe_w2 = torch.stack(moe_w2_tensors, dim=0).contiguous()

        if use_fp8:
            moe_w1_s = torch.stack(moe_s1_tensors, dim=0).contiguous()
            moe_w2_s = torch.stack(moe_s2_tensors, dim=0).contiguous()
        else:
            moe_w1_s = torch.empty(0, dtype=torch.float32)
            moe_w2_s = torch.empty(0, dtype=torch.float32)

        logging.info(f"[EPLB_py][RANK {ep_rank}] Load MOE weight layer {layer_id} done")

        return layer_id, moe_w1, moe_w2, moe_w1_s, moe_w2_s

    @torch.inference_mode()
    def dump_stats(self):
        # TODO(yinzhi): dump some stats for expert balancer
        pass

    @torch.inference_mode()
    def init_random_plan(self):
        log_exp_cnt = torch.zeros_like(self.log_exp_cnt)
        pad_k = self.num_replicas - self.num_experts + 1
        log2phy_pad = torch.empty(
            (self.num_experts, pad_k), dtype=torch.int32, device="cpu"
        )
        log2phy_pad.fill_(-1)

        phy2log, log2phy, logcnt = rebalance_experts(
            log_exp_cnt,
            self.num_replicas,
            self.num_groups,
            self.num_node,
            self.num_gpu,
        )

        k = log2phy.shape[-1]
        log2phy_pad[:, :k] = log2phy

        return phy2log.tolist(), log2phy.tolist(), logcnt.tolist()