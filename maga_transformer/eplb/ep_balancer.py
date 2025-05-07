import datetime
import os
import json
import random
import torch
import logging

from queue import Queue
from enum import Enum
from collections import deque
from typing import Deque, Sequence

from maga_transformer.utils.database import BaseDatabase
from maga_transformer.utils.model_weight import W
from maga_transformer.eplb.eplb import rebalance_experts
from maga_transformer.device import get_current_device
from maga_transformer.model_loader.load_config import LoadConfig
from maga_transformer.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo


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
        weights_info: ModelDeployWeightInfo, 
        compute_dtype: torch.dtype, 
        database: BaseDatabase, 
    ):
        self.database: BaseDatabase = database
        self._weights_info: ModelDeployWeightInfo = weights_info
        self._model_weight_info: ModelWeightInfo = self._weights_info.create_model_weight_info(database)
        use_fp32 = os.environ.get("USE_FLOAT32", None) is not None
        if use_fp32:
            compute_dtype = torch.float32
            
        self._load_config: LoadConfig = self._weights_info.create_load_config(compute_dtype, database, get_current_device())
        self.num_layers = self._load_config.num_layers
        self.num_replicas = self._load_config.phy_exp_num
        self.num_groups = self._load_config.moe_n_group
        self.num_nodes = self._load_config.num_nodes
        self.num_gpu = self._load_config.ep_size
        self.moe_layer_index = self._load_config.moe_layer_index
        self.num_experts = self._load_config.expert_num
        
        
        self.time_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.queue: Queue[int] = Queue()
        select_layer_method = os.environ.get("BALANCE_METHOD", "mix")
        self.select_layer_method = SelectLayerMethod(select_layer_method)
        self.phy2log = phy2log
        self.update_cnt = 0

        window_size = int(os.environ.get("EPLB_STATS_WINDOW_SIZE", 10))
        self.history_log_stats = HistoryStats(window_size=window_size, shape=(self._load_config.num_layers, self._load_config.expert_num))

        logging.info(f"Balance method: {self.select_layer_method}")

        # record the expert count, used for dump stats
        self.log_exp_cnt = torch.zeros((self._load_config.num_layers, self._load_config.expert_num), dtype=torch.int32)


        self.force_repack = os.environ.get("EPLB_FORCE_REPACK", "0") == "1"

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
        max_per_layer = gpu_loads.max(dim=1).values

        # note: select idx must in moe_layer_index
        most_unbalanced_idx = -1
        for idx in self.moe_layer_index:
            if most_unbalanced_idx == -1 or max_per_layer[idx] > max_per_layer[most_unbalanced_idx]:
                most_unbalanced_idx = idx

        return most_unbalanced_idx

    @torch.inference_mode()
    def create_balance_plan(
        self, log_stats: torch.Tensor, gpu_loads: torch.Tensor
    ):
        self.update_cnt += 1
        log_stats = self.history_log_stats.add_stats(log_stats)
        logging.info(log_stats)
        self.log_exp_cnt.copy_(log_stats)

        file_name = f"gpu_load_{self.time_prefix}.json"

        layer_id = self.get_balanced_layer(gpu_loads)

        phy2log, log2phy, logcnt = rebalance_experts(
            log_stats[layer_id : layer_id + 1],
            self.num_replicas,
            self.num_groups,
            self.num_nodes,
            self.num_gpu,
            self.force_repack
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
    def load_moe_weight(
        self,
        layer_id_tensor: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        phy2log: torch.Tensor,
    ):
        layer_id = int(layer_id_tensor.item())
        # Notice now that the phy2log is updated in load_config, not support multi thread
        self._load_config.udpate_layer_experts(layer_id, phy2log)
        choose_expert_id = self._load_config.get_selected_experts(layer_id)
        moe_weight = self._model_weight_info.get_layer_weight_info(layer_id, W.moe)
        assert moe_weight is not None
        logging.info(f"[EPLB_py][RANK {self._load_config.ep_rank}] Load MOE weight layer {layer_id} for {choose_expert_id}")

        res = moe_weight.load(self.database, layer_id, "cpu", self._load_config)

        logging.info(f"[EPLB_py][RANK {self._load_config.ep_rank}] Load MOE weight layer {layer_id} done")
        return layer_id, res.get(W.moe_w1), res.get(W.moe_w2), res.get(W.moe_w1_s), res.get(W.moe_w2_s)

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
            self.num_nodes,
            self.num_gpu,
        )

        k = log2phy.shape[-1]
        log2phy_pad[:, :k] = log2phy

        return phy2log.tolist(), log2phy.tolist(), logcnt.tolist()