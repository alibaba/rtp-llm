import datetime
import gc
import json
import logging
import random
import time
import traceback
from collections import deque
from enum import Enum
from queue import Queue
from typing import Any, Deque, Optional, Sequence

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device import get_current_device
from rtp_llm.eplb.eplb import rebalance_experts
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.utils.database import BaseDatabase
from rtp_llm.utils.model_weight import W


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
        phy2log: Any,
        database: BaseDatabase,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize ExpertBalancer.

        Args:
            weights_info: Model weight information
            compute_dtype: Compute data type
            phy2log: Physical to logical expert mapping
            database: Database for loading weights
            model_config: Optional ModelConfig (used to get eplb_config)
        """
        self.database: BaseDatabase = database
        self._weights_info: ModelDeployWeightInfo = weights_info
        self._model_weight_info: ModelWeightInfo = (
            self._weights_info.create_model_weight_info(database)
        )

        self._load_config: LoadConfig = self._weights_info.create_load_config(
            compute_dtype=compute_dtype,
            database=database,
            phy2log=phy2log,
            exported_device=get_current_device(),
        )
        self.num_layers = self._load_config.num_layers
        self.num_replicas = self._load_config.phy_exp_num
        self.num_groups = self._load_config.moe_n_group
        self.num_nodes = self._load_config.num_nodes
        self.num_gpu = self._load_config.ep_size
        self.moe_layer_index = self._load_config.moe_layer_index
        self.num_experts = self._load_config.expert_num

        self.time_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.queue: Queue[int] = Queue()

        # Get balance_method from model_config.eplb_config
        balance_method = "mix"  # default
        if model_config is not None:
            balance_method = model_config.eplb_config.balance_method

        self.select_layer_method = SelectLayerMethod(balance_method)
        self.phy2log = phy2log
        self.update_cnt = 0

        # Get window_size from model_config.eplb_config
        eplb_stats_window_size = 10  # default
        if model_config is not None:
            eplb_stats_window_size = model_config.eplb_config.eplb_stats_window_size

        self.history_log_stats = HistoryStats(
            window_size=eplb_stats_window_size,
            shape=(self._load_config.num_layers, self._load_config.expert_num),
        )

        logging.info(f"Balance method: {self.select_layer_method}")

        # record the expert count, used for dump stats
        self.log_exp_cnt = torch.zeros(
            (self._load_config.num_layers, self._load_config.expert_num),
            dtype=torch.int32,
        )

        # Get force_repack from model_config.eplb_config
        eplb_force_repack = 0  # default
        if model_config is not None:
            eplb_force_repack = model_config.eplb_config.eplb_force_repack

        self.force_repack = eplb_force_repack == 1

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
            if most_unbalanced_idx == -1 or (
                idx < max_per_layer.shape[0]
                and max_per_layer[idx] > max_per_layer[most_unbalanced_idx]
            ):
                most_unbalanced_idx = idx

        return most_unbalanced_idx

    @torch.inference_mode()
    def create_balance_plan(
        self,
        log_stats: torch.Tensor,
        gpu_loads: torch.Tensor,
        active_ranks: torch.Tensor,
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
            self.force_repack,
            active_ranks,
        )

        self.phy2log[layer_id] = phy2log.tolist()
        log_data = {
            "update_cnt": self.update_cnt,
            "gpu_loads": gpu_loads.tolist(),
            "log_stats": log_stats.tolist(),
            "layer": layer_id,
            "plan": phy2log.tolist(),
            "phy2log": self.phy2log,
            "update_time": time.ctime(),
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

        gc.collect()
        dtype = torch.int32
        return (
            torch.tensor([layer_id], dtype=torch.int32).contiguous(),
            logcnt[0].to(dtype).contiguous(),
            log2phy_pad.to(dtype).contiguous(),
            phy2log[0].to(dtype).contiguous(),
        )

    @torch.inference_mode()
    def create_downscale_plan(
        self, log_stats: torch.Tensor, active_ranks: torch.Tensor
    ):
        phy2log, log2phy, logcnt = rebalance_experts(
            log_stats,
            self.num_replicas,
            self.num_groups,
            self.num_nodes,
            self.num_gpu,
            self.force_repack,
            active_ranks,
        )

        # Get number of layers
        num_layers = log_stats.shape[0]
        pad_k = self.num_replicas - self.num_experts + 1

        # Convert log2phy to int32 for consistency
        log2phy = log2phy.to(torch.int32)

        # Create padded log2phy for all layers: [num_layers, num_experts, pad_k]
        log2phy_pad = torch.empty(
            (num_layers, self.num_experts, pad_k), dtype=torch.int32, device="cpu"
        )
        log2phy_pad.fill_(-1)

        # Pad each layer's log2phy
        k = log2phy.shape[-1]
        for layer_id in range(num_layers):
            log2phy_pad[layer_id, :, :k] = log2phy[layer_id]
            logging.info(
                f"[EPLB_py PLAN] phy2log for layer {layer_id}: {phy2log[layer_id]}"
            )

        gc.collect()
        dtype = torch.int32

        return (
            logcnt.to(dtype).contiguous(),  # [num_layers, num_experts]
            log2phy_pad.to(dtype).contiguous(),  # [num_layers, num_experts, pad_k]
            phy2log.to(dtype).contiguous(),  # [num_layers, num_replicas]
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
        choose_expert_id = self._load_config.get_selected_experts(
            layer_id, self.num_experts
        )
        moe_weight = self._model_weight_info.get_layer_weight_info(layer_id, W.moe)
        assert moe_weight is not None
        logging.info(
            f"[EPLB_py][RANK {self._load_config.ep_rank}] Load MOE weight layer {layer_id} for {choose_expert_id}"
        )
        try:
            res = moe_weight.load(
                DatabaseTensorSource(self.database), layer_id, "cpu", self._load_config
            )
        except:
            logging.error(
                f"[EPLB_py][RANK {self._load_config.ep_rank}] Load MOE weight layer failed: 完整堆栈:\n{traceback.format_exc()}"
            )

        logging.info(
            f"[EPLB_py][RANK {self._load_config.ep_rank}] Load MOE weight layer {layer_id} done"
        )
        gc.collect()
        return (
            layer_id,
            res.get(W.moe_w1),
            res.get(W.moe_w2),
            res.get(W.moe_s1, torch.empty(0, dtype=torch.float32)),
            res.get(W.moe_s2, torch.empty(0, dtype=torch.float32)),
        )

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
