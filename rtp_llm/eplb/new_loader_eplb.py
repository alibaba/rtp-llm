"""动态 EPLB 适配（新 loader / py-model 模式）。

旧 loader 的动态 EPLB：C++ `ExpertBalancer` 回调 Python `py_eplb.load_moe_weight`
从 ckpt 重载重排后的专家权重 → C++ `exchange_dense_weight` 把 `gpt_weights` 指针指向新权重
→ 下次 C++ forward 直接读到。

新 loader 的 forward 在 **py_model**（Python `Qwen3Experts.w13/w2`）里跑，C++ 的指针替换到不了它。
本模块的做法：复用旧 `ExpertBalancer`（它带 `ModelWeightInfo`+`CkptDatabase`，能从 ckpt 重载），
**包一层**——`load_moe_weight` 重载得到 `W.moe_w1/w2` 后，额外把它们 in-place `copy_` 进
py_model 对应层的 `experts.w13/w2`（布局一致：`[E_local, 2*M_tp, H]`、`[up;gate]`），
这样 py_model 的 forward 立刻看到重排结果。返回值仍给 C++，保持两边一致。

NOTE: 当前仅适配 `REDUNDANT_EXPERT=0`（运行期重定位）。冗余专家（phy_exp_num > expert_num）
需要 `BaseMoEExperts` 按 phy_exp_num 分配更多本地槽位，属后续工作。
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _collect_layer_experts(py_model: Any) -> Dict[int, Any]:
    """layer_id -> Qwen3Experts(带 w13/w2) 的索引。"""
    lang = getattr(py_model, "language_model", py_model)
    layers = getattr(lang, "layers", None)
    out: Dict[int, Any] = {}
    if layers is None:
        return out
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is not None and hasattr(experts, "w13") and hasattr(experts, "w2"):
            out[i] = experts
    return out


def _copy_into(param: Any, src: torch.Tensor) -> None:
    """把重载的权重 in-place copy 进 py_model 的 nn.Parameter（自动转 device/dtype/shape）。"""
    dst = param.data
    t = src.to(device=dst.device, dtype=dst.dtype)
    if tuple(t.shape) != tuple(dst.shape):
        t = t.reshape(dst.shape)
    dst.copy_(t)


def build_new_loader_eplb(model: Any, py_model: Any) -> Optional[Any]:
    """为新 loader 构造 py_eplb（动态 EPLB）。未开启 EPLB 返回 None。"""
    from rtp_llm.eplb.ep_balancer import ExpertBalancer
    from rtp_llm.model_loader.load_config import LoadConfig
    from rtp_llm.utils.database import CkptDatabase

    mc = model.model_config
    try:
        weights_info = model.get_weight_cls()(
            model_config=mc,
            parallelism_config=model.parallelism_config,
            hw_kernel_config=model.hw_kernel_config,
            kv_cache_config=model.kv_cache_config,
            merge_lora=getattr(model, "merge_lora", False),
            load_method=getattr(model, "load_method", None),
        )
    except Exception as e:
        logger.warning("[EPLB][new_loader] 构造 weights_info 失败，跳过 EPLB: %s", e)
        return None

    if not getattr(weights_info, "enable_eplb_", False):
        return None

    layer_experts = _collect_layer_experts(py_model)
    if not layer_experts:
        logger.warning("[EPLB][new_loader] py_model 里没找到 MoE experts，跳过 EPLB")
        return None

    class _PyModelExpertBalancer(ExpertBalancer):
        """重载后把权重同步进 py_model 的 w13/w2（in-place，py_model 即时可见）。"""

        def load_moe_weight(self, layer_id_tensor, ep_rank, ep_size, phy2log):
            result = super().load_moe_weight(layer_id_tensor, ep_rank, ep_size, phy2log)
            try:
                layer_id, moe_w1, moe_w2 = result[0], result[1], result[2]
                experts = layer_experts.get(int(layer_id))
                if experts is not None:
                    if moe_w1 is not None:
                        _copy_into(experts.w13, moe_w1)
                    if moe_w2 is not None:
                        _copy_into(experts.w2, moe_w2)
            except Exception as e:
                logger.error("[EPLB][new_loader] 同步重排权重进 py_model 失败: %s", e)
            return result

    phy2log = LoadConfig.create_redundant_expert(
        layer_num=mc.num_layers,
        expert_num=mc.expert_num,
        ep_size=weights_info.ep_size,
        num_nodes=weights_info.num_nodes,
        phy_exp_num=weights_info.phy_exp_num_,
        phy2log_path=getattr(mc, "phy2log_path", None),
    )
    database = CkptDatabase(mc.ckpt_path)

    balancer = _PyModelExpertBalancer(
        weights_info=weights_info,
        compute_dtype=mc.compute_dtype,
        phy2log=phy2log,
        database=database,
        model_config=mc,
    )
    logger.info(
        "[EPLB][new_loader] 已接入动态 EPLB：%d 个 MoE 层，phy_exp_num=%d expert_num=%d",
        len(layer_experts),
        weights_info.phy_exp_num_,
        mc.expert_num,
    )
    return balancer
