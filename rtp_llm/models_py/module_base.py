import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_ALLOWED_DROPPED_WEIGHT_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
    "rope.inv_freq",
    "inv_freq",
)


def _is_allowed_dropped_weight(name: str) -> bool:
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _ALLOWED_DROPPED_WEIGHT_SUFFIXES)


class RtpModule(nn.Module):
    """Base class for RTP-LLM new-loader modules.

    Provides per-tensor streaming weight loading via ``load_weights``. Each
    incoming tensor immediately walks the module tree to its leaf and copies
    directly, so peak memory stays at model-params + one ckpt tensor (no
    intermediate dict buffering).

    Subclasses choose one of two roles:
      * **Container** modules inherit ``load_weights`` unchanged and act as
        recursive dispatchers into their children.
      * **Leaf** modules (e.g. ``MergedColumnParallelLinear``,
        ``BaseMoEExperts``) override ``load_weights`` to handle TP sharding,
        FP8/INT4 quantization fusion, or expert-parallel slicing themselves.
        Their override wins via normal MRO; the parent's recursion stops at
        them via ``hasattr(child, "load_weights")``.
    """

    def _build_redirect(self) -> Dict[str, Tuple[Any, int, str]]:
        redirect: Dict[str, Tuple[Any, int, str]] = {}
        for name, child in self.named_children():
            if hasattr(child, "shard_names") and child.shard_names:
                for idx, shard_name in enumerate(child.shard_names):
                    redirect[shard_name] = (child, idx, name)
        return redirect

    def _is_fused_target(self, child: nn.Module) -> bool:
        return (
            hasattr(child, "shard_names")
            and child.shard_names
            and hasattr(child, "load_weights")
        )

    def _get_child_module(self, name: str) -> Optional[nn.Module]:
        if hasattr(self, name):
            return getattr(self, name)
        try:
            idx = int(name)
            for attr_name, attr in self.named_children():
                if isinstance(attr, nn.ModuleList) and idx < len(attr):
                    return attr[idx]
        except (ValueError, IndexError):
            pass
        return None

    def _assign_weight(self, module: nn.Module, name: str, tensor: torch.Tensor) -> bool:
        if "." in name:
            prefix, rest = name.split(".", 1)
            child = getattr(module, prefix, None)
            if child is None:
                return False
            return self._assign_weight(child, rest, tensor)

        if not hasattr(module, name):
            return False
        param = getattr(module, name)
        if not isinstance(param, nn.Parameter):
            return False
        try:
            param.data.copy_(tensor)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error copying weight to parameter '{name}' in module '{module.__class__.__name__}': "
                f"target shape is {list(param.shape)}, source shape is {list(tensor.shape)}. Original error: {e}"
            ) from e
        return True

    def _groupby_prefix(
        self, weights: Iterator[Tuple[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for full_name, tensor in weights:
            if "." in full_name:
                prefix, rest = full_name.split(".", 1)
                grouped[prefix][rest] = tensor
            else:
                grouped["_self_"][full_name] = tensor
        return dict(grouped)

    def load_weights(self, weights: Any):
        """Per-tensor streaming dispatch: each tensor immediately walks the
        module tree to its leaf and copies directly. No intermediate dict
        buffering, so peak memory stays at model-params + one ckpt tensor.
        """
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        redirect = self._build_redirect()
        dropped: List[str] = []

        for full_name, tensor in weights_iter:
            if "." not in full_name:
                if hasattr(self, full_name):
                    param = getattr(self, full_name)
                    if isinstance(param, nn.Parameter):
                        param.data.copy_(tensor)
                    else:
                        dropped.append(full_name)
                else:
                    dropped.append(full_name)
                continue

            prefix, rest = full_name.split(".", 1)

            if prefix in redirect:
                layer, _shard_id, _target_name = redirect[prefix]
                layer.load_weights({f"{prefix}.{rest}": tensor})
            else:
                child = self._get_child_module(prefix)
                if child is None:
                    dropped.append(full_name)
                    continue

                if isinstance(child, nn.ModuleList):
                    if not _dispatch_single_to_module_list(self, child, rest, tensor):
                        dropped.append(full_name)
                elif hasattr(child, "load_weights"):
                    child.load_weights({rest: tensor})
                elif not self._assign_weight(child, rest, tensor):
                    dropped.append(full_name)

        if dropped:
            cls_name = self.__class__.__name__
            allowed = [name for name in dropped if _is_allowed_dropped_weight(name)]
            unexpected = [name for name in dropped if not _is_allowed_dropped_weight(name)]
            if unexpected:
                sample = unexpected[:10]
                more = (
                    f" (+{len(unexpected) - len(sample)} more)"
                    if len(unexpected) > len(sample)
                    else ""
                )
                raise RuntimeError(
                    f"[{cls_name}] {len(unexpected)} weight(s) had no matching "
                    f"submodule: {sample}{more}. Refusing to continue with "
                    "possibly uninitialized parameters. Add an explicit allowlist "
                    "entry only for known non-persistent checkpoint tensors."
                )
            if allowed:
                sample = allowed[:10]
                more = (
                    f" (+{len(allowed) - len(sample)} more)"
                    if len(allowed) > len(sample)
                    else ""
                )
                logger.info(
                    "[%s] ignored %d known non-persistent checkpoint tensor(s): %s%s",
                    cls_name,
                    len(allowed),
                    sample,
                    more,
                )


def _dispatch_single_to_module_list(
    parent: nn.Module, module_list: nn.ModuleList, name: str, tensor: torch.Tensor
) -> bool:
    if "." not in name:
        return False
    idx_str, rest = name.split(".", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return False
    if idx < 0 or idx >= len(module_list):
        return False
    child = module_list[idx]
    if hasattr(child, "load_weights"):
        child.load_weights({rest: tensor})
        return True
    if isinstance(parent, RtpModule):
        return parent._assign_weight(child, rest, tensor)
    return False


def rtp_module(cls):
    """Deprecated. Use ``class Foo(RtpModule)`` instead.

    Kept as a no-op alias so any external/legacy code still importing the
    decorator does not break.
    """
    return cls
