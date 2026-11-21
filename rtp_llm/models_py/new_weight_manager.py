"""Online checkpoint updates for NewLoader's stable parameter/storage owners.

The service must pause every rank before updating and resume only after all
updates succeed. The shared lock also excludes in-flight model forwards.
Incoming names and tensors use global HF checkpoint layout; leaf loaders own
TP slicing and fusion. Each RPC validates/stages before changing live tensors.
"""

import re
import threading
from contextlib import nullcontext
from dataclasses import replace

import torch

from rtp_llm.models_py.runtime_weight_update import (
    RuntimeWeightTransaction,
    TensorUpdate,
    _signature,
)


def _checkpoint_wire_name(name):
    # Public RL clients use HF/verl names or Megatron's decoder spellings.
    # Normalize only the wire vocabulary; TP and fusion remain leaf-owned.
    prefix = ""
    value = name.removeprefix("decoder.").removeprefix("model.")
    match = re.match(r"layers\.([0-9]+)\.(.+)$", value)
    if match:
        prefix, value = f"layers.{match[1]}.", match[2]
    aliases = {
        "embedding.word_embeddings.weight": "embed_tokens.weight",
        "mbedding.word_embeddings.weight": "embed_tokens.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attention.linear_proj.bias": "self_attn.o_proj.bias",
        "self_attention.linear_qkv.weight": "self_attn.qkv_proj.weight",
        "self_attention.linear_qkv.bias": "self_attn.qkv_proj.bias",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    }
    if name.startswith("decoder.") or value in aliases:
        return "model." + prefix + aliases.get(value, value)
    return name


class NewLoaderWeightManager:
    requires_paused_engine = True

    def __init__(self, model, loader, *, adapters=None, admin_lock=None):
        self._full_model_contract = bool(
            getattr(model, "supports_checkpoint_weight_updates", False)
        )
        self._independent_owners = {
            module
            for module in model.modules()
            if getattr(module, "supports_independent_weight_updates", False)
            and callable(getattr(module, "prepare_weight_updates", None))
        }
        # Update capability belongs to the requested parameter owner, not the
        # model-wide quantization setting. FP4 models still own ordinary
        # embedding/norm/head (and bias) parameters supported by the old RPC.
        self._model = model
        self._loader = loader
        self._adapters = adapters
        self._lock = admin_lock if admin_lock is not None else threading.RLock()
        model._weight_update_lock = self._lock
        self.version = 0
        independent_parameters = {
            id(tensor)
            for module in self._independent_owners
            for tensor in module.parameters(recurse=False)
        }
        self._parameter_contract = {
            name: (tensor, _signature(tensor))
            for name, tensor in model.named_parameters(remove_duplicate=False)
            if self._full_model_contract or id(tensor) in independent_parameters
        }
        self._runtime_contract = {
            name: (tensor, _signature(tensor))
            for name, tensor in model.runtime_weight_view().items()
        }

    def _validate_contract(self):
        current = dict(self._model.named_parameters(remove_duplicate=False))
        if (
            self._full_model_contract
            and current.keys() != self._parameter_contract.keys()
        ):
            raise RuntimeError(
                "NewLoader parameter registrations changed after loading"
            )
        for name, (original, signature) in self._parameter_contract.items():
            if (
                current.get(name) is not original
                or _signature(current[name]) != signature
            ):
                raise RuntimeError(f"NewLoader parameter/storage changed for {name!r}")
        exported = self._model.runtime_weight_view()
        if exported.keys() != self._runtime_contract.keys():
            raise RuntimeError(
                "Exported runtime weight registrations changed after loading"
            )
        for name, (tensor, signature) in self._runtime_contract.items():
            if (
                _signature(tensor) != signature
                or _signature(exported[name]) != signature
            ):
                raise RuntimeError(f"Exported runtime storage changed for {name!r}")

    @torch.inference_mode()
    def update_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Online updates require a non-empty tensor mapping")
        guard = (
            self._adapters.base_update_guard()
            if self._adapters is not None
            else nullcontext()
        )
        with self._lock, guard:
            self._validate_contract()
            devices = {
                tensor.device
                for tensor, _ in self._parameter_contract.values()
                if tensor.is_cuda
            }
            for device in devices:
                torch.cuda.synchronize(device)
            groups = {}
            output_head = getattr(self._model, "lm_head", None)
            tied_output = getattr(
                self._model, "tie_word_embeddings", False
            ) and not getattr(output_head, "_explicit_checkpoint_loaded", False)
            for name, tensor in weights.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(tensor, torch.Tensor)
                ):
                    raise TypeError("Online updates require named checkpoint tensors")
                canonical = _checkpoint_wire_name(name)
                if not self._local_request(canonical):
                    continue
                mapped = self._loader._mapped_checkpoint_name(self._model, canonical)
                if canonical in self._runtime_contract:
                    # Old RPC clients may pass a published global view name.
                    # Resolve it by the tensor already exported by this model;
                    # do not introduce a parallel name→parameter registry.
                    _, signature = self._runtime_contract[canonical]
                    paths = [
                        p
                        for p, (_, v) in self._parameter_contract.items()
                        if v == signature
                    ]
                    if len(paths) != 1:
                        raise ValueError(
                            f"Published global {canonical!r} has no unique parameter owner"
                        )
                    mapped = paths[0]
                if mapped == "lm_head.weight" and tied_output:
                    raise ValueError(
                        "Tied LM head is a non-owning alias; update model.embed_tokens.weight"
                    )
                target, _ = self._model.resolve_checkpoint_load_plan(
                    mapped, tuple(tensor.shape)
                )
                if target is None or not callable(
                    getattr(target.module, "prepare_weight_updates", None)
                ):
                    # A custom parent loader (for example MLA attention) may
                    # own dispatch for its children. Exact canonical paths to
                    # explicitly independent leaves still have their own
                    # conversion contract. Do not guess aliases or descend
                    # into an arbitrary projection behind that custom loader.
                    path, separator, parameter_name = mapped.rpartition(".")
                    try:
                        owner = self._model.get_submodule(path) if separator else None
                    except AttributeError:
                        owner = None
                    if owner in self._independent_owners:
                        target = self._model._direct_checkpoint_target(
                            owner, parameter_name
                        )
                if target is None or not callable(
                    getattr(target.module, "prepare_weight_updates", None)
                ):
                    raise NotImplementedError(
                        f"No online checkpoint update contract for {name!r}"
                    )
                # The serving RPC never silently casts an ordinary parameter.
                # A fused shard (q_proj.weight, etc.) still targets the same
                # parameter dtype. Quantization units own their source dtypes.
                param = target.module._parameters.get(target.name.rsplit(".", 1)[-1])
                quant = getattr(
                    getattr(target.module, "quant_config", None), "quant_type", "none"
                )
                if (
                    param is not None
                    and quant == "none"
                    and tensor.dtype != param.dtype
                ):
                    raise TypeError(
                        f"Update for {name!r} requires {param.dtype}, got {tensor.dtype}"
                    )
                group = groups.setdefault(target.module, {})
                if target.name in group:
                    raise ValueError(f"Duplicate mapped checkpoint tensor {mapped!r}")
                group[target.name] = tensor.detach().to(device="cpu", copy=True)
                if (
                    mapped == "embed_tokens.weight"
                    and tied_output
                    and output_head is not None
                ):
                    groups.setdefault(self._model.lm_head, {})["weight"] = group[
                        target.name
                    ]

            if not groups:
                return  # PP client broadcasts global requests to every rank.
            module_names = {
                module: name for name, module in self._model.named_modules()
            }
            # Validate ownership and quantized-target capability for the entire
            # request before even running private conversion of another leaf.
            for module, incoming in groups.items():
                module_name = module_names[module]
                if (
                    not self._full_model_contract
                    and module not in self._independent_owners
                ):
                    raise NotImplementedError(
                        f"{module_name} has no independent online update contract; "
                        "only parameter owners with an explicit conversion contract are enabled"
                    )
                borrowed = getattr(module, "_rtp_borrowed_parameter_names", frozenset())
                if borrowed:
                    raise PermissionError(
                        f"{module_name} has non-owning parameters {sorted(borrowed)}; "
                        "update their owner instead"
                    )
                if (
                    getattr(module, "quant_method", None) is not None
                    and getattr(
                        getattr(module, "quant_config", None), "quant_type", "none"
                    )
                    != "none"
                ):
                    from rtp_llm.models_py.quantized_weight_update import (
                        validate_quantized_weight_update,
                    )

                    validate_quantized_weight_update(module, incoming)
            updates = []
            for module, incoming in groups.items():
                module_name = module_names[module]
                if (
                    module is output_head
                    and tied_output
                    and set(incoming) == {"weight"}
                ):
                    source = module._prepare_checkpoint_weight(
                        incoming["weight"], None, tied=True
                    )
                    leaf_updates = [TensorUpdate("weight", module.weight, source)]
                else:
                    leaf_updates = module.prepare_weight_updates(incoming)
                if not leaf_updates:
                    raise RuntimeError(
                        "Checkpoint conversion produced no runtime updates"
                    )
                for update in leaf_updates:
                    target_tensor = module._parameters[update.name]
                    if getattr(
                        target_tensor, "_rtp_requires_restart_for_update", False
                    ):
                        raise PermissionError(
                            f"Parameter {module_name}.{update.name} requires restart to update"
                        )
                    source = update.source
                    if module is output_head and update.name == "weight":
                        if getattr(self._model, "normalize_lm_head_weight", False):
                            source = torch.nn.functional.normalize(source, dim=1)
                        source = source * getattr(self._model, "logit_scale", 1.0)
                    name = (
                        f"{module_name}.{update.name}" if module_name else update.name
                    )
                    updates.append(
                        replace(
                            update,
                            name=name,
                            target=module._parameters[update.name],
                            source=source,
                        )
                    )
            transaction = RuntimeWeightTransaction(updates)
            transaction.install()
            transaction.commit()
            self.version += 1

    def _local_request(self, name):
        layer = re.match(r"^(?:model\.)?layers\.(\d+)\.", name)
        if layer and int(layer[1]) >= self._loader.model_config.num_layers:
            raise ValueError(f"Checkpoint layer out of range: {name}")
        select = getattr(self._model, "checkpoint_weight_name_filter", None)
        owns = select() if callable(select) else None
        # Use stage filtering only for known layer/global ownership, not to
        # silently accept arbitrary unknown names rejected by a model's filter.
        if layer or name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.norm.bias",
            "lm_head.weight",
        ):
            return owns is None or owns(name)
        return True

    def update(self, req: dict[str, str]) -> None:
        for field in ("name", "method", "desc"):
            if not isinstance(req.get(field), str) or not req[field]:
                raise ValueError(f"Missing or invalid online update field {field!r}")
        # Import transport dependencies only for RPC callers. Conversion tests
        # and local update_tensors do not require CUDA IPC runtime libraries.
        if req["method"] == "shm":
            from rtp_llm.model_loader.tipc import (
                SharedMemIpcMeta,
                SharedMemoryIPCHelper,
            )

            tensor = SharedMemoryIPCHelper().build_from_meta(
                SharedMemIpcMeta.decode(req["desc"])
            )
        elif req["method"] == "cuda_ipc":
            from rtp_llm.model_loader.tipc import CudaIpcHelper

            tensor = CudaIpcHelper().build_from_meta(bytes.fromhex(req["desc"]))
        else:
            raise ValueError(f"Unsupported online update transport {req['method']!r}")
        name = _checkpoint_wire_name(req["name"])
        # The existing RPC's full fused/linear matrices are in GEMM [in,out]
        # order (including square O projections); shape heuristics silently
        # transpose the wrong values. Local update_tensors uses HF [out,in].
        if tensor.ndim == 2 and name.endswith(
            (
                ".self_attn.qkv_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
            )
        ):
            tensor = tensor.t()
        self.update_tensors({name: tensor})
