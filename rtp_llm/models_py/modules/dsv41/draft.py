"""V4.1 prefill-only DSpark history projection into caller-owned SWA pages.

Only valid target aux rows are selected before any projection. This component
does not execute draft queries, attention, mHC, MoE, embeddings or heads. The
caller owns CP assembly, checkpoint protection and publication of SWA bounds.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.attention import attention_rope
from rtp_llm.models_py.modules.dsv41.cache_layout import DRAFT_LAYERS
from rtp_llm.models_py.modules.dsv41.ced import (
    AuxRowMap,
    ReplayConfig,
    ReplayMode,
    select_aux_hidden,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import SwaBinding
from rtp_llm.models_py.modules.dsv41.compact_writer import (
    CompactWriteResult,
    write_compact,
)
from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear, is_supported
from rtp_llm.models_py.modules.dsv41.math import rms_norm


@dataclass(frozen=True)
class V41DraftCommitResult:
    positions: tuple[int, ...]
    logical_rows: tuple[int, ...]
    image_mask: tuple[bool, ...]
    main_projection_rows: int
    stage_projection_rows: tuple[int, ...]
    writes: tuple[CompactWriteResult, ...]
    replay_floor: int
    write_completed: bool = True


class V41PrefillDraftCommit(nn.Module):
    def __init__(
        self,
        config,
        main_projection: nn.Module,
        main_norm: torch.Tensor,
        stage_projections: Sequence[nn.Module],
        stage_norms: Sequence[torch.Tensor],
    ):
        super().__init__()
        text = config.text
        expected = {
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "num_nextn_predict_layers": 3,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "sliding_window": 128,
            "rope_theta": 10000,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-20,
        }
        if (
            any(text[name] != value for name, value in expected.items())
            or tuple(text["dspark_target_layer_ids"]) != (37, 38, 39)
            or tuple(text["compress_ratios"][40:]) != (0, 0, 0)
        ):
            raise ValueError("prefill draft commit requires the V4.1 DSpark geometry")
        if not is_supported(main_norm):
            raise RuntimeError("V4.1 draft commit requires CUDA13 on a Blackwell GPU")
        if len(stage_projections) != 3 or len(stage_norms) != 3:
            raise ValueError("prefill draft commit requires all three SWA stages")
        projections = (main_projection, *stage_projections)
        shapes = ((15360, 5120), *((5120, 512),) * 3)
        for projection, (inputs, outputs) in zip(projections, shapes):
            if (
                not isinstance(projection, nn.Module)
                or getattr(projection, "in_features", None) != inputs
                or getattr(projection, "out_features", None) != outputs
                or projection.weight.device != main_norm.device
            ):
                raise ValueError("draft projection geometry or weight device changed")
        for norm, dimension in zip((main_norm, *stage_norms), (5120, 512, 512, 512)):
            if (
                norm.shape != (dimension,)
                or norm.dtype != torch.bfloat16
                or norm.device != main_norm.device
                or not norm.is_contiguous()
            ):
                raise ValueError("draft norm must retain its checkpoint BF16 vector")
        self.main_projection = main_projection
        self.register_buffer("main_norm", main_norm)
        self.stage_projections = nn.ModuleList(stage_projections)
        for stage, norm in enumerate(stage_norms):
            self.register_buffer(f"stage_norm_{stage}", norm)

    @classmethod
    def from_model_weights(
        cls, config, weights, *, projection_factory=V41Block32Linear
    ):
        """Bind the installed three-layer descriptor without cloning its weights.

        Global names retain ``v41.mtp.0.*``; each layer uses the same
        ``v41.attn.*`` names as the existing target descriptor. Any target
        embedding/head aliases in the ModelWeights are not bound or used.
        """
        if len(weights.weights) != 3:
            raise ValueError("installed draft weights must contain three stages")
        globals_ = weights.global_weights
        prefix = "v41.mtp.0."
        main = projection_factory(
            globals_[prefix + "main_proj.weight"],
            globals_[prefix + "main_proj.scale"],
        )
        projections, norms = [], []
        for installed in weights.weights:
            if any(not name.startswith("v41.") for name in installed):
                raise ValueError("draft binding received another model weight layout")
            projections.append(
                projection_factory(
                    installed["v41.attn.wkv.weight"],
                    installed["v41.attn.wkv.scale"],
                )
            )
            norms.append(installed["v41.attn.kv_norm.weight"])
        return cls(
            config, main, globals_[prefix + "main_norm.weight"], projections, norms
        )

    @torch.inference_mode()
    def commit(
        self,
        aux_hidden: torch.Tensor,
        row_map: AuxRowMap,
        *,
        required_positions: Sequence[int],
        swa_bindings: Mapping[int, SwaBinding],
        request_id: str,
        forward_epoch: int,
        replay_fingerprint: str,
        replay_floor: int = 0,
        request_index: int = 0,
    ) -> V41DraftCommitResult:
        """Write only selected valid aux rows, then check all GPU writer statuses.

        SWA metadata is deliberately untouched: sparse CP-local rows do not
        certify a complete global interval. The caller updates validity and
        checkpoint state after all ranks and all three writes have completed.
        A failure may leave already-written stages modified; discard or restore
        that request's private cache before retrying.
        """
        positions = tuple(required_positions)
        selected_indices = row_map.selection(
            positions,
            request_id=request_id,
            forward_epoch=forward_epoch,
            replay_fingerprint=replay_fingerprint,
        )
        if replay_fingerprint not in (
            ReplayConfig(ReplayMode.FULL).fingerprint,
            ReplayConfig(ReplayMode.BOUNDED).fingerprint,
        ):
            raise ValueError("draft commit received an unsupported replay policy")
        if (
            type(replay_floor) is not int
            or not 0 <= replay_floor <= 1048576
            or any(not replay_floor <= position < 1048576 for position in positions)
            or (
                replay_floor
                and replay_fingerprint != ReplayConfig(ReplayMode.BOUNDED).fingerprint
            )
        ):
            raise ValueError("draft commit positions or replay floor are invalid")
        if (
            not is_supported(aux_hidden)
            or aux_hidden.dtype != torch.bfloat16
            or aux_hidden.device != self.main_norm.device
            or not aux_hidden.is_contiguous()
        ):
            raise ValueError("draft aux must be contiguous BF16 on the weight GPU")
        if torch.is_autocast_enabled():
            raise RuntimeError("draft commit requires autocast off")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prefill draft commit is an eager, checked writer")
        if set(swa_bindings) != set(DRAFT_LAYERS):
            raise ValueError("draft commit requires SWA bindings for layers 40/41/42")
        if type(request_index) is not int or request_index < 0:
            raise ValueError("draft request index must be nonnegative")
        destinations, occupied = [], []
        for layer in DRAFT_LAYERS:
            binding = swa_bindings[layer]
            requests = binding.validate(aux_hidden.device)
            if request_index >= requests:
                raise ValueError("draft request index is outside its SWA binding")
            entries = binding.pages.entries_per_page
            if entries < 133:
                raise ValueError("draft SWA must retain window128 plus gamma5 slack")
            offsets = tuple(position % entries for position in positions)
            if len(set(offsets)) != len(offsets):
                raise ValueError("selected draft rows collide in the destination ring")
            if positions:
                page_id = int(binding.page_ids[request_index].item())
                if not 0 < page_id < binding.pages.data.shape[0]:
                    raise ValueError("draft SWA destination page is unmapped")
                first_byte = (
                    binding.pages.data.data_ptr()
                    + page_id * binding.pages.data.stride(0)
                )
                end_byte = first_byte + binding.pages.data.shape[1]
                if any(
                    first_byte < end and start < end_byte for start, end in occupied
                ):
                    raise ValueError("draft stages must not alias their SWA pages")
                occupied.append((first_byte, end_byte))
            destinations.append((binding, entries))
        selected = select_aux_hidden(
            aux_hidden,
            row_map,
            positions,
            request_id=request_id,
            forward_epoch=forward_epoch,
            replay_fingerprint=replay_fingerprint,
        )
        absolute = torch.tensor(positions, dtype=torch.int64, device=selected.device)
        count = len(positions)
        writes = []
        if count:
            main_x = rms_norm(self.main_projection(selected), self.main_norm)
            for stage, (binding, entries) in enumerate(destinations):
                values = rms_norm(
                    self.stage_projections[stage](main_x),
                    getattr(self, f"stage_norm_{stage}"),
                )
                values = attention_rope(values, absolute, global_branch=False)
                slots = (
                    binding.page_ids[request_index] * entries + absolute % entries
                ).contiguous()
                result = write_compact(values.contiguous(), binding.pages, slots)
                result.check()
                writes.append(result)
        else:
            for binding, _ in destinations:
                writes.append(
                    CompactWriteResult(
                        binding.pages.data,
                        torch.empty(0, dtype=torch.int32, device=selected.device),
                    )
                )
        return V41DraftCommitResult(
            positions,
            tuple(row_map.logical_rows[index] for index in selected_indices),
            tuple(row_map.image_mask[index] for index in selected_indices),
            count,
            (count, count, count),
            tuple(writes),
            replay_floor,
        )
