"""Eager PPU DSpark on RTP's commit/proposal protocol.

Computation adapted from SGLang, Apache-2.0, revision
b313eda591b53f263c811ac001239855506f64ba: deepseek_v4_dspark.py
(DSparkAttention, commit_hidden_states) and deepseek_v4_platform_stage.py
(NonCudaDSparkMixin). No SGLang scheduler, pool allocator or batch type is used.
"""

import logging
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import primary_attention_inputs
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.cp import DistributedCPGroup, build_zigzag_plan
from rtp_llm.models_py.modules.dsv41.ppu.dspark_weight import load_draft_weights
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_model import NativeModelRunner
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import (
    _rms_norm,
    _rope_tail,
    _torch_sparse_attention,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.ops.compute_ops import PyModelOutputs

logger = logging.getLogger(__name__)


class DeepSeekV41PPUDSparkModel(DSparkProposerMixin, torch.nn.Module):
    """Three independent SWA projections and a noncausal noise query block."""

    def __init__(
        self,
        model_config,
        parallelism_config,
        weight,
        moe_config,
        max_generate_batch_size,
        py_hw_kernel_config=None,
        **kwargs,
    ):
        super().__init__()
        if bool(getattr(py_hw_kernel_config, "enable_cuda_graph", False)):
            raise ValueError(
                "PPU DSpark Graph is not implemented; S1 requires explicit eager mode"
            )
        role = (
            str(getattr(parallelism_config, "role_type", "")).upper().rsplit(".", 1)[-1]
        )
        self.commit_only = role == "PREFILL"
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        config = dict(model_config.deepseek_v41_config)
        self.weights = load_draft_weights(
            config,
            config.get("checkpoint_path") or model_config.ckpt_path,
            ep_rank=int(parallelism_config.ep_rank),
            ep_size=int(parallelism_config.ep_size),
            device=self.device,
            commit_only=self.commit_only,
            confidence_enabled=bool(
                getattr(model_config, "dspark_confidence_enabled", False)
            ),
        )
        self.config = PrefillPolicy.from_config(config).freeze(self.weights.config)
        self.bundle = self.weights.bundle
        self.geometry = self.bundle.geometry
        gamma = int(model_config.gen_num_per_cycle)
        query_width = gamma + int(
            not getattr(model_config, "dspark_sample_from_anchor", True)
        )
        self.init_dspark_proposer(
            width=gamma,
            query_width=query_width,
            noise_token_id=int(model_config.dspark_noise_token_id),
            aux_feature_dim=len(model_config.dspark_target_layer_ids)
            * self.geometry.hidden_size,
            hidden_dim=self.geometry.hidden_size,
        )
        self.tp_size = int(parallelism_config.tp_size)
        self.tp_rank = int(parallelism_config.tp_rank)
        dp_size = int(parallelism_config.dp_size)
        import torch.distributed as dist

        group = dist.group.WORLD if dist.is_initialized() else None
        self.ops = None
        if not self.commit_only:
            g = self.geometry
            self.ops = PPUOps(
                ep_rank=int(parallelism_config.ep_rank),
                ep_size=int(parallelism_config.ep_size),
                num_routed_experts=g.n_routed_experts,
                num_local_experts=g.num_local_experts(int(parallelism_config.ep_size)),
                hidden_size=g.hidden_size,
                inter_dim=g.moe_inter_dim,
                swiglu_limit=g.swiglu_limit,
                mhc_config=g.mhc_config(),
                sm_scale=g.sm_scale(),
                window=g.window,
                deepep_group=group,
            )
            self.bundle.bind(self.ops)
        self.runner = NativeModelRunner(
            SimpleNamespace(
                config=self.config,
                weights=self.bundle,
                geometry=self.geometry,
                ops=self.ops,
                device=self.device,
                capture_layer_ids=(),
                dspark_model_role="draft",
            ),
            dp_rank=int(parallelism_config.dp_rank),
            dp_size=dp_size,
            cp_rank=self.tp_rank,
            cp_size=self.tp_size if dp_size == 1 else 1,
            group=group,
        )
        self.kv_cache = None
        logger.info(
            "PPU DSpark layers=3 target_features=%d experts=%d topk=%d gamma=%d Q=%d commit_only=%s",
            self._dspark_aux_feature_dim,
            self.geometry.n_routed_experts,
            self.geometry.n_activated_experts,
            gamma,
            query_width,
            self.commit_only,
        )

    def initialize(self, resources):
        self.kv_cache = resources.kv_cache
        return self.runner.initialize(resources)

    def bind_native_stop(self, requested, observed):
        self.runner.bind_native_stop(requested, observed)

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False, **kwargs):
        if is_cuda_graph:
            raise ValueError("PPU DSpark proposal/commit Graph requires S4")
        return None

    def combine_hidden_states(self, features):
        return _rms_norm(
            self.weights.main_proj(features),
            self.weights.main_norm,
            self.geometry.norm_eps,
        )

    def map_commit_rows(self, starts, lengths, committed_ends, row_count, inputs):
        primary = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
        cp = getattr(primary, "context_parallel_info", None)
        mask = getattr(cp, "prefill_qkv_padding_mask_cpu", None)
        if self.runner.cp_size <= 1 or mask is None or not mask.numel():
            return super().map_commit_rows(
                starts, lengths, committed_ends, row_count, inputs
            )
        full_lengths = cp.prefill_actual_input_lengths_cpu.cpu().long()
        cu = torch.cat((torch.zeros(1, dtype=torch.long), full_lengths.cumsum(0)))
        plan = build_zigzag_plan(
            torch.zeros(int(cu[-1]), dtype=torch.long),
            cu,
            cp_size=self.runner.cp_size,
            cp_rank=self.tp_rank,
            prefix_lengths=cp.prefill_prefix_lengths_cpu.tolist(),
        )
        if row_count != plan.chunk_length:
            raise ValueError("draft features do not match CP rank-local zigzag rows")
        valid = plan.local_is_real.to(self.device)
        req = plan.local_request_ids.to(device=self.device, dtype=torch.int32)
        pos = plan.local_positions.to(device=self.device, dtype=torch.int32)
        return req.masked_fill(~valid, -1), pos.masked_fill(~valid, -1), plan

    def _bridge(self, inputs):
        return NativeCacheBridge(
            self.config, self.kv_cache, inputs.attention_inputs, device=self.device
        )

    def commit_feature_rows(
        self,
        main_x,
        context_req_ids,
        context_positions,
        committed_ends,
        inputs,
        commit_ctx=None,
    ):
        if bool(getattr(inputs, "is_fake_stream", False)):
            return
        bridge = self._bridge(inputs)
        req, pos = context_req_ids, context_positions
        group = None
        if commit_ctx is not None:
            group = DistributedCPGroup(self.runner.group)
            req, pos = commit_ctx.full_request_ids.to(
                self.device
            ), commit_ctx.full_positions.to(self.device)
        observer, observation = getattr(self.runner, "observer", None), None
        if observer is not None and not bool(getattr(inputs, "warmup", False)):
            primary = primary_attention_inputs(inputs.attention_inputs, self.kv_cache)
            prefixes = (
                [split.prefix for split in commit_ctx.requests]
                if commit_ctx is not None
                else primary.prefix_lengths.detach().cpu().tolist()
            )
            ends = committed_ends.detach().cpu().tolist()
            observed_positions = [
                pos[
                    (req == row)
                    & (pos >= max(0, end - self.geometry.window))
                    & (pos < end)
                ]
                .detach()
                .cpu()
                .tolist()
                for row, end in enumerate(ends)
            ]
            observation = observer.begin(
                "commit",
                bridge,
                observer.input_request_ids(inputs, len(ends)),
                prefixes,
                [end - prefix for end, prefix in zip(ends, prefixes)],
                positions=observed_positions,
            )
        for layer_id, layer in self.bundle.layers.items():
            # Each layer consumes the SAME target projection. Query hidden is
            # never fed into another layer's committed feature KV.
            kv = _rms_norm(layer.wkv(main_x), layer.kv_norm, self.geometry.norm_eps)
            kv = _rope_tail(
                kv,
                self.bundle.freqs_for(layer_id)[context_positions.long().clamp_min(0)],
            )
            if group is not None:
                kv = group.all_gather(kv).index_select(
                    0, commit_ctx.unpad_restore.to(self.device)
                )
            from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv

            packed = encode_kv(kv, pool_tag="swa_kv")
            for request, end in enumerate(committed_ends.cpu().tolist()):
                valid = (
                    (req == request)
                    & (pos >= max(0, end - self.geometry.window))
                    & (pos < end)
                )
                bridge._copy_rows(
                    (layer_id, "swa_kv"), request, pos[valid], packed[valid]
                )
                if observer is not None:
                    observer.add_payload(
                        observation,
                        (layer_id, "swa_kv"),
                        request,
                        pos[valid],
                        packed[valid],
                    )
        # On P the native writers publish draft pools only after all layers
        # have committed. Proposal scratch never reaches CacheStore.
        if self.commit_only and not bool(inputs.warmup):
            bridge.publish()
        if observer is not None and observation is not None:
            observer.finish(observation, "accept", observation["lengths"])
        logger.info(
            "PPU DSpark COMMIT rows=%d ends=%s cp=%d/%d",
            main_x.shape[0],
            committed_ends.cpu().tolist(),
            self.tp_rank,
            self.runner.cp_size,
        )

    def _attention(self, layer_id, layer, x, positions, prefixes, active, bridge):
        g = self.geometry
        rows, width = x.shape[0], self._dspark_query_width
        if rows == 0 or not bool(active.any()):
            return torch.zeros_like(x)
        freqs = self.bundle.freqs_for(layer_id)[positions.reshape(-1).long()]
        qr = _rms_norm(layer.wq_a(x), layer.q_norm, g.norm_eps)
        q = _rope_tail(layer.wq_b(qr).reshape(rows, g.n_heads, g.head_dim), freqs)
        kv = _rope_tail(_rms_norm(layer.wkv(x), layer.kv_norm, g.norm_eps), freqs)
        from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv

        # Query KVs use the same packed codec as committed KVs, but live in
        # scratch so a query can never overwrite the committed SWA ring.
        kv = decode_kv(
            encode_kv(kv, pool_tag="swa_kv"),
            pool_tag="swa_kv",
            out=torch.empty_like(kv),
        )
        workspace, index_rows, counts = [], [], []
        offset = 0
        index_width = ((g.window + width + 127) // 128) * 128
        for request, prefix in enumerate(prefixes.cpu().tolist()):
            live = bool(active[request])
            start = max(0, prefix - g.window)
            past_pos = torch.arange(start, prefix, device=self.device)
            packed = (
                bridge._copy_rows((layer_id, "swa_kv"), request, past_pos)
                if live
                else kv.new_empty((0, 528), dtype=torch.uint8)
            )
            past = decode_kv(
                packed,
                pool_tag="swa_kv",
                out=kv.new_empty((packed.shape[0], g.head_dim)),
            )
            block = (
                torch.cat((past, kv[request * width : (request + 1) * width]))
                if live
                else kv.new_empty((0, g.head_dim))
            )
            workspace.append(block)
            indices = torch.full(
                (width, 1, index_width), -1, dtype=torch.int32, device=self.device
            )
            count = block.shape[0]
            if count:
                indices[:, 0, :count] = torch.arange(
                    offset, offset + count, dtype=torch.int32, device=self.device
                )
            index_rows.append(indices)
            counts.extend([count] * width)
            offset += count
        workspace = torch.cat(workspace).reshape(-1, 1, g.head_dim)
        indices = torch.cat(index_rows)
        counts = torch.tensor(counts, device=self.device, dtype=torch.int32)
        if self.device.type == "cpu":
            out = _torch_sparse_attention(
                q, workspace, indices, counts, g.sm_scale(), layer.attn_sink
            )
        else:
            out, _, _ = flash_mla_sparse_fwd_ppu(
                q.contiguous(),
                workspace,
                indices,
                g.sm_scale(),
                attn_sink=layer.attn_sink,
                topk_length=counts,
            )
        out = _rope_tail(out, freqs, inverse=True).reshape(rows, g.o_groups, -1)
        projected = torch.cat(
            [
                torch.nn.functional.linear(
                    out[:, group].float(),
                    layer.wo_a.weight.reshape(g.o_groups, g.o_lora_rank, -1)[
                        group
                    ].float(),
                ).to(x.dtype)
                for group in range(g.o_groups)
            ],
            dim=-1,
        )
        return layer.wo_b(projected)

    def forward_query_block(
        self,
        query_ids,
        query_positions,
        prefix_lengths,
        active_requests,
        inputs,
        fmha_impl,
    ):
        bridge = self._bridge(inputs)
        g = self.geometry
        active = active_requests & (not bool(inputs.is_fake_stream))
        observer, observation = getattr(self.runner, "observer", None), None
        if (
            observer is not None
            and not bool(inputs.is_fake_stream)
            and not bool(getattr(inputs, "warmup", False))
        ):
            prefixes = prefix_lengths.detach().cpu().tolist()
            observation = observer.begin(
                "proposal",
                bridge,
                observer.input_request_ids(inputs, len(prefixes)),
                prefixes,
                [self._dspark_query_width] * len(prefixes),
            )
        valid = active.repeat_interleave(self._dspark_query_width)
        h = (
            self.bundle.embedding[query_ids.long().reshape(-1)]
            .unsqueeze(1)
            .expand(-1, g.hc_mult, -1)
            .contiguous()
        )
        previous = None
        for layer_id, layer in self.bundle.layers.items():
            step = self.ops.mhc(
                h,
                None,
                phase="pre",
                parameters=layer.mhc_parameters("attn"),
                state=previous,
            )
            x = _rms_norm(step.hidden, layer.attn_norm, g.norm_eps)
            branch = self._attention(
                layer_id, layer, x, query_positions, prefix_lengths, active, bridge
            )
            h = self.ops.mhc(h, branch, phase="post", state=step).hidden
            ffn = self.ops.mhc(
                h, None, phase="pre", parameters=layer.mhc_parameters("ffn"), state=step
            )
            x = _rms_norm(ffn.hidden, layer.ffn_norm, g.norm_eps)
            ids, weights = sqrtsoftplus_gate(
                x,
                layer.router_weight,
                layer.router_bias,
                top_k=g.n_activated_experts,
                routed_scaling_factor=g.routed_scaling_factor,
            )
            ids, weights = pad_topk_for_dispatch(ids, weights)
            ids, weights = ids.masked_fill(~valid[:, None], -1), weights.masked_fill(
                ~valid[:, None], 0
            )
            branch = self.ops.moe(x, ids, weights, layer_id=layer_id)
            if layer.shared_w13 is not None:
                branch = combine_routed_and_shared(
                    branch,
                    shared_expert_forward(
                        x, layer.shared_w13, layer.shared_w2, g.swiglu_limit
                    ),
                )
            h = self.ops.mhc(h, branch, phase="post", state=ffn).hidden
            previous = ffn
        output = self.ops.mhc(h, None, phase="head", state=previous).hidden
        if observer is not None:
            observer.verified(observation)
            observer.finish(observation, "proposal", [0] * len(prefix_lengths))
        return output

    def compute_draft_outputs(self, hidden):
        normalized = _rms_norm(hidden, self.bundle.final_norm, self.geometry.norm_eps)
        confidence = self.weights.confidence_head
        if confidence is None:
            return PyModelOutputs(normalized)
        logits = torch.nn.functional.linear(
            hidden.float(), confidence[:, : self.geometry.hidden_size]
        ).contiguous()
        outputs = PyModelOutputs(normalized)
        outputs.dspark_confidence_hidden_logits = logits
        return outputs

    def dspark_empty_outputs(self, batch_size, device):
        return self.compute_draft_outputs(
            torch.zeros(
                (batch_size * self._dspark_query_width, self.geometry.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
        )

    @torch.inference_mode()
    def forward_commit(self, inputs, fmha_impl=None):
        if self.kv_cache is None:
            return PyModelOutputs(
                self.bundle.embedding.new_empty((0, self.geometry.hidden_size))
            )
        return self.run_commit_step(inputs, self.device)

    @torch.inference_mode()
    def forward_propose(self, inputs, fmha_impl=None):
        if self.commit_only:
            raise RuntimeError("P's DSpark model is commit-only")
        if self.kv_cache is None:
            return self.dspark_empty_outputs(
                max(1, inputs.input_ids.numel() // self._dspark_query_width),
                self.device,
            )
        if self.runner.observer is not None:
            self.runner.observer.warmup = bool(getattr(inputs, "warmup", False))
        if not self.runner._work_available(
            0 if bool(inputs.is_fake_stream) else inputs.input_ids.numel()
        ):
            return self.dspark_empty_outputs(
                inputs.input_ids.numel() // self._dspark_query_width, self.device
            )
        output = self.run_propose_step(inputs, fmha_impl, self.device)
        logger.info(
            "PPU DSpark PROPOSE dp=%d rows=%d gamma=%d query_width=%d noncausal=true native_writes=0 fake=%s",
            self.runner.dp_rank,
            inputs.input_ids.numel(),
            self._dspark_width,
            self._dspark_query_width,
            bool(inputs.is_fake_stream),
        )
        return output

    def forward(self, inputs, fmha_impl=None):
        raise RuntimeError("DSpark requires explicit forward_commit or forward_propose")
