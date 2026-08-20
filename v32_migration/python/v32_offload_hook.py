"""v32_offload_hook.py — wraps DSV32 MLA attention forward for shadow-mode offload.
Deploy: copy this + v32_offload.py to runtime site-packages/rtp_llm/, then in
mla_attention.py add at module end:  import rtp_llm.v32_offload_hook  # noqa
Env: V32_OFFLOAD_MODE=shadow V32_OFFLOAD_VERIFY=1
"""

import logging

import torch

from rtp_llm import v32_offload as vo
from rtp_llm.models_py.modules.hybrid import mla_attention as _mla

_cls = getattr(_mla, "MlaAttention", None) or getattr(_mla, "MLAAttention", None)
if _cls is None:  # fallback: find the class defining _run_sparse_indexer
    for _name in dir(_mla):
        _c = getattr(_mla, _name)
        if isinstance(_c, type) and "_run_sparse_indexer" in vars(_c):
            _cls = _c
            break

_orig_indexer = _cls._run_sparse_indexer


def _hooked_indexer(self, hidden_states, q_c, q_view, kv_cache, fmha_impl):
    topk = _orig_indexer(self, hidden_states, q_c, q_view, kv_cache, fmha_impl)
    try:
        if vo.MODE == "shadow" and kv_cache is not None and topk is not None:
            ai = getattr(fmha_impl, "attn_inputs", None)
            bt = getattr(ai, "kv_cache_block_id", None) if ai is not None else None
            # only decode-shaped batches: one topk row per request row.
            # prefill has num_tokens rows vs batch block-table rows -> skip.
            if bt is not None and topk.shape[0] == bt.shape[0]:
                for i in range(bt.shape[0]):
                    row = bt[i]
                    used = row[row > 0]
                    vo.mirror_blocks(kv_cache, used.tolist())
                torch.cuda.current_stream().synchronize()
                for i in range(bt.shape[0]):
                    rows = vo.gather_from_host(kv_cache, topk[i], bt[i])
                    vo.verify_against_gpu(kv_cache, topk[i], rows, bt[i])
    except Exception:
        logging.exception("[v32_offload_hook] shadow path error (forward unaffected)")
    return topk


try:
    _cls._run_sparse_indexer = _hooked_indexer
    logging.warning(
        f"[v32_offload_hook] installed on {_cls.__name__}, mode={vo.MODE} verify={vo.VERIFY}"
    )
except Exception:
    logging.exception("[v32_offload_hook] install failed; running vanilla")


# ---- P-B1: verify python-recomputed indexer top-k vs kernel (V32_VERIFY_TOPK=1)
import os as _os


def _os_env_install():
    return _os.environ.get("V32_INSTALL", "1") != "0"


# ---- hit-ratio shadow measurement (V32_HITRATIO=1): window / window+LRU
# coverage of the native top-2048 selections. Observation only.
if _os.environ.get("V32_HITRATIO", "0") == "1":
    from rtp_llm import v32_hitratio as _vh
    from rtp_llm.models_py.modules.base.cuda import indexer_op as _iop_h

    _orig_tp_h = _iop_h.IndexerOp._get_topk_paged

    def _hitratio_topk(self, q_fp8, weights, kv_cache, fmha_params, attention_inputs):
        out = _orig_tp_h(self, q_fp8, weights, kv_cache, fmha_params, attention_inputs)
        try:
            _vh.observe(self, kv_cache, fmha_params, attention_inputs, out)
        except Exception:
            logging.exception("[v32_hitratio] observe error (forward unaffected)")
        return out

    _iop_h.IndexerOp._get_topk_paged = _hitratio_topk
    logging.warning("[v32_offload_hook] HITRATIO measurement installed")


if _os.environ.get("V32_VERIFY_TOPK", "0") == "1":
    from rtp_llm.models_py.modules.base.cuda import indexer_op as _iop

    _orig_topk_paged = _iop.IndexerOp._get_topk_paged
    _orig_quant_qk = _iop.IndexerOp.quant_q_k
    _tk_stats = {"calls": 0, "reqs": 0, "min_norelu": 1.0, "min_relu": 1.0}
    _last_key = {}

    def _hooked_quant_qk(self, query, key, kv_cache, slot_mapping):
        _last_key["key"] = key.detach()
        _last_key["slots"] = slot_mapping
        return _orig_quant_qk(self, query, key, kv_cache, slot_mapping)

    _iop.IndexerOp.quant_q_k = _hooked_quant_qk

    def _py_topk_compare(
        self, q_fp8, weights, kv_cache, fmha_params, attention_inputs, kernel_topk
    ):
        bs = self.blocksize
        kbt = attention_inputs.kv_cache_kernel_block_id_device
        kvlen_d = fmha_params.kvlen_d
        pool = kv_cache.kv_scale_base
        cache = pool.view(pool.shape[0], bs, -1).view(torch.uint8)  # [nb_pool, 64, 132]
        w = weights.view(-1, self.index_n_heads).float()
        # layout self-check: dequant current token's cached k vs captured bf16 key
        try:
            kk, ss = _last_key.get("key"), _last_key.get("slots")
            if kk is not None and ss is not None and kk.shape[0] == kbt.shape[0]:
                slot = int(ss.reshape(-1)[0])
                i0 = 0
                kvlen0 = int(kvlen_d[i0])
                last_blk_kernel = int(kbt[i0, (kvlen0 - 1) // bs])
                blk = slot // bs
                raw_blk = cache[blk].reshape(-1)
                nz_direct = int((raw_blk != 0).sum())
                raw_kblk = (
                    cache[last_blk_kernel].reshape(-1)
                    if last_blk_kernel < cache.shape[0]
                    else raw_blk * 0
                )
                nz_kernel_blk = int((raw_kblk != 0).sum())
                raw = cache[blk, slot % bs]
                kf = raw[:128].view(torch.float8_e4m3fn).float()
                s0 = _iop._unpack_ue8m0_scale(
                    raw[128:132].contiguous().view(torch.int32)
                ).item()
                rec = kf * s0
                ref = kk[0].reshape(-1)[:128].float()
                cos = torch.nn.functional.cosine_similarity(rec, ref, dim=0).item()
                # segregated layout: fp8 [0,64*128), float32 scales [64*128,...)
                off = slot % bs
                kf2 = (
                    raw_blk[off * 128 : (off + 1) * 128]
                    .view(torch.float8_e4m3fn)
                    .float()
                )
                s2 = (
                    raw_blk[bs * 128 + off * 4 : bs * 128 + off * 4 + 4]
                    .contiguous()
                    .view(torch.float32)
                    .item()
                )
                cos2 = torch.nn.functional.cosine_similarity(
                    kf2 * s2, ref, dim=0
                ).item()
                logging.warning(
                    f"[v32_topk] layout pool.shape={tuple(pool.shape)} dtype={pool.dtype} "
                    f"slot={slot} blk={blk} last_blk_kernel={last_blk_kernel} "
                    f"nz_direct={nz_direct} nz_kernel_blk={nz_kernel_blk} "
                    f"cos_interleaved={cos:.4f} cos_segregated={cos2:.4f} ref_norm={ref.norm().item():.3f}"
                )
        except Exception:
            logging.exception("[v32_topk] layout check error")
        for i in range(kbt.shape[0]):
            kvlen = int(kvlen_d[i])
            if kvlen < 256:
                continue
            nb = (kvlen + bs - 1) // bs
            blocks = kbt[i, :nb].long()
            # segregated block layout (mla_quant_kernel.cu): per block flat bytes =
            # [bs*128 fp8][bs*4 float32 scales]
            blk_flat = cache[blocks].reshape(nb, -1)  # [nb, 8448]
            kfp8 = (
                blk_flat[:, : bs * 128]
                .view(torch.float8_e4m3fn)
                .reshape(nb, bs, 128)
                .float()
            )
            sf = (
                blk_flat[:, bs * 128 : bs * 128 + bs * 4]
                .contiguous()
                .view(torch.float32)
                .reshape(nb, bs, 1)
            )
            k = (kfp8 * sf).reshape(-1, 128)[:kvlen]
            q = q_fp8[i].float()  # [heads, 128]
            logits_h = q @ k.t()  # [heads, kvlen]
            n = min(self.index_topk, kvlen)
            kern = kernel_topk[i].reshape(-1)
            kern = kern[kern >= 0]
            kmax, kmin = int(kern.max()), int(kern.min())
            n_over = int((kern >= kvlen).sum())
            kern = set(kern.tolist())
            if not kern:
                continue

            # physical slot ids for py logical positions (block-table translate)
            def to_phys(pos_tensor):
                pos_tensor = pos_tensor.to(blocks.device)
                return kbt[i, (pos_tensor // bs).long()].long() * bs + pos_tensor % bs

            for tag, sc in (
                ("norelu", (w[i].view(-1, 1) * logits_h).sum(0)),
                ("relu", (w[i].view(-1, 1) * torch.relu(logits_h)).sum(0)),
            ):
                py_pos = torch.topk(sc, n).indices
                py = set(py_pos.tolist())
                ov_log = len(py & kern) / max(len(kern), 1)
                py_phys = set(to_phys(py_pos).tolist())
                ov_phys = len(py_phys & kern) / max(len(kern), 1)
                key = "min_" + tag
                _tk_stats[key] = min(_tk_stats[key], max(ov_log, ov_phys))
                if _tk_stats["reqs"] % 20 == 0:
                    # percentile of kernel-picked positions under my score
                    kern_t = torch.tensor(
                        sorted(kern), device=sc.device, dtype=torch.long
                    )
                    kern_t = kern_t[kern_t < sc.shape[0]]
                    med_kern = sc[kern_t].median()
                    pct = (sc < med_kern).float().mean().item()
                    logging.warning(
                        f"[v32_topk] req kvlen={kvlen} n={n} {tag} ov_logical={ov_log:.4f} "
                        f"ov_physical={ov_phys:.4f} kern_min={kmin} kern_max={kmax} "
                        f"kern_over_kvlen={n_over} kern_med_pct={pct:.3f}"
                    )
            _tk_stats["reqs"] += 1

    def _hooked_topk_paged(
        self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
    ):
        out = _orig_topk_paged(
            self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )
        try:
            _tk_stats["calls"] += 1
            if _tk_stats["calls"] % 61 == 1:  # ~once per model step
                _py_topk_compare(
                    self, q_fp8, weights, kv_cache, fmha_params, attention_inputs, out
                )
        except Exception:
            logging.exception("[v32_topk] compare error (forward unaffected)")
        return out

    _iop.IndexerOp._get_topk_paged = _hooked_topk_paged
    logging.warning(
        "[v32_offload_hook] topk verify installed on IndexerOp._get_topk_paged"
    )


# ---- P-B1: verify python-recomputed sparse MLA attention vs kernel (V32_VERIFY_ATTN=1)
if _os.environ.get("V32_VERIFY_ATTN", "0") == "1":
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
        flashmla_sparse_impl as _fsi,
    )

    _orig_sparse_fwd = _fsi.SparseMlaOp.forward
    _at_stats = {"calls": 0, "checks": 0, "min_cos": 1.0}

    def _py_sparse_attn(self, q, kv, topk_indices, out_kernel):
        gidx = self._convert_topk_indices_to_global(topk_indices)
        if gidx.dim() == 3 and gidx.shape[1] == 1:
            gidx = gidx.squeeze(1)  # [T, topk]
        pool = kv.reshape(-1, kv.shape[-1])  # [nb*bs, 576]
        row = 0
        idx = gidx[row].long()
        idx = idx[(idx >= 0) & (idx < pool.shape[0])]
        if idx.numel() < 8:
            return
        rows = pool[idx].float()  # [k, 576]
        dv = self.kv_lora_rank
        qn = q[row, :, :dv].float()
        qr = q[row, :, dv:].float()
        scores = (qn @ rows[:, :dv].t() + qr @ rows[:, dv:].t()) * self.scale
        p = torch.softmax(scores, dim=-1)
        out_py = p @ rows[:, :dv]  # [H, dv]
        ok = out_kernel[row].float()
        cos = torch.nn.functional.cosine_similarity(
            out_py.reshape(-1), ok.reshape(-1), dim=0
        ).item()
        _at_stats["checks"] += 1
        _at_stats["min_cos"] = min(_at_stats["min_cos"], cos)
        if _at_stats["checks"] % 20 == 1 or cos < 0.99:
            logging.warning(
                f"[v32_attn] row0 k={idx.numel()} cos={cos:.5f} "
                f"min={_at_stats['min_cos']:.5f} checks={_at_stats['checks']} "
                f"scale={self.scale:.6f} qdim={q.shape[-1]} dv={dv}"
            )

    def _hooked_sparse_fwd(self, q, kv, topk_indices, kv_scale=None, layer_id=0):
        out = _orig_sparse_fwd(self, q, kv, topk_indices, kv_scale, layer_id)
        try:
            _at_stats["calls"] += 1
            if (
                _at_stats["calls"] % 61 == 1
                and kv is not None
                and kv.dtype == torch.bfloat16
            ):
                _py_sparse_attn(self, q, kv, topk_indices, out)
        except Exception:
            logging.exception("[v32_attn] compare error (forward unaffected)")
        return out

    _fsi.SparseMlaOp.forward = _hooked_sparse_fwd
    logging.warning("[v32_offload_hook] attn verify installed on SparseMlaOp.forward")


# ---- capacity mode wiring (V32_OFFLOAD_MODE=capacity; do NOT combine with
# V32_VERIFY_TOPK/ATTN). Requires the C++ offloadPrefixBlocks runtime.
if vo.MODE == "capacity" and _os_env_install():
    from rtp_llm import v32_capacity as vc
    from rtp_llm.models_py.modules.base.cuda import indexer_op as _iop_c

    _MIN_SEQ = int(_os.environ.get("RTP_KV_OFFLOAD_MIN_SEQ", "16384"))
    _orig_tp_c = _iop_c.IndexerOp._get_topk_paged

    import time as _t

    _hb = {"orig": 0.0, "proc": 0.0, "fwd": 0.0, "n": 0}
    _tp = {"prof": None, "state": 0}

    def _prof_tick():
        # env V32_TORCH_PROF=1: profile steps [3050*1+0 .. +3*61] once, dump top kernels
        if _os.environ.get("V32_TORCH_PROF", "0") != "1" or _tp["state"] > 1:
            return
        n = _hb["n"]
        if _tp["state"] == 0 and n >= 3050:
            import torch.profiler as tpr

            _tp["prof"] = tpr.profile(
                activities=[tpr.ProfilerActivity.CUDA], record_shapes=False
            )
            _tp["prof"].__enter__()
            _tp["state"] = 1
            return
        if _tp["state"] == 1 and n >= 3050 + 3 * 61:
            _tp["prof"].__exit__(None, None, None)
            tbl = (
                _tp["prof"]
                .key_averages()
                .table(sort_by="cuda_time_total", row_limit=18)
            )
            for line in tbl.splitlines():
                logging.warning("[v32_prof] " + line)
            _tp["state"] = 2

    _SANITIZE_KVLEN = _os.environ.get("V32_SANITIZE_KVLEN", "0") == "1"

    _HOOK_LEVEL = int(_os.environ.get("V32_HOOK_LEVEL", "2"))

    def _cap_topk(self, q_fp8, weights, kv_cache, fmha_params, attention_inputs):
        if _HOOK_LEVEL == 0:  # pure passthrough
            return _orig_tp_c(
                self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if _HOOK_LEVEL == 1:  # + attribute access only
            out = _orig_tp_c(
                self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
            _kbt = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
            _kvl = fmha_params.kvlen_d
            return out
        _prof_tick()
        t0 = _t.perf_counter()
        out = None
        try:
            out = vc.pre_topk(
                self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        except Exception:
            logging.exception("[v32_capacity] pre_topk error — native wave")
            out = None
        if out is None:
            saved = rows = None
            if _SANITIZE_KVLEN:
                try:
                    rows = vc.offloaded_rows_hint(None)
                    if rows is not None and rows.numel():
                        kvl = fmha_params.kvlen_d
                        saved = kvl.index_select(0, rows)
                        kvl.index_fill_(
                            0, rows, 1
                        )  # native indexer skips the dead scan
                except Exception:
                    saved = rows = None
            out = _orig_tp_c(
                self, q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
            if saved is not None:
                fmha_params.kvlen_d.index_copy_(0, rows, saved)
        _hb["orig"] += _t.perf_counter() - t0
        try:
            kbt = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
            if out is not None and kbt is not None and out.shape[0] == kbt.shape[0]:
                t0 = _t.perf_counter()
                vc.process_layer(
                    self,
                    q_fp8,
                    weights,
                    kv_cache,
                    kbt,
                    fmha_params.kvlen_d,
                    out,
                    _MIN_SEQ,
                )
                _hb["proc"] += _t.perf_counter() - t0
                _hb["n"] += 1
                if _hb["n"] % 3050 == 0:
                    logging.warning(
                        f"[v32_hb] hook-boundary(s)={ {k: round(v,2) for k,v in _hb.items()} }"
                    )
        except Exception:
            logging.exception("[v32_capacity] indexer hook error")
        return out

    # C++ writes served indices back into kernel_topk in-place; the native
    # convert-to-global feeds attention, so SparseMlaOp.forward stays vanilla.
    _iop_c.IndexerOp._get_topk_paged = _cap_topk
    logging.warning(
        f"[v32_offload_hook] CAPACITY mode installed (topk-only), min_seq={_MIN_SEQ}"
    )

    # staging-ring admission: bind the engine's admission mirror exports so
    # v32_capacity can adopt engine-produced host/idxp buffers at first serve.
    try:
        import os.path as _osp

        import rtp_llm as _rtp

        _so = _osp.join(_osp.dirname(_rtp.__file__), "libs", "libth_transformer.so")
        if vc._ctx is not None and hasattr(vc._ctx, "ctx_admission_open"):
            _ok = vc._ctx.ctx_admission_open(_so)
            logging.warning(
                f"[v32_offload_hook] admission adoption channel open={_ok} ({_so})"
            )
    except Exception:
        logging.exception(
            "[v32_offload_hook] admission adoption open failed (ring admission disabled)"
        )
