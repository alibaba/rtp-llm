"""Fixed L1 replay only: digest real boundaries; shadow kernels never feed the service."""

import hashlib
import importlib
import threading

import torch


def _id(obj):
    return (
        getattr(obj, "__module__", type(obj).__module__)
        + "."
        + getattr(obj, "__qualname__", type(obj).__qualname__)
    )


def _geometry(x):
    return {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "offset": x.storage_offset(),
        "alignment_mod_256": x.data_ptr() % 256,
    }


def rms_references(x, w, eps):
    """Plain IEEE math, not a claim about a vendor kernel's exact reduction tree."""
    a = x.float()
    wt = w.float()
    mean = a.square().mean(-1, keepdim=True)
    inv = (mean + eps).rsqrt()
    normalized = a * inv
    precise = (
        x.double() * (x.double().square().mean(-1, keepdim=True) + eps).rsqrt()
    ) * w.double()
    return {
        "ref.mean2": mean,
        "ref.inv_rms": inv,
        "ref.normalized32": normalized,
        "ref.weighted32": normalized * wt,
        "ref.f32_once": (normalized * wt).to(x.dtype),
        "ref.bf16_before_weight": (normalized.to(x.dtype).float() * wt).to(x.dtype),
        "ref.weight_first": ((a * wt) * inv).to(x.dtype),
        "ref.f64_once": precise.to(x.dtype),
    }, precise


def support40(x, precise, index):
    """Signed BF16 ULP deviation relative to a matching high precision reference.

    No raw hidden elements / one-element hashes are exported. If reference
    row hashes agree, two sparse deviation lists suffice for an exact per-
    channel comparison including zero sign bits. No tolerance is applied.
    """
    live = x[index].reshape(-1).detach().cpu()
    real = precise[index].reshape(-1).detach().cpu()
    nearest = real.bfloat16()
    if (
        live.dtype != torch.bfloat16
        or not torch.isfinite(live).all()
        or not torch.isfinite(real).all()
    ):
        return {"valid": False, "reason": "nonfinite_or_non_bf16"}

    def order(t):
        u = t.contiguous().view(torch.int16).int() & 65535
        return torch.where(u < 32768, 32768 + u, 65535 - u)

    diff = order(live) - order(nearest)
    ix = diff.nonzero().flatten()
    # Count exact and near half-way cases relative to the neighboring BF16 in
    # the direction of the FP64 value, without publishing the hidden values.
    near = nearest.double()
    direction = torch.where(
        real >= near,
        torch.full_like(nearest, float("inf")),
        torch.full_like(nearest, -float("inf")),
    )
    adjacent = torch.nextafter(nearest, direction).double()
    ulp = (adjacent - near).abs()
    mid = (real - (near + adjacent) * 0.5).abs() / ulp
    return {
        "valid": True,
        "channels": live.numel(),
        "delta": [[int(i), int(diff[i])] for i in ix],
        "ties": int((mid == 0).sum()),
        "within_1e_4_ulp_of_tie": int((mid < 1e-4).sum()),
    }


class NormProbe:
    def __init__(self, trace):
        self.trace = trace
        self.saved = {}
        self.docs = {}
        self.meta = {}
        self.errors = {}
        self.ulp = {}
        self.path = {"hc": [], "rms": []}
        self.in_hc = self.in_norm = False
        self.thread = threading.get_ident()

    def active(self, flag):
        return getattr(self, flag) and threading.get_ident() == self.thread

    def save(self, k, x):
        if k in self.saved:
            raise ValueError("duplicate V4 L1 witness " + k)
        self.meta[k] = _geometry(x)
        self.saved[k] = x.detach().clone(memory_format=torch.preserve_format)

    def record(self, k, x):
        if x.ndim >= 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        z = x.detach().index_select(0, self.trace.rows).contiguous()
        rows = z.view(torch.uint8).reshape(len(self.trace.pos), -1).cpu().numpy()
        self.docs[k] = {
            "pos": self.trace.pos,
            "shape": list(z.shape[1:]),
            "dtype": str(z.dtype),
            "rows": [hashlib.sha256(b).hexdigest() for b in rows],
        }

    def hc_call(self, fn, args, kw):
        self.save("live.hc_residual", args[0])
        self.meta["hc_callable"] = _id(fn)
        self.in_hc = True
        try:
            val = fn(*args, **kw)
        finally:
            self.in_hc = False
        for k, x in zip(("mix", "post", "comb"), val):
            self.save("live.hc_" + k, x)
        return val

    def hooks(self, layer, replace):
        module = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
        )
        big = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )

        def spy(obj, name, flag, category, after=None):
            fn = getattr(obj, name)

            def wrapped(*args, **kw):
                active = self.active(flag)
                if active:
                    self.path[category].append(name)
                res = fn(*args, **kw)
                if active and after:
                    after(args, kw, res)
                return res

            replace(obj, name, wrapped)

        for n in (
            "mhc_pre_big_fuse",
            "mhc_pre_norm_fn",
            "mhc_pre_split_mixes",
            "sinkhorn_normalize",
            "mhc_pre_apply_mix",
        ):
            spy(module, n, "in_hc", "hc")

        def capture_split(a, kw, res):
            self.meta["live_n_splits"] = int(a[4])
            self.save("live.hc_splitk_dot", a[2].transpose(0, 1))
            self.save("live.hc_splitk_sq", a[3].transpose(0, 1))

        spy(big, "_run_deepgemm_splitk_gemm", "in_hc", "hc", capture_split)
        import flashinfer.norm as fin

        self.meta["flashinfer_cuda_dispatch"] = getattr(fin, "_USE_CUDA_NORM", None)
        props = torch.cuda.get_device_properties(layer.ffn_norm.weight.device)
        self.meta["rms_gpu"] = {
            "sm": [props.major, props.minor],
            "multiprocessors": props.multi_processor_count,
            "block_shared_mem_optin": props.shared_memory_per_block_optin,
            "cuda_toolkit": torch.version.cuda,
        }

        def native(a, kw, out):
            self.meta["flashinfer_pdl_argument"] = kw.get("enable_pdl", "default")
            self.meta["flashinfer_output_alias"] = a[0].data_ptr() == out.data_ptr()

        spy(fin, "rmsnorm", "in_norm", "rms", native)
        if callable(getattr(layer.ffn_norm, "_implementation", None)):
            self.meta["configured_rms"] = _id(layer.ffn_norm._implementation)
            spy(layer.ffn_norm, "_implementation", "in_norm", "rms", native)
        orig = layer.ffn_norm.forward
        self.meta["norm_callable"] = _id(orig)

        def norm(*args, **kw):
            if threading.get_ident() != self.thread:
                return orig(*args, **kw)
            x = args[0] if args else kw["hidden_states"]
            self.save("live.norm_input", x)
            self.in_norm = True
            try:
                out = orig(*args, **kw)
            finally:
                self.in_norm = False
            self.meta["norm_inplace"] = x.data_ptr() == out.data_ptr()
            self.save("live.norm_output", out)
            return out

        replace(layer.ffn_norm, "forward", norm)

    def finish(self, layer):
        for k, x in self.saved.items():
            self.record(k, x)

        def attempt(k, fn):
            try:
                return fn()
            except (
                AttributeError,
                ImportError,
                TypeError,
                ValueError,
                AssertionError,
                NotImplementedError,
                RuntimeError,
            ) as e:
                # Failed probes are absence of evidence, never model parity.
                if isinstance(e, RuntimeError) and any(
                    s in str(e).lower()
                    for s in (
                        "illegal memory",
                        "device-side assert",
                        "out of memory",
                        "misaligned address",
                    )
                ):
                    raise
                self.errors[k] = type(e).__name__
                return None

        def hc():
            unit = layer.ffn_hc
            f = importlib.import_module(
                "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
            )
            x = self.saved["live.hc_residual"].reshape(1, -1, 4, 4096)
            opts = {
                "norm_eps": float(unit.norm_eps),
                "pre_eps": float(unit.hc_eps),
                "sinkhorn_eps": float(unit.hc_eps),
                "sinkhorn_repeat": int(unit.hc_sinkhorn_iters),
                "mhc_mult": 4,
                "post_mult_value": 2.0,
            }

            def fused():
                y, (p, c) = f.mhc_pre(x.clone(), unit.fn, unit.scale, unit.base, **opts)
                for k, v in (("mix", y), ("post", p), ("comb", c)):
                    self.record("fused." + k, v)
                return y.reshape(-1, 4096)

            def from_splitk():
                # FP32 reconstruction from live native intermediates. Match
                # TileLang's serial split-K accumulation; PyTorch sigmoid /
                # rsqrt are references, not the kernel's local registers.
                dots = self.saved["live.hc_splitk_dot"]
                sq = self.saved["live.hc_splitk_sq"]
                acc = torch.zeros_like(dots[:, 0])
                ss = torch.zeros_like(sq[:, 0])
                for j in range(sq.shape[1]):
                    acc = acc + dots[:, j]
                    ss = ss + sq[:, j]
                f32 = acc * (ss[:, None] / (4 * 4096) + opts["norm_eps"]).rsqrt()
                pre = (f32[:, :4] * unit.scale[0] + unit.base[:4]).sigmoid() + opts[
                    "pre_eps"
                ]
                v = x.reshape(-1, 4, 4096).float()
                # addcmul permits FMA (as in the native serial multiply-add).
                y = torch.zeros_like(v[:, 0])
                u = torch.zeros_like(y)
                for j in range(4):
                    y = torch.addcmul(y, pre[:, j, None], v[:, j])
                    u = u + pre[:, j, None] * v[:, j]
                for k, z in (
                    ("mix_logits", f32),
                    ("pre", pre),
                    ("sum_fma32", y),
                    ("sum_nonfma32", u),
                    ("fma_bf16", y.bfloat16()),
                    ("nonfma_bf16", u.bfloat16()),
                ):
                    self.record("reconstructed." + k, z)

            def split():
                r = x.clone()
                m = f.mhc_pre_norm_fn(
                    r, unit.fn, None, opts["norm_eps"], fuse_grad_acc=False, n_splits=16
                )
                pre, p, c = f.mhc_pre_split_mixes(
                    m, unit.scale, unit.base, 4, 2.0, opts["pre_eps"]
                )
                c = f.sinkhorn_normalize(
                    c, repeat=opts["sinkhorn_repeat"], eps=opts["sinkhorn_eps"]
                )
                y = f.mhc_pre_apply_mix(r, pre)
                weighted = pre.float() * r.float()
                self.record("split.dot_norm", m)
                self.record("split.pre", pre)
                self.record("split.sum32", weighted.sum(-2))
                for k, v in (("mix", y), ("post", p), ("comb", c)):
                    self.record("split." + k, v)
                return y.reshape(-1, 4096)

            if "live.hc_splitk_dot" in self.saved:
                attempt("hc.from_native_splitk", from_splitk)
            return attempt("hc.fused", fused), attempt("hc.split", split)

        with torch.inference_mode():
            hs = attempt("hc", hc)
            if "live.norm_input" in self.saved:
                x = self.saved["live.norm_input"].reshape(-1, 4096)
                w = layer.ffn_norm.weight.detach()
                eps = float(layer.ffn_norm.variance_epsilon)
                self.meta["rms_eps"] = eps
                values, precise = rms_references(x, w, eps)

                def measure(k, y):
                    self.record(k, y)
                    if 40 in self.trace.pos and y.dtype == torch.bfloat16:
                        p = int(self.trace.rows[self.trace.pos.index(40)].item())
                        self.ulp[k] = support40(y.reshape(len(x), -1), precise, p)

                for k, y in values.items():
                    measure(k, y)
                for k in ("live.norm_output", "live.moe_input", "live.gate_input"):
                    if k in self.saved:
                        measure(k, self.saved[k])

                def launch(z, inplace, **kw):
                    import flashinfer.norm as fin

                    scratch = z.clone()
                    return fin.rmsnorm(
                        scratch, w, eps=eps, out=scratch if inplace else None, **kw
                    )

                for tag, inplace, opts in (
                    ("out", False, {}),
                    ("in", True, {}),
                    ("pdl_on", False, {"enable_pdl": True}),
                    ("pdl_off", False, {"enable_pdl": False}),
                ):
                    k = "rms." + tag
                    attempt(
                        k,
                        lambda k=k, ip=inplace, opts=opts: measure(
                            k, launch(x, ip, **opts)
                        ),
                    )

                def explicit_cuda():
                    from flashinfer.jit.norm import gen_norm_module
                    from flashinfer.utils import device_support_pdl

                    scratch = x.clone()
                    out = torch.empty_like(scratch)
                    gen_norm_module().build_and_load().rmsnorm(
                        out, scratch, w, eps, device_support_pdl(scratch.device)
                    )
                    measure("rms.cuda_jit", out)

                def explicit_cute():
                    from flashinfer.norm.kernels import rmsnorm_cute
                    from flashinfer.utils import device_support_pdl

                    scratch = x.clone()
                    out = torch.empty_like(scratch)
                    rmsnorm_cute(
                        scratch,
                        w,
                        out,
                        eps,
                        weight_bias=0.0,
                        enable_pdl=device_support_pdl(scratch.device),
                    )
                    measure("rms.cute", out)

                attempt("rms.cuda_jit", explicit_cuda)
                attempt("rms.cute", explicit_cute)
                for i, y in enumerate(hs or (None, None)):
                    if y is not None:
                        k = ("fused", "split")[i] + ".then_native_rms"
                        attempt(k, lambda k=k, y=y: measure(k, launch(y, False)))
        return dict(
            schema=1,
            path=self.path,
            meta=self.meta,
            events=self.docs,
            rounding40=self.ulp,
            unavailable=self.errors,
        )
