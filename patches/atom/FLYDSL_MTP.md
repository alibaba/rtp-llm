# ROCm ATOM installation and checks

The ROCm image installs the locked ATOM and AITER wheels and applies their
overlays from this checkout. To reproduce the installation on fresh wheels
using the service interpreter, run:

```sh
bash patches/apply_qwen38_rocm.sh /opt/conda310/bin/python /path/to/flydsl-0.4.2-cp310-cp310-manylinux_2_27_x86_64.whl
```

RTP uses FlyDSL GDN with hybrid replay on supported ROCm targets.
`ATOM_ENABLE_REPLAYSSM=0` selects plain FlyDSL. The small-row MoE TopK
specialization applies to eligible BF16 shapes with 512 routed experts,
top-k 10 and 1–4 rows. Larger batches use the normal selector.
`ATOM_RTP_TOPK512=0` disables the specialization.
The MTP drafter uses fused hyper-connection operations on its ROCm RTP path.

Eligible ROCm ATOM MTP decode workers capture draft forwards, LM heads, TP
gathers and greedy sampling together by default. Set
`RTP_LLM_MTP_DRAFT_LOOP_GRAPH=0` to disable draft-loop capture.
Restart workers after changing these settings.

Run these checks with the same Python interpreter and installed packages:

```sh
python patches/atom/tests/test_flydsl_gdn_mtp.py
python patches/atom/tests/test_hybrid_gdn_graph.py
python patches/atom/topk512/test_topk512.py
```
