# ROCm ATOM installation and checks

The ROCm image applies the overlays from the external checkout to the pinned
ATOM and AITER wheels. To reproduce this on fresh wheels, run:

```sh
bash patches/apply_qwen38_rocm.sh /opt/conda310/bin/python /path/to/flydsl-0.4.2-cp310-cp310-manylinux_2_27_x86_64.whl
```

On MI308X, Qwen4 uses fused routing and HC projection for eligible four- and
eight-row target verifier batches in both decode-only and combined workers.
Combined workers select this path for pure verification batches. The regular
small-row TopK specialization supports other eligible shapes as well.
`ATOM_RTP_TOPK512=0` disables the TopK specialization and fused routing.

`ATOM_ENABLE_REPLAYSSM=0` selects plain FlyDSL GDN and
`RTP_LLM_MTP_DRAFT_LOOP_GRAPH=0` disables draft-loop capture. Restart the
workers after changing either setting.
