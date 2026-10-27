# LM head and sampling qualification

Keep the existing serving path. The tested LM-head variants offer at most a
small operator improvement, while the proposed sampling shortcut violates the
current numerical contract. No end-to-end gain is claimed for this item.

## LM head

The oracle calls the installed native
`torch_ext::cublas_gemm_bf16_bf16_fp32` symbol, not a replacement PyTorch matmul.
Inputs and checkpoint weights are BF16, accumulation and logits output are FP32.
The measured vocabulary projection is N=129280, K=4096 on one M890P.

cuBLAS algorithm, transposed-weight, and vocabulary-chunk candidates were tested
with changing inputs. Only default/99 were supported among the enumerated
algorithms. Pretransposing the weight was bitwise equal but gave no useful gain:

| M | Default (us) | Pretransposed (us) |
| ---: | ---: | ---: |
| 8 | 502.440 | 506.525 |
| 24 | 527.500 | 526.620 |
| 32 | 535.355 | 534.365 |

A separate prototype adapted DeepGEMM's BF16 dense kernel to an FP32 epilogue,
without split-K or intermediate BF16 rounding. Six block/warp/stage choices
passed all **108** byte-equality checks (M=8/24/32, six input magnitudes,
including zero, tiny and large values). Best measured pairs were:

| M | Tile (BM, BN, BK, WM, WN, stages) | Native (us) | Candidate (us) |
| ---: | --- | ---: | ---: |
| 8 | 16,128,64,16,64,4 | 510.110 | 505.855 |
| 24 | 32,256,128,32,64,3 | 529.585 | 523.160 |
| 32 | 32,256,128,32,64,3 | 537.630 | 529.445 |

This is about 0.8-1.5% on the operator. The previously captured rank-0 timeline
placed LM head at 5.77% of summed kernel durations (overlapping durations, not
wall time). That makes the likely service opportunity too small to justify a new
native dependency on this evidence. The prototypes were not installed in the
service. Some other tiles were substantially slower.

The custom-kernel timing used 256 MiB cache flush outside each one-operation
CUDA graph, three alternating rounds of 30 samples. The cuBLAS/layout sweep used
a 1 GiB flush. Compare candidates to their own paired baseline, not between the
two sweeps. A first custom-kernel attempt failed to compile against older installed
headers; the successful run used the matching DeepGEMM source/header package.

## Sampling

The prototype specialized native `rtp_llm::top_p_renorm_probs` for top_p=1 by
skipping radix threshold search while retaining the 1024-thread FP32 reduction
and normalization. It failed **94/144** same-input byte comparisons over batch
1/8/24/32, six seeds and six logit scales. Merely copying softmax probabilities
also changes results. The native radix threshold/reduction behavior cannot be
assumed numerically equivalent to either shortcut, even for top_p=1.

The candidate was rejected before timing or service integration. Replacing the
original probability path would also affect speculative verification, so an
argmax-only comparison is insufficient. The service and model precision remain
unchanged by these experiments; no full model-quality claim follows from the
operator checks.

Source basis: RTP-LLM `7dda4bb2602ffd063e658e2de639793f7003f111`;
custom kernel source DeepGEMM `f94750b2f765875e58dbfcb5d088e00930dfe58c`.
Machine-readable results and experiment file hashes are in
`lm_head_sampling_results.json`.
