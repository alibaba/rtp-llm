# Private, pinned GEMM headers

These are unmodified private copies from RTP kernel revision
`bdafb874f725dd1aebf587c3b99d8c87ea77d42d`, read from
`/tmp/cmp-cuda-extension-source/rtp_kernel/glm5/include/` and independently
matched to installed `rtp_kernel==0.1.0+bdafb874.cu132` headers.

| Relative path | SHA256 |
|---|---|
| `rtp_kernel/glm5/deep_gemm/fp8_gemm.cuh` | `7dd9eaa62a7909253f2a044870863c1dc0e94d22e3b71b79721dc241c15e3025` |
| `rtp_kernel/glm5/deep_gemm/fp8_gemm_config.cuh` | `2a7b714d92077676cd73b01ee506d1ddf9a78fb3d613d382ad92e7d804f6c50f` |
| `LICENSE-DeepGEMM` | `6c73a30c14a2938abaa6afe98ebb19e6d81dac5bd4ace0cb05c61c20afa37b85` |

The first header identifies its DeepGEMM origin as
`54e22612409371d6364144b69086735beb54e98b`. DeepGEMM is Copyright (c) 2025
DeepSeek, MIT licensed; the accompanying complete license is copied from the
installed `deep_gemm==2.6.1+7232a0c.cu132` distribution's LICENSE.
The RTP wheel METADATA declares `License: Apache-2.0`; the available pinned RTP
source checkout does not contain a top-level LICENSE file. This notice records
that limitation rather than attributing a fabricated license file to it.

CUTLASS and the remaining DeepGEMM headers are included from the existing pinned
RTP wheel `glm5/include/third_party` tree, retaining their original notices and
licenses in place. They are not edited or redistributed here. Extension naming
hashes their actual bytes in addition to these private copies, preventing stale
extension reuse after dependency changes.

The separate `gemm.cu` host dispatch instantiates group32 A/B scale granularities
for HY4 roles; the copied namespace containing `glm5` is provenance only. It
does not select the GLM5 group128 projection wrappers or their math contracts.
