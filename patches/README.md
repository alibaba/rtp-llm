# Third-party patch registry

Ownership registry and tracking for all third-party patches in use in this repo:

- **Newly added patches: ownership must be 100%** -- all four items present:
  owner, necessity, upstream tracking, regression coverage
  (see the header of `rules_python/0001-wheelmaker-store-and-zip64.patch` for the template).
- **Legacy patches are listed and tracked as inherited debt** (table below): those
  introduced from upstream keep their original commit header; those with only an
  internal author line use that author line as the claiming starting point; patches
  with no header at all are upgraded to full ownership once an owner claims them
  and adds a header.
- Patches in use are fetch inputs of Bazel extensions: after changing any patch
  bytes you must re-run `scripts/rtpcli bazel lock-update` to refresh the
  recordedFileInputs digests.

| patch | target repo | ownership status | claiming starting point / notes |
| --- | --- | --- | --- |
| `rules_python/0001-wheelmaker-store-and-zip64.patch` | rules_python | **Full** | store+zip64; header contains owner/rationale/upstream tracking/regression coverage |
| `rules_python/0003-fail-closed-on-missing-locked-artifact.patch` | rules_python | **Full** | repository-local policy; no upstream issue filed; removal condition and regression coverage are in the header |
| `grpc/0002-retire-external-binds.patch` | grpc | **Full** | retires bind label mapping; header contains all four items |
| `grpc/0001-Rename-gettid-functions.patch` | grpc | Upstream-introduced | upstream commit ca8b5a914 (Benjamin Peterson), glibc 2.30 gettid conflict |
| `kai/0001-add-a8w4-fp16-support.patch` | KleidiAI | Upstream-introduced | author Tianyu Li @arm.com; a8w4 fp16 support |
| `cutlass/0001-cuda12.4-compat.patch` (3rdparty) | cutlass | Legacy, author line | liukan.lk@alibaba-inc.com; cuda12.4 compatibility |
| `flashinfer/0001-fix-compile.patch` (3rdparty) | flashinfer | Legacy, author line | liukan.lk@alibaba-inc.com (commit 72b94b67c) |
| `flashinfer/0002-dispatch-group-size.patch` (3rdparty) | flashinfer | Legacy, author line | liukan.lk@alibaba-inc.com |
| `flashinfer/0003-tanh-compatibility.patch` (3rdparty) | flashinfer | Legacy, author line | liukan.lk@alibaba-inc.com |
| `flashinfer/0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch` (3rdparty) | flashinfer | Legacy, author line | baowending.bwd@alibaba-inc.com; MLA write kvcache |
| `flashinfer/0006-add-mla-dispatch-inc.patch` (3rdparty) | flashinfer | Legacy, author line | baowending.bwd@alibaba-inc.com |
| `havenask/0001-fix-PrometheusSink-need-header.patch` | havenask | Legacy, author line | zw193905@alibaba-inc.com |
| `rules_python/0002-remove-import-from-rules_cc.patch` | rules_python | Legacy, author line | shuoshu.yh@alibaba-inc.com |
| `flashinfer/0007-fix-nan.patch` (3rdparty) | flashinfer | Legacy, unowned | to be claimed |
| `flashinfer/0008-enable-pdl.patch` (3rdparty) | flashinfer | Legacy, unowned | to be claimed |
| `flashinfer/0009-sp-sample.patch` (3rdparty) | flashinfer | Legacy, unowned | to be claimed |
| `flashinfer/0010-silu-mul-vec-size.patch` (3rdparty) | flashinfer | Legacy, unowned | to be claimed |
| `flashmla/0001-add-interface.patch` (3rdparty) | FlashMLA | Legacy, unowned | to be claimed; internal-source `internal_source/RTP_LLM-PPU/3rdparty/flashmla/0001-add-interface.patch` is the same file |
| `rapidjson/0001-document_h.patch` (3rdparty) | rapidjson | Legacy, unowned | to be claimed |
| `boost/boost.patch` | boost | Legacy, unowned | to be claimed |
| `havenask/anet.patch` | havenask (anet) | Legacy, unowned | to be claimed |
| `havenask/havenask.patch` | havenask | Legacy, unowned | to be claimed |
| `nacos_sdk_cpp/nacos-compile.patch` | nacos_sdk_cpp | Legacy, unowned | to be claimed |

Total 24 in use: full 3 / upstream-introduced 2 / legacy author
line 8 / legacy unowned 11.
