# NaviSched source migration

Source: `/home/zq103303/alibaba/aios/navi_sched`, branch `navi_sched`, base commit
`5a44853cd6aa350d17ac1b7794adef10aa31090a`. The snapshot includes its existing
uncommitted changes; the original workspace is preserved.

Destination on 133:
`/home/zq103303/RTP-LLM/github-opensource/rtp_llm/flexlb/navi_sched`.

## Directory layout

Vendored modules are directly under `vendor/`, including `autil`, `alog`,
`network`, `navi`, `kmonitor`, `unittest_framework`, `bazel` and `third_party`.
The `vendor/alibaba/aios` wrapper directories have been removed. Local include
paths and Bazel labels use `rtp_llm/flexlb/navi_sched/vendor/`.

## Local file operations

The entire `vendor/filesystem` tree and its BUILD dependencies were removed.
Actual call sites now use `autil::FileUtil` in `vendor/autil/autil/FileUtil.h`:

- `readFile`: graph definitions and commit metadata.
- `writeFileAtomically`: compressed Navi history, replacing the destination only
  after the temporary file is written and closed successfully.
- `joinPath`: Navi Python/commit paths and file recorder paths.
- `writeFile`, `makeDirs`, `listDir`, `removePath`: local snapshot recording and
  cleanup.

These are the only FileUtil methods. They use standard C++17 file streams and
filesystem operations. No remote filesystem, plugin loading, fslib API shim or
asynchronous file API remains. A migration check verifies that every declared
FileUtil method has an actual non-test caller.

## Source pruning

The subscription implementation retains VIP only; CM2, Istio/XDS and Local
backends were removed. The allocator, future_lite, tair, std_module, RDMA/ACCL
and OpenTelemetry source modules were removed. Gig outgoing connection creation
retains gRPC stream; the existing ARPC/HTTP server endpoints are still used.

VIP transport is an application-supplied interface declared in
`vendor/network/gig/multi_call/subscribe/VipSubscriptionClient.h`. It exposes
only initialization and typed host queries. Supply an implementation through
`SubscribeServiceFactory::setVipClientFactory` before enabling VIP discovery.
Without an implementation, VIP initialization explicitly fails. No SDK, HTTP
transport, default fake implementation or easy dependency is included. The
unused curl build descriptor was also removed.

## Verified on 2026-10-30

- Eight migration checks pass, including flat directories, no fslib source/build
  references, no unused FileUtil methods, and an abstract VIP interface without
  SDK or easy dependencies.
- The real `local_file_util_test` Bazel target passes with clang, new ABI and
  debug configuration, with test result caching disabled. The isolated metadata
  overlay uses the original workspace configuration and cached external repos;
  this is not a standalone RTP-LLM release build.
- The same test compiles with GCC/C++17 and `-O2 -DNDEBUG -Wall -Wextra -Werror`.
  It exercises actual files: binary/empty content, truncation, missing paths,
  directory errors, write/close failure, atomic replacement failure preserving
  old content, symlinks and FileRecorder cleanup.
- Three isolated negative fixtures deliberately truncate binary data, delete
  the old destination prematurely, or ignore write failures. All three are
  rejected by the test; the original implementation passes again afterward.
- The real `subscribe:vip_subscription_test` Bazel target passes alongside the
  file IO test. It checks missing/null injection, external initialization and
  query failures, successful empty results and preservation of host metadata.
- The complete dependency query for `libnavi_sched.so` resolves to 4,401 labels,
  with zero vipserver, easy, fslib, OpenTelemetry or RDMA dependencies.
- Three additional isolated negative fixtures reintroduce an actual SDK BUILD
  dependency, an easy directory, or a concrete client implementation. All are
  rejected; the unchanged source passes all eight checks again.
- All remaining BUILD/bzl files pass syntax parsing.

Evidence on 133:
`/tmp/navi-fileutil-flat-bazel.log`,
`/tmp/navi-fileutil-flat-profile-20261030.json.gz`, and
`/tmp/navi-fileutil-verification-20261030/results.json` (individual failure logs
and mutated headers are in the adjacent fixture directories).
VIP evidence: `/tmp/navi-vip-interface-bazel.log`,
`/tmp/navi-vip-interface-profile-20261030.json.gz`,
`/tmp/navi-vip-interface-closure.labels`, and
`/tmp/navi-vip-interface-verification-20261030/results.json`.

## Remaining integration work

The full migration is not yet accepted as a buildable release. Old Local-discovery
integration fixtures and adapter examples need updating; complete native service
integration and the remaining affected suites have not passed. Applications
enabling VIP must supply the transport implementation themselves. The selected
subscription/FileUtil tests do not establish those broader integration results.

No production service was changed, and no commit or push was performed.
