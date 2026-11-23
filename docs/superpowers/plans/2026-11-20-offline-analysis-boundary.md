# BTC offline analysis boundary implementation plan

Goal: remove offline branch history and stable-ID eviction ordering from native BTC.
Architecture: native nodes and tree mutation retain upstream behavior. The offline replay reads quiescent native metadata with native group validators for diagnostic FULL/joint lengths; actual reuse, insert and eviction remain native. Static tree export uses children.size() > 1, excluding root, after settlement. Historical insertion-endpoint and eviction-heat counters are no longer collected or represented as zero.
Tech stack: C++17, Python unittest, Bazel metadata target, CPU-only container.

- [x] Add architecture regression tests proving tree mutation/node/comparator match upstream.
- [x] Add native tests for static fork/chain/root detection, read-only exports and checkpoints.
- [x] Remove branch metadata and matching statistics interfaces from production; put read-only analysis in benchmark/metadata.
- [x] Remove history from checkpoint payload; bump schema and explicitly reject old schema.
- [x] Keep generic lifecycle observability, metadata-only pools, native transfer completion, logical refs and watermarks.
- [x] Delete legacy adapter patch that could reintroduce old analysis changes.
- [x] Apply equivalent changes to both delivery branches; run native selfchecks, CP and F/J cases, pressure and resume tests.
- [x] Keep PR #1643 experiment separate; no production branches or automatic publishing.


Validation: both branches passed seven Python build/architecture checks, all three fixed-target native selfchecks, and 128/256-granularity smoke/resume checks in CPU-only containers. For each branch, 1,000 warmup + 2,000 measured real requests and the existing 2,000-request pressure trace matched the respective pre-change FULL/joint arrays exactly. Continuous and resumed real traces also matched. Static export is checked against normalized native state before/after, and legacy checkpoint schema is explicitly rejected. The test cleanup uses POSIX unlink/rmdir because this build environment's libtorch filesystem remove_all symbol faults. No BTC workaround was added for that cleanup issue.
