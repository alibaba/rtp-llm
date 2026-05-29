# PD KV Writeback TP Equal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make decode-to-prefill PD KV writeback work for `prefill TP == decode TP`, with dedicated runtime metrics and a qwen3 TP4 smoke assertion proving prefill local reuse.

**Architecture:** Reuse the existing P2P writeback data path, but add an explicit topology helper and all-rank fanout. The selected decode gang uses its static local `runtime_config.worker_grpc_addrs`, while the source prefill peer list is still carried per request from forward PD allocate. Writeback remains best effort and gated by `enable_pd_kv_cache_writeback`.

**Tech Stack:** C++17, Bazel, gRPC/protobuf, kmonitor MetricsGroup, existing P2P connector workers, Python smoke test config.

---

## Scope Notes

- Work only in `github-opensource`.
- Do not touch existing dirty test data files:
  - `rtp_llm/test/smoke/data/model/bert/expect.pt`
  - `rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_0.pt`
  - `rtp_llm/test/smoke/data/model/bert/sparse_roberta_expect_1.pt`
- Do not use KVCM or remote cache for this feature.
- Do not change request entrance routing; requests still enter through prefill.
- Do not add runtime metrics for service-startup init failure. Startup init failure should keep failing fast.

## File Map

- Create `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h`
  - Holds topology mode, rank mapping, topology input, and topology build/validation API.
- Create `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.cc`
  - Implements phase-1 `tp_equal` validation and identity rank mapping.
- Modify `rtp_llm/cpp/cache/writeback/BUILD`
  - Adds topology library and test.
- Create `rtp_llm/cpp/cache/writeback/test/PdKvWritebackTopologyTest.cc`
  - Unit tests TP4 identity mapping and invalid topology rejection.
- Create `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h`
  - Adds a dedicated writeback metric group and helper report functions.
- Create `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.cc`
  - Registers and reports dedicated writeback qps/latency/block metrics.
- Modify `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h`
  - Adds topology-aware RPC client interface, send-on-decode entrypoint, metrics reporter, and cache hold helper.
- Modify `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc`
  - Validates topology, fans out prefill receive and decode send RPCs, reports metrics, and exposes `sendOnDecode`.
- Modify `rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc`
  - Adds tests for TP4 fanout, skip reasons, send-on-decode hold behavior, and receive failure cleanup.
- Modify `rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc`
  - Injects local decode gRPC worker list into `PdKvWritebackManager`.
- Modify `rtp_llm/cpp/cache/connector/test/PdKvWritebackCoordinatorStaticTest.py`
  - Asserts no `.front()`-only receive path remains and decode worker addrs are injected.
- Modify `rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h`
  - Routes `PdKvWriteback` to prefill receive or decode send by local role.
- Modify `rtp_llm/cpp/model_rpc/DecodeRpcServer.h`
  - Declares `PdKvWritebackSend`.
- Modify `rtp_llm/cpp/model_rpc/DecodeRpcServer.cc`
  - Converts request PB and calls `writebackManager()->sendOnDecode`.
- Modify `rtp_llm/cpp/model_rpc/PrefillRpcServer.cc`
  - Reuses common writeback PB conversion and reports receive metrics.
- Modify `rtp_llm/test/smoke/suites_h20_oss.bzl`
  - Adds qwen3 TP4 writeback reuse smoke case.
- Modify `rtp_llm/test/smoke/test/pd_writeback_smoke_config_static_test.py`
  - Asserts TP1 and TP4 smoke cases exist and use aux info assertions.

## Task 1: Add Topology Helper

**Files:**
- Create: `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h`
- Create: `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.cc`
- Modify: `rtp_llm/cpp/cache/writeback/BUILD`
- Test: `rtp_llm/cpp/cache/writeback/test/PdKvWritebackTopologyTest.cc`

- [ ] **Step 1: Write the failing topology unit test**

Create `rtp_llm/cpp/cache/writeback/test/PdKvWritebackTopologyTest.cc`:

```cpp
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(PdKvWritebackTopologyTest, TpEqualMapsSameRankForFourWayTp) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 4;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_TRUE(plan.ok()) << plan.status();
    ASSERT_EQ(plan->mode, PdKvWritebackTopologyMode::TpEqual);
    ASSERT_EQ(plan->mappings.size(), 4);
    for (size_t i = 0; i < plan->mappings.size(); ++i) {
        EXPECT_EQ(plan->mappings[i].decode_rank, static_cast<int32_t>(i));
        EXPECT_EQ(plan->mappings[i].prefill_rank, static_cast<int32_t>(i));
        EXPECT_EQ(plan->mappings[i].decode_grpc_addr, input.decode_grpc_addrs[i]);
        EXPECT_EQ(plan->mappings[i].prefill_grpc_addr, input.prefill_grpc_addrs[i]);
        EXPECT_EQ(plan->mappings[i].prefill_worker_addr, input.prefill_worker_addrs[i]);
        EXPECT_EQ(plan->mappings[i].local_partition_count, 1);
        EXPECT_EQ(plan->mappings[i].local_partition_id, 0);
        EXPECT_EQ(plan->mappings[i].remote_partition_count, 1);
        EXPECT_EQ(plan->mappings[i].remote_partition_id, 0);
    }
}

TEST(PdKvWritebackTopologyTest, RejectsMismatchedPrefillGrpcAndWorkerAddrs) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 4;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(std::string(plan.status().message()).find("prefill address count mismatch"), std::string::npos);
}

TEST(PdKvWritebackTopologyTest, RejectsUnequalTpInPhaseOne) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 2;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000", "p1:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), absl::StatusCode::kUnimplemented);
    EXPECT_NE(std::string(plan.status().message()).find("unsupported_topology"), std::string::npos);
}

}  // namespace rtp_llm
```

- [ ] **Step 2: Add the topology test target and run it to verify it fails**

Modify `rtp_llm/cpp/cache/writeback/BUILD`:

```python
cc_library(
    name = "pd_kv_writeback_topology",
    srcs = ["PdKvWritebackTopology.cc"],
    hdrs = ["PdKvWritebackTopology.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "test_pd_kv_writeback_topology",
    srcs = ["test/PdKvWritebackTopologyTest.cc"],
    copts = copts(),
    deps = [
        ":pd_kv_writeback_topology",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
```

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_topology
```

Expected: FAIL because `PdKvWritebackTopology.h` does not exist yet.

- [ ] **Step 3: Implement the topology API**

Create `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace rtp_llm {

enum class PdKvWritebackTopologyMode {
    TpEqual,
};

struct PdKvWritebackTopologyInput {
    int32_t local_tp_size               = 1;
    int32_t source_partition_count      = 1;
    int32_t destination_partition_count = 1;
    bool    prefill_cp_enabled          = false;

    std::vector<std::string> decode_grpc_addrs;
    std::vector<std::string> prefill_grpc_addrs;
    std::vector<std::string> prefill_worker_addrs;
};

struct PdKvWritebackRankMapping {
    int32_t decode_rank  = 0;
    int32_t prefill_rank = 0;

    std::string decode_grpc_addr;
    std::string prefill_grpc_addr;
    std::string prefill_worker_addr;

    int32_t local_partition_count  = 1;
    int32_t local_partition_id     = 0;
    int32_t remote_partition_count = 1;
    int32_t remote_partition_id    = 0;
};

struct PdKvWritebackTopologyPlan {
    PdKvWritebackTopologyMode              mode = PdKvWritebackTopologyMode::TpEqual;
    std::vector<PdKvWritebackRankMapping> mappings;
};

absl::StatusOr<PdKvWritebackTopologyPlan>
buildPdKvWritebackTopology(const PdKvWritebackTopologyInput& input);

}  // namespace rtp_llm
```

Create `rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.cc`:

```cpp
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"

#include "absl/status/status.h"

namespace rtp_llm {

absl::StatusOr<PdKvWritebackTopologyPlan>
buildPdKvWritebackTopology(const PdKvWritebackTopologyInput& input) {
    if (input.local_tp_size <= 0) {
        return absl::FailedPreconditionError("local_tp_size must be positive");
    }
    if (input.source_partition_count <= 0 || input.destination_partition_count <= 0) {
        return absl::FailedPreconditionError("partition_count must be positive");
    }
    if (input.prefill_grpc_addrs.size() != input.prefill_worker_addrs.size()) {
        return absl::FailedPreconditionError("prefill address count mismatch");
    }
    if (input.prefill_cp_enabled) {
        return absl::UnimplementedError("unsupported_topology: prefill cp writeback is not supported in phase 1");
    }
    const size_t local_tp_size = static_cast<size_t>(input.local_tp_size);
    if (input.decode_grpc_addrs.size() != local_tp_size) {
        return absl::FailedPreconditionError("decode grpc address count mismatch");
    }
    if (input.prefill_grpc_addrs.size() != local_tp_size) {
        return absl::UnimplementedError("unsupported_topology: prefill tp must equal decode tp in phase 1");
    }
    if (input.source_partition_count != input.destination_partition_count
        || input.source_partition_count != input.local_tp_size) {
        return absl::UnimplementedError("unsupported_topology: source/destination partition counts must equal local tp");
    }

    PdKvWritebackTopologyPlan plan;
    plan.mode = PdKvWritebackTopologyMode::TpEqual;
    plan.mappings.reserve(local_tp_size);
    for (size_t i = 0; i < local_tp_size; ++i) {
        PdKvWritebackRankMapping mapping;
        mapping.decode_rank            = static_cast<int32_t>(i);
        mapping.prefill_rank           = static_cast<int32_t>(i);
        mapping.decode_grpc_addr       = input.decode_grpc_addrs[i];
        mapping.prefill_grpc_addr      = input.prefill_grpc_addrs[i];
        mapping.prefill_worker_addr    = input.prefill_worker_addrs[i];
        mapping.local_partition_count  = 1;
        mapping.local_partition_id     = 0;
        mapping.remote_partition_count = 1;
        mapping.remote_partition_id    = 0;
        plan.mappings.push_back(std::move(mapping));
    }
    return plan;
}

}  // namespace rtp_llm
```

- [ ] **Step 4: Run topology tests**

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_topology
```

Expected: PASS.

- [ ] **Step 5: Commit topology helper**

Run:

```bash
git add rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h \
        rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.cc \
        rtp_llm/cpp/cache/writeback/test/PdKvWritebackTopologyTest.cc \
        rtp_llm/cpp/cache/writeback/BUILD
git commit -m "feat: add PD KV writeback topology helper"
```

## Task 2: Add Dedicated Writeback Metrics

**Files:**
- Create: `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h`
- Create: `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.cc`
- Modify: `rtp_llm/cpp/cache/writeback/BUILD`

- [ ] **Step 1: Add metrics library target**

Modify `rtp_llm/cpp/cache/writeback/BUILD`:

```python
cc_library(
    name = "pd_kv_writeback_metrics",
    srcs = ["PdKvWritebackMetrics.cc"],
    hdrs = ["PdKvWritebackMetrics.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)
```

Run:

```bash
bazel build //rtp_llm/cpp/cache/writeback:pd_kv_writeback_metrics
```

Expected: FAIL because the files do not exist yet.

- [ ] **Step 2: Implement metrics header**

Create `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class PdKvWritebackMetricsCollector final {
public:
    bool    launch_qps          = false;
    bool    launch_failed_qps   = false;
    bool    launch_skipped_qps  = false;
    int64_t launch_latency_us   = 0;

    bool    rpc_qps        = false;
    bool    rpc_failed_qps = false;
    int64_t rpc_latency_us = 0;

    bool    transfer_qps        = false;
    bool    transfer_failed_qps = false;
    int64_t transfer_latency_us = 0;

    bool    receive_qps        = false;
    bool    receive_failed_qps = false;
    int64_t receive_latency_us = 0;

    int64_t malloc_latency_us = 0;
    int64_t commit_latency_us = 0;
    int64_t block_count       = 0;
    int64_t token_count       = 0;
};

class PdKvWritebackMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, PdKvWritebackMetricsCollector* collector);

private:
    kmonitor::MutableMetric* launch_qps_metric          = nullptr;
    kmonitor::MutableMetric* launch_failed_qps_metric   = nullptr;
    kmonitor::MutableMetric* launch_skipped_qps_metric  = nullptr;
    kmonitor::MutableMetric* launch_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* rpc_qps_metric             = nullptr;
    kmonitor::MutableMetric* rpc_failed_qps_metric      = nullptr;
    kmonitor::MutableMetric* rpc_latency_us_metric      = nullptr;
    kmonitor::MutableMetric* transfer_qps_metric        = nullptr;
    kmonitor::MutableMetric* transfer_failed_qps_metric = nullptr;
    kmonitor::MutableMetric* transfer_latency_us_metric = nullptr;
    kmonitor::MutableMetric* receive_qps_metric         = nullptr;
    kmonitor::MutableMetric* receive_failed_qps_metric  = nullptr;
    kmonitor::MutableMetric* receive_latency_us_metric  = nullptr;
    kmonitor::MutableMetric* malloc_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* commit_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* block_count_metric         = nullptr;
    kmonitor::MutableMetric* token_count_metric         = nullptr;
};

void reportPdKvWritebackMetric(const kmonitor::MetricsReporterPtr& reporter,
                               PdKvWritebackMetricsCollector&      collector,
                               const std::string&                  stage,
                               const std::string&                  status,
                               const std::string&                  reason,
                               const std::string&                  role,
                               int32_t                             tp_size,
                               const std::string&                  topology);

}  // namespace rtp_llm
```

- [ ] **Step 3: Implement metrics source**

Create `rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.cc`:

```cpp
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h"

namespace rtp_llm {

bool PdKvWritebackMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(launch_qps_metric, "rtp_llm_pd_kv_writeback_launch_qps");
    REGISTER_QPS_MUTABLE_METRIC(launch_failed_qps_metric, "rtp_llm_pd_kv_writeback_launch_failed_qps");
    REGISTER_QPS_MUTABLE_METRIC(launch_skipped_qps_metric, "rtp_llm_pd_kv_writeback_launch_skipped_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(launch_latency_us_metric, "rtp_llm_pd_kv_writeback_launch_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(rpc_qps_metric, "rtp_llm_pd_kv_writeback_rpc_qps");
    REGISTER_QPS_MUTABLE_METRIC(rpc_failed_qps_metric, "rtp_llm_pd_kv_writeback_rpc_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(rpc_latency_us_metric, "rtp_llm_pd_kv_writeback_rpc_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(transfer_qps_metric, "rtp_llm_pd_kv_writeback_transfer_qps");
    REGISTER_QPS_MUTABLE_METRIC(transfer_failed_qps_metric, "rtp_llm_pd_kv_writeback_transfer_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_latency_us_metric, "rtp_llm_pd_kv_writeback_transfer_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(receive_qps_metric, "rtp_llm_pd_kv_writeback_receive_qps");
    REGISTER_QPS_MUTABLE_METRIC(receive_failed_qps_metric, "rtp_llm_pd_kv_writeback_receive_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(receive_latency_us_metric, "rtp_llm_pd_kv_writeback_receive_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(malloc_latency_us_metric, "rtp_llm_pd_kv_writeback_malloc_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(commit_latency_us_metric, "rtp_llm_pd_kv_writeback_commit_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(block_count_metric, "rtp_llm_pd_kv_writeback_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(token_count_metric, "rtp_llm_pd_kv_writeback_token_count");
    return true;
}

void PdKvWritebackMetrics::report(const kmonitor::MetricsTags* tags, PdKvWritebackMetricsCollector* collector) {
    if (collector->launch_qps) {
        REPORT_MUTABLE_QPS(launch_qps_metric);
    }
    if (collector->launch_failed_qps) {
        REPORT_MUTABLE_QPS(launch_failed_qps_metric);
    }
    if (collector->launch_skipped_qps) {
        REPORT_MUTABLE_QPS(launch_skipped_qps_metric);
    }
    if (collector->launch_latency_us > 0) {
        REPORT_MUTABLE_METRIC(launch_latency_us_metric, collector->launch_latency_us);
    }
    if (collector->rpc_qps) {
        REPORT_MUTABLE_QPS(rpc_qps_metric);
    }
    if (collector->rpc_failed_qps) {
        REPORT_MUTABLE_QPS(rpc_failed_qps_metric);
    }
    if (collector->rpc_latency_us > 0) {
        REPORT_MUTABLE_METRIC(rpc_latency_us_metric, collector->rpc_latency_us);
    }
    if (collector->transfer_qps) {
        REPORT_MUTABLE_QPS(transfer_qps_metric);
    }
    if (collector->transfer_failed_qps) {
        REPORT_MUTABLE_QPS(transfer_failed_qps_metric);
    }
    if (collector->transfer_latency_us > 0) {
        REPORT_MUTABLE_METRIC(transfer_latency_us_metric, collector->transfer_latency_us);
    }
    if (collector->receive_qps) {
        REPORT_MUTABLE_QPS(receive_qps_metric);
    }
    if (collector->receive_failed_qps) {
        REPORT_MUTABLE_QPS(receive_failed_qps_metric);
    }
    if (collector->receive_latency_us > 0) {
        REPORT_MUTABLE_METRIC(receive_latency_us_metric, collector->receive_latency_us);
    }
    if (collector->malloc_latency_us > 0) {
        REPORT_MUTABLE_METRIC(malloc_latency_us_metric, collector->malloc_latency_us);
    }
    if (collector->commit_latency_us > 0) {
        REPORT_MUTABLE_METRIC(commit_latency_us_metric, collector->commit_latency_us);
    }
    if (collector->block_count > 0) {
        REPORT_MUTABLE_METRIC(block_count_metric, collector->block_count);
    }
    if (collector->token_count > 0) {
        REPORT_MUTABLE_METRIC(token_count_metric, collector->token_count);
    }
}

void reportPdKvWritebackMetric(const kmonitor::MetricsReporterPtr& reporter,
                               PdKvWritebackMetricsCollector&      collector,
                               const std::string&                  stage,
                               const std::string&                  status,
                               const std::string&                  reason,
                               const std::string&                  role,
                               int32_t                             tp_size,
                               const std::string&                  topology) {
    if (!reporter) {
        return;
    }
    kmonitor::MetricsTags tags;
    tags.AddTag("stage", stage);
    tags.AddTag("status", status);
    tags.AddTag("reason", reason);
    tags.AddTag("role", role);
    tags.AddTag("tp_size", std::to_string(tp_size));
    tags.AddTag("topology", topology);
    reporter->report<PdKvWritebackMetrics, PdKvWritebackMetricsCollector>(&tags, &collector);
}

}  // namespace rtp_llm
```

- [ ] **Step 4: Run metrics build**

Run:

```bash
bazel build //rtp_llm/cpp/cache/writeback:pd_kv_writeback_metrics
```

Expected: PASS.

- [ ] **Step 5: Commit metrics scaffold**

Run:

```bash
git add rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h \
        rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.cc \
        rtp_llm/cpp/cache/writeback/BUILD
git commit -m "feat: add PD KV writeback metrics"
```

## Task 3: Make Launch Topology-Aware and Fan Out to All Ranks

**Files:**
- Modify: `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h`
- Modify: `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc`
- Modify: `rtp_llm/cpp/cache/writeback/BUILD`
- Modify: `rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc`

- [ ] **Step 1: Write failing manager fanout tests**

Add to `rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc`:

```cpp
class RecordingPdKvWritebackRpcClient: public PdKvWritebackRpcClient {
public:
    absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                       const PdKvWritebackTopologyPlan&   topology) override {
        prefill_targets.clear();
        for (const auto& mapping : topology.mappings) {
            prefill_targets.push_back(mapping.prefill_grpc_addr);
        }
        observed_request_key = request.manifest.request_key;
        return absl::OkStatus();
    }

    absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                   const PdKvWritebackTopologyPlan&   topology) override {
        decode_targets.clear();
        for (const auto& mapping : topology.mappings) {
            decode_targets.push_back(mapping.decode_grpc_addr);
        }
        observed_request_key = request.manifest.request_key;
        return absl::OkStatus();
    }

    std::vector<std::string> prefill_targets;
    std::vector<std::string> decode_targets;
    std::string              observed_request_key;
};

TEST(PdKvWritebackManagerTest, LaunchTpEqualFansOutToAllPrefillAndDecodeRanks) {
    PDSepConfig config;
    config.enable_pd_kv_cache_writeback = true;
    auto rpc_client = std::make_shared<RecordingPdKvWritebackRpcClient>();
    auto transfer_client = std::make_shared<NoopPdKvWritebackTransferClient>();
    const std::vector<std::string> decode_worker_grpc_addrs = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    PdKvWritebackManager manager(config, nullptr, transfer_client, rpc_client, decode_worker_grpc_addrs, nullptr);

    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 7;
    request.manifest.request_key          = "request_7";
    request.manifest.final_token_count    = 128;
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {11, 12};
    request.source.partition_count        = 4;
    request.destination.partition_count   = 4;
    request.source.seq_size_per_block      = request.destination.seq_size_per_block = 64;
    request.source.layer_count             = request.destination.layer_count = 1;
    request.source.group_count             = request.destination.group_count = 1;
    request.source.layer_to_group_id       = request.destination.layer_to_group_id = {0};
    request.source.group_types             = request.destination.group_types = {0};
    request.source_prefill_grpc_addrs     = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    request.prefill_worker_addrs          = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};

    auto result = manager.launchFromDecode(request);
    ASSERT_EQ(result.status, PdKvWritebackLaunchStatus::Started);
    ASSERT_EQ(result.reason, "started");
    manager.waitForWritebackTasksForTest();

    EXPECT_EQ(rpc_client->prefill_targets, request.source_prefill_grpc_addrs);
    EXPECT_EQ(rpc_client->decode_targets, decode_worker_grpc_addrs);
    EXPECT_EQ(rpc_client->observed_request_key, "request_7");
}
```

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: FAIL because the `PdKvWritebackRpcClient` signature and `waitForWritebackTasksForTest` do not exist.

- [ ] **Step 2: Update manager header**

Modify `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h`:

```cpp
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"

class PdKvWritebackRpcClient {
public:
    virtual ~PdKvWritebackRpcClient() = default;
    virtual absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                               const PdKvWritebackTopologyPlan&   topology) = 0;
    virtual absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                           const PdKvWritebackTopologyPlan&   topology) = 0;
};
```

Add constructor and test hook:

```cpp
PdKvWritebackManager(const PDSepConfig&                           pd_config,
                     PdKvWritebackCacheWriter*                    cache_writer,
                     std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                     std::shared_ptr<PdKvWritebackRpcClient>      rpc_client,
                     std::vector<std::string>                     decode_worker_grpc_addrs,
                     kmonitor::MetricsReporterPtr                 metrics_reporter);

absl::Status sendOnDecode(const PdKvWritebackLaunchRequest& request,
                          const BatchKVCacheResourcePtr&    source_resource);

void waitForWritebackTasksForTest() const;
```

Add members:

```cpp
kmonitor::MetricsReporterPtr metrics_reporter_;
std::vector<std::string>     decode_worker_grpc_addrs_;
mutable std::mutex           writeback_tasks_mutex_;
mutable std::vector<std::future<void>> writeback_tasks_;
```

- [ ] **Step 3: Update manager implementation**

Modify `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc`:

```cpp
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
```

Add topology input builder:

```cpp
namespace {

std::string pdKvRoleName(RoleType role_type) {
    return role_type == RoleType::PREFILL ? "prefill" : "decode";
}

PdKvWritebackTopologyInput buildTopologyInput(const PdKvWritebackLaunchRequest& request) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size               = request.destination.partition_count;
    input.source_partition_count      = request.source.partition_count;
    input.destination_partition_count = request.destination.partition_count;
    input.decode_grpc_addrs           = request.decode_worker_addrs;
    input.prefill_grpc_addrs          = request.source_prefill_grpc_addrs;
    input.prefill_worker_addrs        = request.prefill_worker_addrs;
    return input;
}

}  // namespace
```

Update the owning constructor:

```cpp
PdKvWritebackManager::PdKvWritebackManager(const PDSepConfig&                           pd_config,
                                           PdKvWritebackCacheWriter*                    cache_writer,
                                           std::shared_ptr<PdKvWritebackTransferClient> transfer_client,
                                           std::shared_ptr<PdKvWritebackRpcClient>      rpc_client,
                                           std::vector<std::string>                     decode_worker_grpc_addrs,
                                           kmonitor::MetricsReporterPtr                 metrics_reporter):
    pd_config_(pd_config),
    cache_writer_(cache_writer),
    transfer_client_(transfer_client.get()),
    rpc_client_(rpc_client.get()),
    owned_transfer_client_(std::move(transfer_client)),
    owned_rpc_client_(std::move(rpc_client)),
    metrics_reporter_(std::move(metrics_reporter)),
    decode_worker_grpc_addrs_(std::move(decode_worker_grpc_addrs)) {}
```

Update `launchFromDecode` to validate topology and fan out:

```cpp
PdKvWritebackLaunchResult PdKvWritebackManager::launchFromDecode(const PdKvWritebackLaunchRequest& request) const {
    const int64_t launch_begin_us = currentTimeUs();
    auto report_launch = [&](PdKvWritebackLaunchStatus status, const std::string& reason) {
        PdKvWritebackMetricsCollector collector;
        collector.launch_qps = true;
        collector.launch_latency_us = currentTimeUs() - launch_begin_us;
        collector.block_count = request.manifest.reusable_block_count;
        collector.token_count = request.manifest.final_token_count;
        if (status == PdKvWritebackLaunchStatus::Skipped) {
            collector.launch_skipped_qps = true;
        } else if (status == PdKvWritebackLaunchStatus::Failed) {
            collector.launch_failed_qps = true;
        }
        reportPdKvWritebackMetric(metrics_reporter_,
                                  collector,
                                  "launch",
                                  status == PdKvWritebackLaunchStatus::Started ? "started" :
                                      status == PdKvWritebackLaunchStatus::Skipped ? "skipped" : "failed",
                                  reason,
                                  pdKvRoleName(pd_config_.role_type),
                                  request.destination.partition_count,
                                  "tp_equal");
    };

    if (!pd_config_.enable_pd_kv_cache_writeback) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "disabled");
        return {PdKvWritebackLaunchStatus::Skipped, "disabled"};
    }
    if (request.manifest.reusable_block_count == 0) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "empty_manifest");
        return {PdKvWritebackLaunchStatus::Skipped, "empty_manifest"};
    }
    auto compatibility_status = validatePdKvWritebackCompatibility(request.source, request.destination);
    if (!compatibility_status.ok()) {
        report_launch(PdKvWritebackLaunchStatus::Skipped, "compatibility_mismatch");
        return {PdKvWritebackLaunchStatus::Skipped, std::string(compatibility_status.message())};
    }
    auto request_for_topology = request;
    if (request_for_topology.decode_worker_addrs.empty()) {
        request_for_topology.decode_worker_addrs = decode_worker_grpc_addrs_;
    }
    auto topology_status = buildPdKvWritebackTopology(buildTopologyInput(request_for_topology));
    if (!topology_status.ok()) {
        const std::string reason = topology_status.status().code() == absl::StatusCode::kUnimplemented ?
            "unsupported_topology" : "topology_mismatch";
        report_launch(PdKvWritebackLaunchStatus::Skipped, reason);
        return {PdKvWritebackLaunchStatus::Skipped, std::string(topology_status.status().message())};
    }
    auto* rpc_client = owned_rpc_client_ ? owned_rpc_client_.get() : rpc_client_;
    if (!rpc_client) {
        report_launch(PdKvWritebackLaunchStatus::Failed, "rpc_client_null");
        return {PdKvWritebackLaunchStatus::Failed, "rpc_client_null"};
    }

    auto request_copy = request_for_topology;
    auto topology     = std::move(topology_status).value();
    auto rpc_owner    = owned_rpc_client_;
    auto transfer_owner = owned_transfer_client_;
    {
        std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
        writeback_tasks_.push_back(std::async(std::launch::async,
                                              [this, request_copy, topology, rpc_client, rpc_owner, transfer_owner]() {
            auto receive_status = rpc_client->requestPrefillReceive(request_copy, topology);
            auto send_status    = rpc_client->requestDecodeSend(request_copy, topology);
            if (!receive_status.ok()) {
                RTP_LLM_LOG_WARNING("PD KV writeback prefill receive fanout failed, request_id=%ld, error=%s",
                                    request_copy.manifest.request_id,
                                    receive_status.ToString().c_str());
            }
            if (!send_status.ok()) {
                RTP_LLM_LOG_WARNING("PD KV writeback decode send fanout failed, request_id=%ld, error=%s",
                                    request_copy.manifest.request_id,
                                    send_status.ToString().c_str());
            }
        }));
    }
    report_launch(PdKvWritebackLaunchStatus::Started, "started");
    return {PdKvWritebackLaunchStatus::Started, "started"};
}
```

Add `waitForWritebackTasksForTest`:

```cpp
void PdKvWritebackManager::waitForWritebackTasksForTest() const {
    std::vector<std::future<void>> tasks;
    {
        std::lock_guard<std::mutex> lock(writeback_tasks_mutex_);
        tasks.swap(writeback_tasks_);
    }
    for (auto& task : tasks) {
        task.get();
    }
}
```

- [ ] **Step 4: Add topology and metrics deps**

Modify `rtp_llm/cpp/cache/writeback/BUILD`:

```python
deps = [
    ":pd_kv_writeback_manifest",
    ":pd_kv_writeback_metrics",
    ":pd_kv_writeback_topology",
    ":pd_kv_writeback_transfer",
    "//rtp_llm/cpp/cache:batch_kv_cache_resource",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/utils:time_util",
    "@com_google_absl//absl/status",
    "@havenask//aios/kmonitor:kmonitor_client_cpp",
]
```

- [ ] **Step 5: Run manager tests**

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: PASS.

- [ ] **Step 6: Commit manager fanout**

Run:

```bash
git add rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h \
        rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc \
        rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc \
        rtp_llm/cpp/cache/writeback/BUILD
git commit -m "feat: fan out PD KV writeback by topology"
```

## Task 4: Inject Decode Worker gRPC Addresses and Implement RPC Fanout

**Files:**
- Modify: `rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc`
- Modify: `rtp_llm/cpp/cache/connector/test/PdKvWritebackCoordinatorStaticTest.py`

- [ ] **Step 1: Add static test assertions**

Modify `rtp_llm/cpp/cache/connector/test/PdKvWritebackCoordinatorStaticTest.py`:

```python
def test_writeback_rpc_client_uses_all_prefill_and_decode_workers(self):
    source = self._read("rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc")
    self.assertIn("requestPrefillReceive(const PdKvWritebackLaunchRequest& request,", source)
    self.assertIn("requestDecodeSend(const PdKvWritebackLaunchRequest& request,", source)
    self.assertNotIn("request.source_prefill_grpc_addrs.front()", source)
    self.assertIn("runtime_config_.worker_grpc_addrs", source)
    self.assertIn("std::make_shared<PdKvWritebackManager>(pd_sep_config_,", source)
```

Run:

```bash
bazel test //rtp_llm/cpp/cache/connector/test:pd_kv_writeback_coordinator_static_test
```

Expected: FAIL because the RPC client still uses `.front()`.

- [ ] **Step 2: Update `GrpcPdKvWritebackRpcClient` constructor and fanout methods**

Modify the class inside `rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc`:

```cpp
class GrpcPdKvWritebackRpcClient: public PdKvWritebackRpcClient {
public:
    explicit GrpcPdKvWritebackRpcClient(kmonitor::MetricsReporterPtr metrics_reporter):
        metrics_reporter_(std::move(metrics_reporter)) {}

    absl::Status requestPrefillReceive(const PdKvWritebackLaunchRequest& request,
                                       const PdKvWritebackTopologyPlan& topology) override {
        return fanoutPdKvWriteback(request, topology, /*target_decode=*/false);
    }

    absl::Status requestDecodeSend(const PdKvWritebackLaunchRequest& request,
                                   const PdKvWritebackTopologyPlan& topology) override {
        return fanoutPdKvWriteback(request, topology, /*target_decode=*/true);
    }

private:
    absl::Status fanoutPdKvWriteback(const PdKvWritebackLaunchRequest& request,
                                     const PdKvWritebackTopologyPlan& topology,
                                     bool target_decode) {
        absl::Status first_error = absl::OkStatus();
        for (const auto& mapping : topology.mappings) {
            const std::string& addr = target_decode ? mapping.decode_grpc_addr : mapping.prefill_grpc_addr;
            auto status = sendPdKvWritebackRpc(addr, request);
            if (!status.ok() && first_error.ok()) {
                first_error = status;
            }
        }
        return first_error;
    }

    absl::Status sendPdKvWritebackRpc(const std::string& addr, const PdKvWritebackLaunchRequest& request) {
        auto connection_status = rpc_pool_.getConnection(addr);
        if (!connection_status.ok()) {
            return connection_status.status();
        }
        grpc::ClientContext client_context;
        if (request.deadline_ms > 0) {
            const auto timeout_ms = request.deadline_ms - currentTimeMs();
            if (timeout_ms > 0) {
                client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
            }
        }
        auto request_pb = buildPdKvWritebackRequestPB(request);
        PdKvWritebackResponsePB response;
        auto grpc_status = connection_status.value().stub->PdKvWriteback(&client_context, request_pb, &response);
        if (!grpc_status.ok()) {
            return absl::InternalError(grpc_status.error_message());
        }
        if (response.has_error_info() && response.error_info().error_code() != ErrorCodePB::NONE_ERROR) {
            return absl::InternalError(response.error_info().error_message());
        }
        if (!response.accepted()) {
            return absl::FailedPreconditionError(response.reason());
        }
        return absl::OkStatus();
    }

private:
    RPCPool rpc_pool_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
};
```

- [ ] **Step 3: Populate `request.decode_worker_addrs` from the manager-owned decode gang list**

Modify `PdKvWritebackManager::launchFromDecode` before building topology:

```cpp
auto request_for_topology = request;
if (request_for_topology.decode_worker_addrs.empty()) {
    request_for_topology.decode_worker_addrs = decode_worker_grpc_addrs_;
}
```

- [ ] **Step 4: Inject local decode worker list at coordinator init**

Modify `initPdKvWriteback()`:

```cpp
if (pd_sep_config_.role_type == RoleType::DECODE) {
    pd_kv_writeback_rpc_client_ =
        std::make_shared<GrpcPdKvWritebackRpcClient>(metrics_reporter_);
}
pd_kv_writeback_manager_ = std::make_shared<PdKvWritebackManager>(pd_sep_config_,
                                                                  pd_kv_writeback_cache_writer_.get(),
                                                                  pd_kv_writeback_transfer_client_,
                                                                  pd_kv_writeback_rpc_client_,
                                                                  runtime_config_.worker_grpc_addrs,
                                                                  metrics_reporter_);
```

- [ ] **Step 5: Run static and manager tests**

Run:

```bash
bazel test //rtp_llm/cpp/cache/connector/test:pd_kv_writeback_coordinator_static_test
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: PASS.

- [ ] **Step 6: Commit RPC fanout**

Run:

```bash
git add rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc \
        rtp_llm/cpp/cache/connector/test/PdKvWritebackCoordinatorStaticTest.py \
        rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h \
        rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc
git commit -m "feat: fan out PD KV writeback RPCs"
```

## Task 5: Add Decode-Side Writeback Send Handler and Lifecycle Hold

**Files:**
- Modify: `rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h`
- Modify: `rtp_llm/cpp/model_rpc/DecodeRpcServer.h`
- Modify: `rtp_llm/cpp/model_rpc/DecodeRpcServer.cc`
- Modify: `rtp_llm/cpp/model_rpc/PrefillRpcServer.cc`
- Create: `rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h`
- Create: `rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.cc`
- Modify: `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h`
- Modify: `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc`
- Test: `rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc`

- [ ] **Step 1: Add manager test for send-on-decode hold and transfer**

Add to `PdKvWritebackManagerTest.cc`:

```cpp
class RecordingTransferClient: public PdKvWritebackTransferClient {
public:
    absl::Status transfer(const PdKvWritebackTransferPlan& plan) override {
        seen_plan = plan;
        return transfer_status;
    }

    PdKvWritebackTransferPlan seen_plan;
    absl::Status transfer_status = absl::OkStatus();
};

TEST(PdKvWritebackManagerTest, SendOnDecodeTransfersFromHeldSourceBlocks) {
    PDSepConfig config;
    config.enable_pd_kv_cache_writeback = true;
    auto transfer_client = std::make_shared<RecordingTransferClient>();
    PdKvWritebackManager manager(config, nullptr, transfer_client, nullptr, {}, nullptr);

    auto source_resource = std::make_shared<BatchKVCacheResource>();
    source_resource->resetBatchSize(1);
    source_resource->initBatchGroups(0, 1, 1, {0});
    source_resource->setBatchBlocks(0, 0, {101, 102});
    source_resource->setBatchCacheKeys(0, {11, 12});

    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = 9;
    request.manifest.request_key          = "request_9";
    request.manifest.reusable_block_count = 2;
    request.manifest.cache_keys           = {11, 12};
    request.manifest.group_block_ids      = {{101, 102}};
    request.source.partition_count        = 4;
    request.destination.partition_count   = 4;
    request.source.layer_count            = 1;
    request.source.group_count            = 1;
    request.source.layer_to_group_id      = {0};
    request.prefill_worker_addrs          = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};

    auto status = manager.sendOnDecode(request, source_resource);
    ASSERT_TRUE(status.ok()) << status;
    EXPECT_EQ(transfer_client->seen_plan.request_key, "request_9");
    EXPECT_EQ(transfer_client->seen_plan.decode_group_block_ids.size(), 1);
    EXPECT_EQ(transfer_client->seen_plan.decode_group_block_ids[0], std::vector<int32_t>({101, 102}));
    EXPECT_EQ(transfer_client->seen_plan.prefill_transfer_servers.size(), 4);
}
```

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: FAIL because `sendOnDecode` does not exist.

- [ ] **Step 2: Implement `sendOnDecode` in manager**

Add to `PdKvWritebackManager.cc`:

```cpp
absl::Status PdKvWritebackManager::sendOnDecode(const PdKvWritebackLaunchRequest& request,
                                                const BatchKVCacheResourcePtr&    source_resource) {
    const int64_t begin_us = currentTimeUs();
    PdKvWritebackMetricsCollector collector;
    collector.transfer_qps = true;
    collector.block_count = request.manifest.reusable_block_count;
    collector.token_count = request.manifest.final_token_count;

    auto* transfer_client = owned_transfer_client_ ? owned_transfer_client_.get() : transfer_client_;
    if (!transfer_client) {
        collector.transfer_failed_qps = true;
        collector.transfer_latency_us = currentTimeUs() - begin_us;
        reportPdKvWritebackMetric(metrics_reporter_, collector, "decode_send", "failed", "transfer_client_null",
                                  "decode", request.source.partition_count, "tp_equal");
        return absl::FailedPreconditionError("transfer_client is null");
    }
    if (!source_resource || source_resource->batchSize() != 1) {
        collector.transfer_failed_qps = true;
        collector.transfer_latency_us = currentTimeUs() - begin_us;
        reportPdKvWritebackMetric(metrics_reporter_, collector, "decode_send", "failed", "source_resource_invalid",
                                  "decode", request.source.partition_count, "tp_equal");
        return absl::FailedPreconditionError("source_resource is invalid");
    }
    auto plan = buildDecodeTransferPlan(request);
    plan.decode_group_block_ids = extractPdKvWritebackGroupBlockIds(source_resource);
    auto status = transfer_client->transfer(plan);
    collector.transfer_latency_us = currentTimeUs() - begin_us;
    if (!status.ok()) {
        collector.transfer_failed_qps = true;
        reportPdKvWritebackMetric(metrics_reporter_, collector, "decode_send", "failed", "transfer_failed",
                                  "decode", request.source.partition_count, "tp_equal");
        return status;
    }
    reportPdKvWritebackMetric(metrics_reporter_, collector, "decode_send", "success", "ok",
                              "decode", request.source.partition_count, "tp_equal");
    return absl::OkStatus();
}
```

- [ ] **Step 3: Route `PdKvWriteback` by local role**

Modify `rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h`:

```cpp
grpc::Status PdKvWriteback(grpc::ServerContext*          context,
                           const PdKvWritebackRequestPB* request,
                           PdKvWritebackResponsePB*      response) override {
    if (prefill_server_) {
        return prefill_server_->PdKvWriteback(context, request, response);
    }
    if (decode_server_) {
        return decode_server_->PdKvWritebackSend(context, request, response);
    }
    auto error_msg = "server not implement PdKvWriteback";
    RTP_LLM_LOG_ERROR(error_msg);
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, error_msg);
}
```

- [ ] **Step 4: Declare decode send handler**

Modify `rtp_llm/cpp/model_rpc/DecodeRpcServer.h`:

```cpp
grpc::Status PdKvWritebackSend(grpc::ServerContext*          server_context,
                               const PdKvWritebackRequestPB* request,
                               PdKvWritebackResponsePB*      response);
```

- [ ] **Step 5: Add shared PB conversion helper and implement decode send handler**

Create `rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h`:

```cpp
#pragma once

#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

PdKvWritebackLaunchRequest pdKvWritebackLaunchRequestFromPB(const PdKvWritebackRequestPB& pb);

}  // namespace rtp_llm
```

Create `rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.cc`:

```cpp
#include "rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h"

namespace rtp_llm {
namespace {

PdKvWritebackCompatibility pdKvWritebackCompatibilityFromPB(const PdKvWritebackCompatibilityPB& pb) {
    PdKvWritebackCompatibility compatibility;
    compatibility.seq_size_per_block = pb.seq_size_per_block();
    compatibility.layer_count        = pb.layer_count();
    compatibility.group_count        = pb.group_count();
    compatibility.partition_count    = pb.partition_count();
    compatibility.layer_to_group_id.assign(pb.layer_to_group_id().begin(), pb.layer_to_group_id().end());
    compatibility.group_types.assign(pb.group_types().begin(), pb.group_types().end());
    return compatibility;
}

}  // namespace

PdKvWritebackLaunchRequest pdKvWritebackLaunchRequestFromPB(const PdKvWritebackRequestPB& pb) {
    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = pb.request_id();
    request.manifest.request_key          = pb.request_key();
    request.manifest.final_token_count    = pb.final_token_count();
    request.manifest.reusable_block_count = pb.reusable_block_count();
    request.manifest.cache_keys.assign(pb.cache_keys().begin(), pb.cache_keys().end());
    request.manifest.group_block_ids.reserve(pb.group_block_ids_size());
    for (const auto& group_pb : pb.group_block_ids()) {
        request.manifest.group_block_ids.emplace_back(group_pb.block_ids().begin(), group_pb.block_ids().end());
    }
    request.source      = pdKvWritebackCompatibilityFromPB(pb.source());
    request.destination = pdKvWritebackCompatibilityFromPB(pb.destination());
    request.decode_worker_addrs.assign(pb.decode_worker_addrs().begin(), pb.decode_worker_addrs().end());
    request.prefill_worker_addrs.assign(pb.prefill_worker_addrs().begin(), pb.prefill_worker_addrs().end());
    request.deadline_ms = pb.deadline_us() > 0 ? pb.deadline_us() / 1000 : 0;
    return request;
}

}  // namespace rtp_llm
```

Remove the anonymous `convertPdKvWritebackCompatibility` and `convertPdKvWritebackRequest` helpers from `PrefillRpcServer.cc`, include the shared helper, and replace the conversion call:

```cpp
#include "rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h"

auto launch_request = pdKvWritebackLaunchRequestFromPB(*request);
```

Add the same include to `DecodeRpcServer.cc`. The handler body should be:

```cpp
grpc::Status DecodeRpcServer::PdKvWritebackSend(grpc::ServerContext*          server_context,
                                                const PdKvWritebackRequestPB* request,
                                                PdKvWritebackResponsePB*      response) {
    (void)server_context;
    auto cache_manager = engine_->resourceContext().cache_manager;
    if (!cache_manager || !cache_manager->writebackManager()) {
        response->mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
        response->mutable_error_info()->set_error_message("writeback_manager is null");
        response->set_accepted(false);
        response->set_reason("writeback_manager is null");
        return grpc::Status::OK;
    }

    auto launch_request = pdKvWritebackLaunchRequestFromPB(*request);
    auto source_resource = std::make_shared<BatchKVCacheResource>();
    source_resource->resetBatchSize(1);
    source_resource->initBatchGroups(0,
                                     launch_request.source.group_count,
                                     launch_request.source.layer_count,
                                     launch_request.source.layer_to_group_id);
    for (int group_id = 0; group_id < launch_request.manifest.group_block_ids.size(); ++group_id) {
        source_resource->setBatchBlocks(0, group_id, launch_request.manifest.group_block_ids[group_id]);
    }
    source_resource->setBatchCacheKeys(0, launch_request.manifest.cache_keys);

    auto held_resource = cache_manager->incrKVCacheRef(source_resource->cacheResource(0),
                                                       launch_request.manifest.cache_keys,
                                                       true);
    if (!held_resource) {
        response->mutable_error_info()->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
        response->mutable_error_info()->set_error_message("hold source cache failed");
        response->set_accepted(false);
        response->set_reason("hold_failed");
        return grpc::Status::OK;
    }
    launch_request.held_resource = std::move(held_resource);

    auto status = cache_manager->writebackManager()->sendOnDecode(launch_request, source_resource);
    response->mutable_error_info()->set_error_code(status.ok() ? ErrorCodePB::NONE_ERROR : ErrorCodePB::UNKNOWN_ERROR);
    response->mutable_error_info()->set_error_message(status.ok() ? "" : std::string(status.message()));
    response->set_accepted(status.ok());
    response->set_reason(status.ok() ? "accepted" : std::string(status.message()));
    return grpc::Status::OK;
}
```

- [ ] **Step 6: Run manager test and model RPC compile target**

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
bazel build //rtp_llm/cpp/model_rpc:model_rpc_server
```

Expected: PASS.

- [ ] **Step 7: Commit decode send handler**

Run:

```bash
git add rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h \
        rtp_llm/cpp/model_rpc/DecodeRpcServer.h \
        rtp_llm/cpp/model_rpc/DecodeRpcServer.cc \
        rtp_llm/cpp/model_rpc/PrefillRpcServer.cc \
        rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h \
        rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.cc \
        rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h \
        rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc \
        rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc
git commit -m "feat: handle decode-side PD KV writeback send"
```

## Task 6: Add Receive Metrics and Failure Cleanup Assertions

**Files:**
- Modify: `rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc`
- Modify: `rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc`

- [ ] **Step 1: Add receive failure cleanup test**

Add to `PdKvWritebackManagerTest.cc`:

```cpp
class RecordingCacheWriter: public PdKvWritebackCacheWriter {
public:
    absl::Status mallocWritebackBlocks(const BatchKVCacheResourcePtr& resource, size_t block_count) override {
        ++malloc_calls;
        malloc_block_count = block_count;
        return malloc_status;
    }
    void commitWritebackBlocks(const BatchKVCacheResourcePtr& resource,
                               const CacheKeysType& keys,
                               bool is_resident) override {
        ++commit_calls;
        committed_keys = keys;
        committed_resident = is_resident;
    }
    void freeWritebackBlocks(const BatchKVCacheResourcePtr& resource) override {
        ++free_calls;
    }

    int malloc_calls = 0;
    int commit_calls = 0;
    int free_calls = 0;
    size_t malloc_block_count = 0;
    bool committed_resident = true;
    CacheKeysType committed_keys;
    absl::Status malloc_status = absl::OkStatus();
};

TEST(PdKvWritebackManagerTest, ReceiveOnPrefillFreesWithoutCommitWhenTransferFails) {
    PDSepConfig config;
    config.enable_pd_kv_cache_writeback = true;
    RecordingCacheWriter writer;
    auto transfer_client = std::make_shared<RecordingTransferClient>();
    transfer_client->transfer_status = absl::InternalError("transfer failed");
    PdKvWritebackManager manager(config, &writer, transfer_client);

    PdKvWritebackLaunchRequest request = makeValidSingleRankWritebackRequestForTest();
    auto destination = std::make_shared<BatchKVCacheResource>();

    auto status = manager.receiveOnPrefill(request, destination);
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(writer.malloc_calls, 1);
    EXPECT_EQ(writer.commit_calls, 0);
    EXPECT_EQ(writer.free_calls, 1);
}
```

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: PASS if existing cleanup behavior is preserved; FAIL if helper names need alignment.

- [ ] **Step 2: Add receive metrics timing**

Modify `receiveOnPrefill` to measure and report:

```cpp
const int64_t receive_begin_us = currentTimeUs();
PdKvWritebackMetricsCollector collector;
collector.receive_qps = true;
collector.block_count = request.manifest.reusable_block_count;
collector.token_count = request.manifest.final_token_count;

const int64_t malloc_begin_us = currentTimeUs();
auto status = cache_writer_->mallocWritebackBlocks(destination_resource,
                                                   static_cast<size_t>(request.manifest.reusable_block_count));
collector.malloc_latency_us = currentTimeUs() - malloc_begin_us;
if (!status.ok()) {
    collector.receive_failed_qps = true;
    collector.receive_latency_us = currentTimeUs() - receive_begin_us;
    reportPdKvWritebackMetric(metrics_reporter_, collector, "prefill_receive", "failed", "malloc_failed",
                              "prefill", request.destination.partition_count, "tp_equal");
    return status;
}

auto transfer_status = transfer_client_->transfer(buildTransferPlan(request, destination_resource));
if (!transfer_status.ok()) {
    cache_writer_->freeWritebackBlocks(destination_resource);
    collector.receive_failed_qps = true;
    collector.receive_latency_us = currentTimeUs() - receive_begin_us;
    reportPdKvWritebackMetric(metrics_reporter_, collector, "prefill_receive", "failed", "transfer_failed",
                              "prefill", request.destination.partition_count, "tp_equal");
    return transfer_status;
}

const int64_t commit_begin_us = currentTimeUs();
cache_writer_->commitWritebackBlocks(destination_resource, request.manifest.cache_keys, false);
collector.commit_latency_us = currentTimeUs() - commit_begin_us;
cache_writer_->freeWritebackBlocks(destination_resource);
collector.receive_latency_us = currentTimeUs() - receive_begin_us;
reportPdKvWritebackMetric(metrics_reporter_, collector, "prefill_receive", "success", "ok",
                          "prefill", request.destination.partition_count, "tp_equal");
return absl::OkStatus();
```

- [ ] **Step 3: Run manager tests**

Run:

```bash
bazel test //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager
```

Expected: PASS.

- [ ] **Step 4: Commit receive metrics**

Run:

```bash
git add rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc \
        rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc
git commit -m "feat: report PD KV writeback receive metrics"
```

## Task 7: Add TP4 qwen3 Smoke Case

**Files:**
- Modify: `rtp_llm/test/smoke/suites_h20_oss.bzl`
- Modify: `rtp_llm/test/smoke/test/pd_writeback_smoke_config_static_test.py`

- [ ] **Step 1: Add static test for TP4 smoke config**

Modify `rtp_llm/test/smoke/test/pd_writeback_smoke_config_static_test.py`:

```python
def test_qwen3_pd_writeback_tp4_case_exists(self):
    suite = SUITE.read_text()
    self.assertIn('name="dense_pd_writeback_reuse_tp4"', suite)
    case_block = _case_block(suite, "dense_pd_writeback_reuse_tp4")
    self.assertIn("--tp_size 4 --world_size 4", case_block)
    self.assertIn("ENABLE_PD_KV_CACHE_WRITEBACK=1", case_block)
    task = json.loads(TASK_INFO.read_text())
    second_assertions = task["query_result"][1]["aux_info_assertions"]
    self.assertEqual(second_assertions["mode"], "aux_info_only")
    self.assertIn("aux_info.prefill_local_reuse_len", second_assertions["fields"])
```

Run:

```bash
bazel test //rtp_llm/test/smoke/test:pd_writeback_smoke_config_static_test
```

Expected: FAIL because the TP4 case is not in the suite.

- [ ] **Step 2: Add qwen3 TP4 smoke suite entry**

Modify `rtp_llm/test/smoke/suites_h20_oss.bzl` next to `dense_pd_writeback_reuse`:

```python
            smoke_test(
                name="dense_pd_writeback_reuse_tp4",
                task_info="data/model/qwen3/q_r_h20_pd_writeback_reuse.json",
                envs={
                    "prefill": ["ENABLE_PD_KV_CACHE_WRITEBACK=1"],
                    "decode": ["ENABLE_PD_KV_CACHE_WRITEBACK=1"],
                },
                smoke_args={
                    "prefill": "--act_type BF16 --reserver_runtime_mem_mb 8192 --tp_size 4 --world_size 4 --warm_up 0 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --seq_size_per_block 64 --reuse_cache 1 --load_cache_timeout_ms 120000",
                    "decode": "--act_type BF16 --reserver_runtime_mem_mb 8192 --tp_size 4 --world_size 4 --warm_up 0 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --seq_size_per_block 64 --reuse_cache 1 --load_cache_timeout_ms 120000",
                },
                sleep_time_qr=10,
                gpu_type=["H20"],
            ),
```

- [ ] **Step 3: Run static smoke config test**

Run:

```bash
bazel test //rtp_llm/test/smoke/test:pd_writeback_smoke_config_static_test
```

Expected: PASS.

- [ ] **Step 4: Run TP1 smoke regression**

Run:

```bash
bazel test //rtp_llm/test/smoke:dense_pd_writeback_reuse
```

Expected: PASS. This preserves the already-working TP1 behavior.

- [ ] **Step 5: Run TP4 smoke**

Run:

```bash
bazel test //rtp_llm/test/smoke:dense_pd_writeback_reuse_tp4
```

Expected: PASS. The second request must satisfy aux info assertions:

```json
{
  "aux_info.pd_sep": {"eq": true},
  "aux_info.prefill_local_reuse_len": {"ge": 64}
}
```

- [ ] **Step 6: Commit smoke case**

Run:

```bash
git add rtp_llm/test/smoke/suites_h20_oss.bzl \
        rtp_llm/test/smoke/test/pd_writeback_smoke_config_static_test.py
git commit -m "test: add TP4 PD KV writeback smoke"
```

## Task 8: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused C++ unit/static suite**

Run:

```bash
bazel test \
  //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_topology \
  //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manager \
  //rtp_llm/cpp/cache/writeback:test_pd_kv_writeback_manifest \
  //rtp_llm/cpp/cache/connector/test:pd_kv_writeback_coordinator_static_test \
  //rtp_llm/test/smoke/test:pd_writeback_smoke_config_static_test
```

Expected: PASS.

- [ ] **Step 2: Run smoke verification**

Run:

```bash
bazel test //rtp_llm/test/smoke:dense_pd_writeback_reuse
bazel test //rtp_llm/test/smoke:dense_pd_writeback_reuse_tp4
```

Expected: both PASS.

- [ ] **Step 3: Run formatting and diff checks**

Run:

```bash
git diff --check
pre-commit run --files \
  rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h \
  rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.cc \
  rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h \
  rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.cc \
  rtp_llm/cpp/cache/writeback/PdKvWritebackManager.h \
  rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc \
  rtp_llm/cpp/cache/writeback/test/PdKvWritebackTopologyTest.cc \
  rtp_llm/cpp/cache/writeback/test/PdKvWritebackManagerTest.cc \
  rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.cc \
  rtp_llm/cpp/cache/connector/test/PdKvWritebackCoordinatorStaticTest.py \
  rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h \
  rtp_llm/cpp/model_rpc/DecodeRpcServer.h \
  rtp_llm/cpp/model_rpc/DecodeRpcServer.cc \
  rtp_llm/cpp/model_rpc/PrefillRpcServer.cc \
  rtp_llm/test/smoke/suites_h20_oss.bzl \
  rtp_llm/test/smoke/test/pd_writeback_smoke_config_static_test.py
```

Expected: PASS. Use `--files` because the worktree currently contains unrelated dirty binary test data.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional branch changes plus the pre-existing unrelated dirty files. The unrelated dirty files must not be staged.

- [ ] **Step 5: Push branch after verification**

Run:

```bash
git push yzyDavid HEAD:feature/qwen35vl-yzy-pd-kv-writeback
```

Expected: push succeeds to the existing `yzyDavid` remote for `feature/qwen35vl-yzy-pd-kv-writeback`.

## Plan Self-Review

- Spec coverage:
  - TP equal topology: Task 1.
  - All-rank fanout: Tasks 3 and 4.
  - Decode local lifecycle hold: Task 5.
  - Prefill receive commit/free semantics: Task 6.
  - Environment gate: preserved in Task 3 launch checks.
  - Dedicated monitoring: Task 2 and Task 6.
  - TP4 smoke aux-info proof: Task 7.
- Placeholder scan:
  - No unnamed future work remains.
- Type consistency:
  - `PdKvWritebackTopologyPlan` is introduced before manager and RPC signatures use it.
  - `sendOnDecode` accepts `PdKvWritebackLaunchRequest` and `BatchKVCacheResourcePtr` consistently in manager and decode RPC tasks.
  - Metrics collector and report helper names are consistent across tasks.
