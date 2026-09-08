// Microbenchmark + correctness harness for the radix topk paths behind
// invokeTopkLastDim (multi-block vs one-block, selected via force_path).
//
// Usage:
//   topk_bench                                  run matrix + semantic checks
//   topk_bench --check                          run only the semantic checks (fast; used by cc_test)
//   topk_bench BATCH LEN K SORTED PATH [REPS]   time one combo
//     SORTED: 0/1 (1 = trailing by-value sort, production stage-2 behavior)
//     PATH:   0 = auto, 1 = force multi-block, 2 = force one-block
//
// Part A (routing matrix): shapes x ks x sorted variants x {multi, one-block},
//   prints CSV rows and checks both paths are bitwise identical per combo.
// Part B (semantic checks): compares invokeTopkLastDim output against a host
//   reference (value desc, index asc) — covers the ROCm k<=256 WarpSort gate
//   and the radix path directly.
// Matrix mode exits non-zero if any comparison fails (single-combo mode always
// returns 0).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

#include "3rdparty/trt_beam_search/common.h"
#include "3rdparty/trt_beam_search/topkLastDim.h"

namespace
{

using tensorrt_llm::kernels::invokeComputeTopkLastDimWorkspaceSize;
using tensorrt_llm::kernels::invokeTopkLastDim;

// check_cuda_error() is debug-gated (syncAndCheckInDebug), so it is a no-op in
// normal bench runs; check every CUDA API return code explicitly instead.
char const* benchErrString(cudaError_t err)
{
#if USING_ROCM
    return hipGetErrorString(err);
#else
    return cudaGetErrorString(err);
#endif
}

#define BENCH_CHECK(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const bench_err_ = (call);                                                                         \
        if (bench_err_ != cudaSuccess)                                                                                 \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", benchErrString(bench_err_), __FILE__, __LINE__);          \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// Logprob-like values in [-20, 0]; every 7919th element pinned to create
// exact-equal ties so the tie-break path is exercised. The pinned value must sit
// above the top-k threshold or the ties are never selected: for the matrix shapes
// the threshold ranges from about -0.02 (k=50) to -1.25 (k=4096), and -0.05 is
// inside the top-k for every shape with k >= 200.
void fillInput(std::vector<float>& host, int batch, int len)
{
    host.resize(static_cast<size_t>(batch) * len);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-20.f, 0.f);
    for (auto& v : host)
    {
        v = dist(rng);
    }
    for (size_t i = 0; i < host.size(); i += 7919)
    {
        host[i] = -0.05f;
    }
}

float* toDevice(std::vector<float> const& host)
{
    float* dev = nullptr;
    BENCH_CHECK(cudaMalloc(&dev, host.size() * sizeof(float)));
    BENCH_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

// Returns microseconds per call. dOutVal/dOutIdx receive the last call's output.
double timePath(int batch, int len, int k, bool sorted, int forcePath, float const* dIn, float* dOutVal,
    int* dOutIdx, void* workspace, int reps, cudaStream_t stream)
{
    std::optional<float> mask(-std::numeric_limits<float>::infinity());  // same as beam search
    for (int i = 0; i < 3; ++i)
    {
        invokeTopkLastDim<float>(batch, len, k, true, mask, dIn, dOutVal, dOutIdx, workspace, stream, sorted, forcePath);
    }
    BENCH_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    BENCH_CHECK(cudaEventCreate(&start));
    BENCH_CHECK(cudaEventCreate(&stop));
    BENCH_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < reps; ++i)
    {
        invokeTopkLastDim<float>(batch, len, k, true, mask, dIn, dOutVal, dOutIdx, workspace, stream, sorted, forcePath);
    }
    BENCH_CHECK(cudaEventRecord(stop, stream));
    BENCH_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    BENCH_CHECK(cudaEventElapsedTime(&ms, start, stop));
    BENCH_CHECK(cudaEventDestroy(start));
    BENCH_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) * 1000.0 / reps;
}

struct PathOutput
{
    double us;
    std::vector<float> vals;
    std::vector<int> idxs;
};

PathOutput runPath(int batch, int len, int k, bool sorted, int forcePath, float const* dIn, int reps, cudaStream_t stream)
{
    size_t const wsBytes = invokeComputeTopkLastDimWorkspaceSize<float>(batch, len, k, true, forcePath);
    void* workspace = nullptr;
    BENCH_CHECK(cudaMalloc(&workspace, wsBytes));
    float* dOutVal = nullptr;
    int* dOutIdx = nullptr;
    BENCH_CHECK(cudaMalloc(&dOutVal, sizeof(float) * k * batch));
    BENCH_CHECK(cudaMalloc(&dOutIdx, sizeof(int) * k * batch));

    PathOutput out;
    out.us = timePath(batch, len, k, sorted, forcePath, dIn, dOutVal, dOutIdx, workspace, reps, stream);
    out.vals.resize(static_cast<size_t>(k) * batch);
    out.idxs.resize(static_cast<size_t>(k) * batch);
    BENCH_CHECK(cudaMemcpy(out.vals.data(), dOutVal, out.vals.size() * sizeof(float), cudaMemcpyDeviceToHost));
    BENCH_CHECK(cudaMemcpy(out.idxs.data(), dOutIdx, out.idxs.size() * sizeof(int), cudaMemcpyDeviceToHost));

    BENCH_CHECK(cudaFree(workspace));
    BENCH_CHECK(cudaFree(dOutVal));
    BENCH_CHECK(cudaFree(dOutIdx));
    return out;
}

bool sameOutput(PathOutput const& a, PathOutput const& b)
{
    return a.vals.size() == b.vals.size() && 0 == std::memcmp(a.vals.data(), b.vals.data(), a.vals.size() * sizeof(float))
        && 0 == std::memcmp(a.idxs.data(), b.idxs.data(), a.idxs.size() * sizeof(int));
}

// Host reference: per row, top-k by (value desc, index asc). indexAscending=true
// returns the same set ordered by index (the sorted=false production layout).
void hostTopk(std::vector<float> const& in, int batch, int len, int k, bool indexAscending,
    std::vector<float>& refVals, std::vector<int>& refIdxs)
{
    refVals.assign(static_cast<size_t>(k) * batch, 0.f);
    refIdxs.assign(static_cast<size_t>(k) * batch, 0);
    std::vector<int> ord(len);
    for (int b = 0; b < batch; ++b)
    {
        std::iota(ord.begin(), ord.end(), 0);
        float const* row = in.data() + static_cast<size_t>(b) * len;
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
            [&](int i, int j) { return row[i] > row[j] || (row[i] == row[j] && i < j); });
        if (indexAscending)
        {
            std::sort(ord.begin(), ord.begin() + k);
        }
        for (int i = 0; i < k; ++i)
        {
            refIdxs[static_cast<size_t>(b) * k + i] = ord[i];
            refVals[static_cast<size_t>(b) * k + i] = row[ord[i]];
        }
    }
}

// Part B: one semantic check of invokeTopkLastDim against the host reference.
// strictOrder=true: exact match on values and index order (radix path semantics).
// strictOrder=false (WarpSort): exact match on values; for indices, exact set
// equality strictly above the k-th value and tie-count equality within the
// boundary group (WarpSort's boundary-tie pick among equal values is
// path-specific by design). Tie-order divergence is reported as informational.
bool checkVsHost(int batch, int len, int k, bool sorted, std::vector<float> const& hostIn, float const* dIn,
    cudaStream_t stream, char const* label, bool strictOrder)
{
    PathOutput dev = runPath(batch, len, k, sorted, /*forcePath=*/0, dIn, /*reps=*/3, stream);
    std::vector<float> refVals;
    std::vector<int> refIdxs;
    hostTopk(hostIn, batch, len, k, /*indexAscending=*/!sorted, refVals, refIdxs);
    bool const valuesOk
        = 0 == std::memcmp(dev.vals.data(), refVals.data(), refVals.size() * sizeof(float));
    bool ok;
    if (strictOrder)
    {
        ok = valuesOk
            && 0 == std::memcmp(dev.idxs.data(), refIdxs.data(), refIdxs.size() * sizeof(int));
    }
    else
    {
        ok = valuesOk;
        int tieRows = 0;
        for (int b = 0; b < batch && ok; ++b)
        {
            size_t const off = static_cast<size_t>(b) * k;
            // Boundary-tie aware comparison: elements strictly above the k-th value
            // must have identical index sets; within the boundary tie group only the
            // counts must match — WarpSort's pick among exactly-equal boundary values
            // is documented as path-specific (see the gate comment in topkLastDim.cu).
            float const thr = refVals[off + k - 1];
            std::vector<int> devCore, refCore;
            int devTies = 0, refTies = 0;
            bool orderDiff = false;
            for (int i = 0; i < k; ++i)
            {
                float const dv = dev.vals[off + i], rv = refVals[off + i];
                if (dv > thr)
                {
                    devCore.push_back(dev.idxs[off + i]);
                }
                else if (dv == thr)
                {
                    ++devTies;
                }
                if (rv > thr)
                {
                    refCore.push_back(refIdxs[off + i]);
                }
                else if (rv == thr)
                {
                    ++refTies;
                }
                if (dev.idxs[off + i] != refIdxs[off + i])
                {
                    orderDiff = true;
                }
            }
            if (orderDiff)
            {
                ++tieRows;
            }
            std::sort(devCore.begin(), devCore.end());
            std::sort(refCore.begin(), refCore.end());
            if (devCore != refCore || devTies != refTies)
            {
                ok = false;
            }
        }
        std::printf("check[%s]: batch=%d len=%d k=%d sorted=%d -> %s (values %s, rows with tie-order diff %d)\n",
            label, batch, len, k, sorted ? 1 : 0, ok ? "PASS" : "FAIL", valuesOk ? "exact" : "MISMATCH", tieRows);
        return ok;
    }
    std::printf("check[%s]: batch=%d len=%d k=%d sorted=%d -> %s\n", label, batch, len, k, sorted ? 1 : 0,
        ok ? "PASS" : "FAIL");
    if (!ok)
    {
        for (size_t i = 0; i < refIdxs.size(); ++i)
        {
            if (dev.idxs[i] != refIdxs[i] || dev.vals[i] != refVals[i])
            {
                std::printf("  first mismatch at i=%zu: dev(idx=%d val=%f) ref(idx=%d val=%f)\n", i, dev.idxs[i],
                    dev.vals[i], refIdxs[i], refVals[i]);
                break;
            }
        }
    }
    return ok;
}

struct Shape
{
    // cls: 'A' production stage-1, 'C' production stage-2, 'X' crossover probe,
    //      'B' beam-4096/8192 spot check.
    char cls;
    int batch;
    int len;
    char const* note;
    int k = 0;  // 0 = use the class default k list
};

// Part B: semantic checks vs host reference. Returns the number of failures.
int runSemanticChecks()
{
    int failures = 0;
    cudaStream_t stream;
    BENCH_CHECK(cudaStreamCreate(&stream));
    std::vector<float> hostIn;
    float* dIn = nullptr;
    auto prep = [&](int batch, int len) {
        if (dIn)
        {
            BENCH_CHECK(cudaFree(dIn));
        }
        fillInput(hostIn, batch, len);
        dIn = toDevice(hostIn);
    };
    // ROCm k<=256 gate (WarpSort path, pre-existing behavior for beams <=256).
    prep(50, 65660);
    failures += checkVsHost(50, 65660, 200, true, hostIn, dIn, stream, "warpsort-A", false) ? 0 : 1;
    prep(1, 75000);
    failures += checkVsHost(1, 75000, 200, true, hostIn, dIn, stream, "warpsort-C", false) ? 0 : 1;
    // k=400: beams 257..512 stay on the radix path after the gate narrowed to 256.
    prep(50, 217303);
    failures += checkVsHost(50, 217303, 400, true, hostIn, dIn, stream, "radix-k400-fullvocab", true) ? 0 : 1;
    // Radix path vs reference (k>512): sorted=true layout and sorted=false layout.
    prep(50, 65660);
    failures += checkVsHost(50, 65660, 1500, true, hostIn, dIn, stream, "radix-sorted", true) ? 0 : 1;
    failures += checkVsHost(50, 65660, 1500, false, hostIn, dIn, stream, "radix-unsorted", true) ? 0 : 1;
    // Gate boundary through the production (auto) path: k=256 takes WarpSort,
    // k=257 takes radix.
    prep(50, 65660);
    failures += checkVsHost(50, 65660, 256, true, hostIn, dIn, stream, "gate-k256-warpsort", false) ? 0 : 1;
    failures += checkVsHost(50, 65660, 257, true, hostIn, dIn, stream, "gate-k257-radix", true) ? 0 : 1;
    // step1 C-stage edge: k == len (single beam expanded once).
    prep(1, 50);
    failures += checkVsHost(1, 50, 50, true, hostIn, dIn, stream, "warpsort-C-step1-k-eq-len", false) ? 0 : 1;
    if (dIn)
    {
        BENCH_CHECK(cudaFree(dIn));
        dIn = nullptr;
    }

    // Dual-path consistency: the routing change's premise is "same result on either
    // radix path". The full matrix checks this but only runs manually; these three
    // shapes put it under --check/CI, covering the ROCm crossover band (grid 11/16)
    // and the grid_dim<2 clamp (batch=1024 computes to grid_dim 1).
    struct DualCase
    {
        int batch;
        int len;
        int k;
        char const* label;
    };
    DualCase const dualCases[] = {
        {16, 65660, 1500, "dual-grid11"},
        {8, 32000, 1500, "dual-grid16"},
        {1024, 217303, 1024, "dual-grid1-clamp"},
    };
    for (auto const& c : dualCases)
    {
        fillInput(hostIn, c.batch, c.len);
        float* dDual = toDevice(hostIn);
        PathOutput multi = runPath(c.batch, c.len, c.k, /*sorted=*/true, /*forcePath=*/1, dDual, /*reps=*/3, stream);
        PathOutput oneblk = runPath(c.batch, c.len, c.k, /*sorted=*/true, /*forcePath=*/2, dDual, /*reps=*/3, stream);
        bool const ok = sameOutput(multi, oneblk);
        std::printf("dual[%s]: batch=%d len=%d k=%d multi vs one-block -> %s\n", c.label, c.batch, c.len, c.k,
            ok ? "PASS" : "FAIL");
        failures += ok ? 0 : 1;
        BENCH_CHECK(cudaFree(dDual));
    }
    BENCH_CHECK(cudaStreamDestroy(stream));
    return failures;
}

int runMatrix(int reps)
{
    // (batch, len) shapes. grid_dim is a function of (batch, len) only; the x-*
    // probes land in the grid_dim 10..36 crossover region (approximate values
    // computed for gfx942 fp32, active_blocks=240).
    Shape const shapes[] = {
        // production shapes
        {'A', 50, 65660, "A-step2"},
        {'A', 1500, 65660, "A-steady"},
        {'C', 1, 150000, "C-2k"},
        {'C', 1, 75000, "C-1k"},
        {'C', 1, 16777216, "C-steady-4096"},
        // vocab variants
        {'A', 50, 217303, "A-fullvocab"},
        {'A', 50, 262144, "A-bigvocab"},
        {'A', 50, 32000, "A-smallvocab"},
        // crossover probes
        {'X', 16, 65660, "x-grid11"},
        {'X', 16, 32000, "x-grid15"},
        {'X', 8, 32000, "x-grid16"},
        {'X', 10, 217303, "x-grid22"},
        {'X', 8, 217303, "x-grid27"},
        {'X', 6, 131072, "x-grid32"},
        {'X', 10, 65660, "x-grid33"},
        {'X', 5, 217303, "x-grid36"},
        // beam-4096 (k) spot checks on the production A shapes (4096 = max supported
        // beam width; wider beams are not instantiated, see beamSearchKernels4096.cu).
        {'B', 50, 65660, "A-step2-beam4096", 4096},
        {'B', 1500, 65660, "A-steady-beam4096", 4096},
        // k=256/257: boundary of the ROCm k<=256 WarpSort gate.
        {'B', 50, 65660, "A-gate-k256", 256},
        {'B', 50, 65660, "A-gate-k257", 257},
        // grid_dim==1 (batch=1024): force multi-block clamps to one-block (multi is
        // invalid below 2 blocks); grid_dim==2 (batch=512): multi must stay valid.
        {'B', 1024, 217303, "A-grid1-clamp", 1024},
        {'B', 512, 217303, "A-grid2-multi", 1024},
        // step1 shape: single beam in (nBMIn=1), small k.
        {'B', 1, 65660, "A-step1", 50},
    };
    auto ksFor = [](Shape const& s) -> std::vector<int> {
        if (s.k != 0)
        {
            return {s.k};
        }
        switch (s.cls)
        {
        case 'C': return s.len > 1000000 ? std::vector<int>{4096} : std::vector<int>{1500, 3000};
        case 'X': return {1500};
        default: return {1500, 3000};
        }
    };
    auto sortedFor = [](Shape const& s) -> std::vector<bool> {
        return s.cls == 'C' ? std::vector<bool>{true} : std::vector<bool>{true, false};
    };

    cudaStream_t stream;
    BENCH_CHECK(cudaStreamCreate(&stream));
    std::printf("batch,len,k,sorted,note,multi_us,oneblock_us,speedup,identical\n");
    int failures = 0;

    auto runCombo = [&](Shape const& s, int k, bool sorted) {
        std::vector<float> hostIn;
        fillInput(hostIn, s.batch, s.len);
        float* dIn = toDevice(hostIn);
        PathOutput multi = runPath(s.batch, s.len, k, sorted, 1, dIn, reps, stream);
        PathOutput oneblk = runPath(s.batch, s.len, k, sorted, 2, dIn, reps, stream);
        bool const identical = sameOutput(multi, oneblk);
        if (!identical)
        {
            ++failures;
        }
        std::printf("%d,%d,%d,%d,%s,%.1f,%.1f,%.2fx,%s\n", s.batch, s.len, k, sorted ? 1 : 0, s.note, multi.us,
            oneblk.us, multi.us / oneblk.us, identical ? "yes" : "NO");
        BENCH_CHECK(cudaFree(dIn));
    };

    for (auto const& s : shapes)
    {
        for (int k : ksFor(s))
        {
            for (bool sorted : sortedFor(s))
            {
                runCombo(s, k, sorted);
            }
        }
    }

    BENCH_CHECK(cudaStreamDestroy(stream));
    failures += runSemanticChecks();
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        return runMatrix(20);
    }
    if (argc == 2 && 0 == std::strcmp(argv[1], "--check"))
    {
        return runSemanticChecks() == 0 ? 0 : 1;
    }
    if (argc < 6 || argc > 7)
    {
        std::fprintf(stderr, "usage: %s [--check] [BATCH LEN K SORTED PATH [REPS]]\n", argv[0]);
        return 2;
    }
    int const batch = std::atoi(argv[1]);
    int const len = std::atoi(argv[2]);
    int const k = std::atoi(argv[3]);
    bool const sorted = std::atoi(argv[4]) != 0;
    int const path = std::atoi(argv[5]);
    int const reps = argc > 6 ? std::atoi(argv[6]) : 20;

    cudaStream_t stream;
    BENCH_CHECK(cudaStreamCreate(&stream));
    std::vector<float> hostIn;
    fillInput(hostIn, batch, len);
    float* dIn = toDevice(hostIn);
    PathOutput out = runPath(batch, len, k, sorted, path, dIn, reps, stream);
    std::printf("batch=%d len=%d k=%d sorted=%d path=%d: %.1f us/call\n", batch, len, k, sorted ? 1 : 0, path, out.us);
    BENCH_CHECK(cudaFree(dIn));
    BENCH_CHECK(cudaStreamDestroy(stream));
    return 0;
}
