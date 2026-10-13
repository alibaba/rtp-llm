// Reuse the exact layouts, CPU oracle, guard checks and synchronous copy harness.
// Only the CRC backend is substituted; the ordinary five-path entry is unused.
#include "rtp_llm/models_py/bindings/cuda/test/ForcedCrcBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"
#define CrcBlockCopyBatch ForcedCrcBlockCopyBatch
#define main ordinaryCrcCopyBenchmarkMain
#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmark.cu"
#undef main
#undef CrcBlockCopyBatch

namespace rtp_llm::crc_copy_benchmark {
namespace {
constexpr size_t kSegmentBytes = 16 * 1024;
const char*      abNames[]{"crc_whole", "crc_segmented"};

template<bool Segmented>
class MixedBackend: public ForcedCrcBlockCopyBatch {
public:
    MixedBackend(int device, size_t items, size_t payload, size_t tiles):
        ForcedCrcBlockCopyBatch(device, items, payload, tiles) {
        setSegmented(Segmented);
    }
};

size_t checkMessages(const Benchmark& benchmark, int blocks, bool segmented) {
    const size_t expected = size_t(blocks) * (segmented ? (benchmark.w.p + kSegmentBytes - 1) / kSegmentBytes : 1);
    const size_t actual   = benchmark.production.lastMessageCount();
    requireSync(actual == expected, "forced CRC policy did not produce the expected message count");
    return actual;
}

void abMetadata(std::ostream& os, const Options& options, const cudaDeviceProp& prop, int driver, int runtime) {
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    const auto provenance = [](const char* name) {
        const auto* value = std::getenv(name);
        return value ? value : "unavailable";
    };
    os << "{\"type\":\"metadata\",\"protocol\":\"crc_segmentation_ab_v1\",\"seed\":" << options.seed
       << ",\"source_commit\":\"" << provenance("CRC_BENCH_SOURCE_COMMIT") << "\",\"base_commit\":\""
       << provenance("CRC_BENCH_BASE_COMMIT") << "\",\"binary_sha256\":\"" << provenance("CRC_BENCH_BINARY_SHA256")
       << "\",\"production_base_commit\":\"c351ffa4f470f18c6c4499d0ba8d6155e1f07877\""
       << ",\"variants\":[\"crc_whole\",\"crc_segmented\"],\"repeat\":" << options.repeat
       << ",\"iterations\":" << options.iterations << ",\"warmup\":" << options.warmup << ",\"gpu_name\":\""
       << prop.name << "\",\"gpu\":\"" << prop.name << "\",\"gpu_uuid\":\"" << uuid.str()
       << "\",\"sm\":" << prop.major * 10 + prop.minor << ",\"driver\":" << driver << ",\"runtime\":" << runtime
       << ",\"l2_bytes\":" << prop.l2CacheSize
       << ",\"timing\":\"wall\",\"regime\":\"cold\",\"evict_multiplier\":" << kEvictMultiplier
       << ",\"max_backings\":32,\"segment_bytes\":" << kSegmentBytes
       << ",\"shared_workspace\":true,\"equal_workspace_bytes\":true,\"forced_mode_outside_timer\":true"
       << ",\"measured_h2d_verified\":true,\"correctness_only\":" << (options.correctnessOnly ? "true" : "false")
       << ",\"profiled\":false,\"block_counts\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]"
       << ",\"model\":\"DeepSeek-V4-Flash-0731\",\"target_layers\":43,\"kv_cache_sharded\":true"
       << ",\"seq_size_per_block\":128,\"kernel_tokens_per_block\":128,\"cp_size\":4,\"tp_size\":4"
       << ",\"cp_mode\":\"CP_RR\",\"gen_num_per_cycle\":0,\"local_backing_count\":true"
       << ",\"shape_source\":\"CacheConfigCreator+DeviceBlockPoolConfigHelper\""
       << ",\"boundary\":\"complete synchronous store/load: descriptors, checks, locks, metadata H2D, gather/scatter, payload DMA, nvCOMP, finish or combine, synchronization and verdict; policy setter and oracle excluded\""
       << ",\"host_input\":\"independent CPU whole-record oracle; same payload per paired round\""
       << ",\"h2d_host_payload\":\"recently prepared on CPU once per shared round; host cache is not claimed cold\""
       << ",\"policy\":\"forced whole or forced 16KiB segmentation for every BS; production threshold bypassed only in generated test backend\""
       << ",\"fallback_allowed\":false}\n";
}

void verifyCrossMode(Benchmark& benchmark, size_t ix) {
    auto&     w = benchmark.w;
    const int n = w.plans[ix]->n;
    benchmark.restoreDeviceOracle();
    benchmark.production.setSegmented(false);
    benchmark.verify(ix, 0, true);
    checkMessages(benchmark, n, false);
    std::vector<unsigned char> whole(w.host->bytes(), w.host->bytes() + kMaxBackings * w.hostStride);
    benchmark.production.setSegmented(true);
    benchmark.poisonDevice(ix);
    requireSync(benchmark.invoke(ix, 0, false) == CrcCopyStatus::OK, "segmented load of whole store failed");
    checkMessages(benchmark, n, true);
    benchmark.verifyDevice(true);
    benchmark.verify(ix, 0, true);
    checkMessages(benchmark, n, true);
    requireSync(std::equal(whole.begin(), whole.end(), w.host->bytes()), "whole and segmented records differ");
    benchmark.production.setSegmented(false);
    benchmark.poisonDevice(ix);
    requireSync(benchmark.invoke(ix, 0, false) == CrcCopyStatus::OK, "whole load of segmented store failed");
    checkMessages(benchmark, n, false);
    benchmark.verifyDevice(true);
}

int runSegmentationBenchmark(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);
    requireSync(!options.tileThreadProfile && !options.profileCapture && !options.exclude1dH2d,
                "segmentation benchmark does not accept ordinary benchmark profiling/exclusion modes");
    std::ofstream os(options.output);
    requireSync(bool(os), "cannot open output: " + options.output);
    os << std::setprecision(12);
    cu(cudaSetDevice(0));
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop, 0));
    requireSync(prop.l2CacheSize > 0, "known GPU L2 size is required");
    int driver = 0, runtime = 0;
    cu(cudaDriverGetVersion(&driver));
    cu(cudaRuntimeGetVersion(&runtime));
    requireSync(driver >= 13000 && runtime >= 13000 && ForcedCrcBlockCopyBatch::available(),
                "CUDA13 runtime, driver and CRC backend are required");
    abMetadata(os, options, prop, driver, runtime);
    os.flush();
    knownVector();
    mixedCrcSelfTest<MixedBackend<false>>();
    mixedCrcSelfTest<MixedBackend<true>>();
    for (const auto* variant : abNames)
        os << "{\"type\":\"mixed_crc_selftest\",\"variant\":\"" << variant << "\",\"success\":true}\n";
    std::mt19937 sourceRng(options.seed), orderRng(options.seed ^ 0x728193U);
    for (bool isFull : {true, false}) {
        // Both modes use this one backend allocation; no allocation/reset lies
        // between the paired calls. Segment capacity is preallocated for BS=32.
        Benchmark    benchmark(orderedLayout(isFull), prop.l2CacheSize);
        auto&        w              = benchmark.w;
        const size_t workspaceBytes = benchmark.production.deviceMemoryBytes();
        os << "{\"type\":\"layout\",\"layout\":\"" << w.layout.name << "\",\"payload_bytes\":" << w.p
           << ",\"encoded_bytes\":" << w.e << ",\"host_stride\":" << w.hostStride << ",\"staging_stride\":" << w.stride
           << ",\"tiles\":" << w.layout.sizes.size() << ",\"tile_bytes\":[";
        for (size_t t = 0; t < w.layout.sizes.size(); ++t) {
            if (t)
                os << ',';
            os << w.layout.sizes[t];
        }
        os << "],\"geometry\":" << w.layout.geometryJson() << ",\"pool_blocks\":" << w.poolBlocks
           << ",\"reserved_sentinel_blocks\":1"
           << ",\"source_bytes\":" << w.sourceBytes << ",\"evict_bytes\":" << w.evictBytes
           << ",\"workspace_device_bytes\":" << workspaceBytes << ",\"equal_workspace_bytes\":true"
           << ",\"whole_messages_per_backing\":1,\"segmented_messages_per_backing\":"
           << (w.p + kSegmentBytes - 1) / kSegmentBytes
           << ",\"host_pinned_verified\":true,\"source_device_verified\":true}\n";
        for (int mode : {0, 1}) {
            benchmark.production.setSegmented(mode == 1);
            benchmark.restoreDeviceOracle();
            for (bool store : {true, false}) {
                for (int n = 1; n <= kMaxBackings; ++n) {
                    for (int rotation : {0, 1}) {
                        benchmark.verify(w.planIndex(n, rotation), 0, store);
                        checkMessages(benchmark, n, mode == 1);
                    }
                    os << "{\"type\":\"correctness\",\"layout\":\"" << w.layout.name << "\",\"direction\":\""
                       << (store ? "d2h" : "h2d") << "\",\"blocks\":" << n << ",\"variant\":\"" << abNames[mode]
                       << "\",\"crc_messages\":" << benchmark.production.lastMessageCount()
                       << ",\"rotations_checked\":[0,1],\"success\":true}\n";
                }
            }
            for (int n : {1, 8, 16, 32}) {
                benchmark.verifyCorruption(w.planIndex(n, 1), 0);
                checkMessages(benchmark, n, mode == 1);
                os << "{\"type\":\"crc_corruption\",\"layout\":\"" << w.layout.name
                   << "\",\"direction\":\"h2d\",\"blocks\":" << n << ",\"variant\":\"" << abNames[mode]
                   << "\",\"cases\":6,\"all_targets_unchanged\":true,\"recovery_success\":true"
                   << ",\"good_bad_good\":true,\"entire_pool_guards\":true,\"success\":true}\n";
            }
        }
        for (int n : {1, 8, 16, 32}) {
            verifyCrossMode(benchmark, w.planIndex(n, 1));
            os << "{\"type\":\"cross_mode\",\"layout\":\"" << w.layout.name << "\",\"blocks\":" << n
               << ",\"whole_store_segmented_load\":true,\"segmented_store_whole_load\":true"
               << ",\"footer_equal\":true,\"success\":true}\n";
        }
        benchmark.restoreDeviceOracle();
        os.flush();
        if (options.correctnessOnly)
            continue;
        for (bool store : {true, false}) {
            for (int n = 1; n <= kMaxBackings; ++n) {
                std::vector<int> sources(w.rotationCount(n));
                std::iota(sources.begin(), sources.end(), 0);
                std::shuffle(sources.begin(), sources.end(), sourceRng);
                std::vector<int> order{0, 1};
                for (int round = -options.warmup; round < options.iterations; ++round) {
                    const int    source = sources[(round + options.warmup) % sources.size()];
                    const size_t ix     = w.planIndex(n, source);
                    if (!store)
                        benchmark.prepareHost(ix);
                    std::shuffle(order.begin(), order.end(), orderRng);
                    for (size_t position = 0; position < order.size(); ++position) {
                        const int mode = order[position];
                        benchmark.production.setSegmented(mode == 1);
                        if (!store)
                            benchmark.poisonDevice(ix);
                        w.flush(static_cast<unsigned>(round + options.warmup) * 2U + static_cast<unsigned>(position));
                        benchmark.primeCpuMetadata(ix);
                        const auto   begin  = std::chrono::steady_clock::now();
                        const auto   status = benchmark.invoke(ix, 0, store);
                        const double us =
                            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
                        requireSync(status == CrcCopyStatus::OK, "measured CRC call failed");
                        const size_t messages = checkMessages(benchmark, n, mode == 1);
                        if (!store)
                            benchmark.verifyDevice(true);
                        requireSync(benchmark.production.deviceMemoryBytes() == workspaceBytes,
                                    "CRC workspace changed between modes");
                        if (round >= 0)
                            os << "{\"type\":\"sample\",\"layout\":\"" << w.layout.name << "\",\"direction\":\""
                               << (store ? "d2h" : "h2d") << "\",\"blocks\":" << n << ",\"variant\":\"" << abNames[mode]
                               << "\",\"repeat\":" << options.repeat << ",\"round\":" << round
                               << ",\"position\":" << position << ",\"source_plan\":" << source
                               << ",\"crc_messages\":" << messages
                               << ",\"regime\":\"cold\",\"timing\":\"wall\",\"us\":" << us << "}\n";
                    }
                }
                os.flush();
            }
        }
    }
    os << "{\"type\":\"complete\",\"success\":true}\n";
    os.flush();
    requireSync(bool(os), "failed to write segmentation benchmark output");
    return 0;
}
}  // namespace
}  // namespace rtp_llm::crc_copy_benchmark

int main(int argc, char** argv) {
    try {
        return rtp_llm::crc_copy_benchmark::runSegmentationBenchmark(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "CRC segmentation benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
