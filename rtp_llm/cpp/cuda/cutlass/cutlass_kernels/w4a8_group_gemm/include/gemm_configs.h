#pragma once

// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 64, CTA_K = 256
    CtaShape16x64x256_WarpShape16x16x256,
    CtaShape32x64x256_WarpShape16x32x256,

    // TensorCore configs CTA_N = 32, CTA_K = 256
    CtaShape32x32x256_WarpShape16x16x256,

    // TensorCore configs CTA_N = 128, CTA_K = 128
    CtaShape32x128x128_WarpShape32x32x128,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    CtaShape16x256x64_WarpShape16x64x64,
    CtaShape16x256x128_WarpShape16x64x128,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,
    CtaShape128x256x128_WarpShape64x64x128,

    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,
    CtaShape256x128x128_WarpShape64x64x128,
};

enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    // SPLIT_K_PARALLEL // Not supported yet
};

enum class CutlassTileConfigSM90 {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // CTA configs for M=64
    CtaShape64x16x128B,
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,

    // CTA configs for M=128
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,

    CtaShape128x16x256B,

    // CTA configs for M=128
    CtaShape256x16x128B,
    CtaShape256x32x128B,
    CtaShape256x64x128B,
    CtaShape256x128x128B,
};

enum class MainloopScheduleType {
    AUTO  // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
          // defaults to the "legacy" main loop schedule.
};

enum class EpilogueScheduleType {
    AUTO  // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
          // architectures older than hopper, the epilogue is always performed by the same thread block as the main
          // loop.
};

enum class ClusterShape {
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1,
    ClusterShape_1x8x1,
    ClusterShape_8x1x1
};

struct CutlassGemmConfig {
    enum CandidateConfigTypeParam : int {
        NONE         = 0,
        WEIGHT_ONLY  = 1u << 0,
        SIMT_ONLY    = 1u << 1,
        INT8_ONLY    = 1u << 2,
        HOPPER       = 1u << 3,
        GROUPED_GEMM = 1u << 4,
        FP8_ONLY     = 1u << 5,
    };

    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle       split_k_style  = SplitKStyle::NO_SPLIT_K;
    int               split_k_factor = -1;
    int               stages         = -1;

    CutlassTileConfigSM90 tile_config_sm90  = CutlassTileConfigSM90::ChooseWithHeuristic;
    MainloopScheduleType  mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType  epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape          cluster_shape     = ClusterShape::ClusterShape_1x1x1;
    bool                  is_sm90           = false;

    CutlassGemmConfig() = default;

    CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages):
        tile_config(tile_config), split_k_style(split_k_style), split_k_factor(split_k_factor), stages(stages) {}

    CutlassGemmConfig(CutlassTileConfig tile_config, int split_k_factor, int stages):
        tile_config(tile_config),
        split_k_style(split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K),
        split_k_factor(split_k_factor),
        stages(stages) {}

    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90,
                      MainloopScheduleType  mainloop_schedule,
                      EpilogueScheduleType  epilogue_schedule,
                      ClusterShape          cluster_shape):
        tile_config_sm90(tile_config_sm90),
        mainloop_schedule(mainloop_schedule),
        epilogue_schedule(epilogue_schedule),
        cluster_shape(cluster_shape),
        is_sm90(true) {}

    bool operator==(CutlassGemmConfig const& r) const {
        return tile_config == r.tile_config && split_k_style == r.split_k_style && split_k_factor == r.split_k_factor
               && stages == r.stages;
    }
};
