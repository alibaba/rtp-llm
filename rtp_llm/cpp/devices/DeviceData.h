#pragma once
#include <cstdint>
#include <stddef.h>
#include <string>
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"

namespace rtp_llm {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
    ArmCpu = 3,
    ROCm = 4,
    Ppu = 5,
};

enum class MicroBatchType {
    NONE = 0,
    DS_PREFILL = 1,
    DS_DECODE = 2,
};

struct DeviceInitParams {
    DeviceType device_type;
    size_t device_id       = 0;
    size_t max_batch_size  = 256;
    size_t max_generate_batch_size = 128;

    size_t tp_rank         = 0;
    size_t tp_size         = 1;
    size_t ep_rank         = 0;
    size_t ep_size         = 1;
    size_t dp_rank         = 0;
    size_t dp_size         = 1;
    size_t ffn_tp_rank     = 0;
    size_t ffn_tp_size     = 1;
    bool use_all_gather    = false;

    // this ip:port pair should be unused, typically provided by gang,
    // to create temporary torch::TcpStore for exchanging communication id.
    // they are only needed when tp_size > 1.
    std::string master_ip          = "";
    int64_t     tp_master_port     = 0;
    int64_t     dp_tp_master_port  = 0;
    int64_t     ffn_tp_master_port = 0;

    // size (bytes) of device memory preallocated and managed by MemoryTracker.
    // negative value means reserving all free memory but remains abs(value) bytes.
    // 0 disables memory reservation
    int64_t device_reserve_memory_bytes = 0;
    int64_t host_reserve_memory_bytes   = 0;
    size_t tokens_per_block = 0;

    MlaOpsType mla_ops_type = MlaOpsType::AUTO;

    bool enable_comm_overlap = true;
    MicroBatchType enable_layer_micro_batch  = MicroBatchType::NONE;

    bool enable_sp = false;
    size_t overlap_math_sm_count = 0;
    size_t overlap_comm_type = 0;
    size_t m_split = 0;

    // to init deepep
    int64_t max_seq_len = 0;
    int64_t hidden_size = 0;
    int64_t num_experts = 0;
    int64_t extra_experts = 0;

    bool use_deepep_moe = false;
    bool use_deepep_internode = false;
    bool use_deepep_low_latency = false;
    bool is_mtp = false;
};

// immutable device properties. Can not change since device is initialized.
struct DeviceProperties {
    DeviceType type;
    size_t id = 0;

    /* -- distributed properties -- */
    size_t tp_rank = 0;
    size_t tp_size = 1;

    size_t dp_rank = 0;
    size_t dp_size = 1;
    size_t ffn_tp_rank = 0;
    size_t ffn_tp_size = 1;

    bool enable_sp = false;
    size_t overlap_math_sm_count = 0;
    size_t overlap_comm_type = 0;
    size_t m_split = 0;

    /* -- device implementation detail -- */
    // These two options are prepared for intel cpu device.
    // xfastertransformer fuses adding residual in their layer implementation.
    bool attn_fuse_add_residual = false;
    bool ffn_fuse_add_residual  = false;
    bool sq_fuse_bias_activation = false;

    bool enable_comm_overlap = true;
    MicroBatchType enable_layer_micro_batch = MicroBatchType::NONE;

    bool use_deepep_moe = false;
    bool use_deepep_internode = false;
    bool use_deepep_low_latency = false;
    bool is_mtp = false;
    bool use_all_gather = false;
};

struct MemoryStatus {
    size_t used_bytes          = 0;
    size_t free_bytes          = 0;
    size_t available_bytes     = 0; // free + preserved
    size_t allocated_bytes     = 0; // memory allocated via current device
    size_t preserved_bytes     = 0; // memory preserved by current Device object, but not allocated yet
    size_t max_consumed_bytes  = 0; // only applicable if RTP_LLM_TRACE_MEMORY is enabled.
};

// runtime device status, such as available memory.
struct DeviceStatus {
    MemoryStatus device_memory_status;
    MemoryStatus host_memory_status;
};

}; // namespace rtp_llm
