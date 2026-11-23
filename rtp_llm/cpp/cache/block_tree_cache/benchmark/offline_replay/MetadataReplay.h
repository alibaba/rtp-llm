#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace rtp_llm::benchmark {
struct MetadataReplayConfig {
    bool disk_fifo{false};
    MetadataReplayConfig();
    size_t block_tokens{256}, cp_size{1}, cp_rank{0}, min_replay_tokens{1};
    size_t device_blocks{64}, host_blocks{128}, disk_blocks{256};
    size_t host_swa_blocks{0}, disk_swa_blocks{0};
    bool   swa{true}, dsv41_topology{false};
    size_t swa_block_tokens{0}, swa_blocks{64}, sliding_window{128}, last_n, linear_step;
    double device_low{0}, device_high{0}, host_low{0}, host_high{0}, disk_low{0}, disk_high{0};
};
struct MetadataRequest {
    int64_t              id{0};
    uint64_t             timestamp_ns{0};
    int                  input_tokens{0};
    std::vector<int64_t> prefix_keys;
    bool                 observe{true};
};
struct MetadataRequestResult {
    bool                            success{false};
    int                             reuse_tokens{0};
    uint64_t                        host_hit_tokens{0};
    uint64_t                        disk_hit_tokens{0};
    uint64_t                        full_match_tokens{0}, joint_match_tokens{0};
    std::map<std::string, uint64_t> evictions;
    std::string                     status;
};
struct MetadataReplayStats {
    uint64_t watermark_calls{0}, watermark_blocks{0}, force_calls{0}, force_blocks{0};
    uint64_t transfers{0}, failed_transfers{0}, device_host{0}, host_disk{0}, host_device{0}, disk_device{0};
    uint64_t payload_bytes{0}, copy_attempts{0};
    uint64_t tree_nodes{0}, device_used{0}, host_used{0}, disk_used{0};
};
class MetadataReplay {
public:
    explicit MetadataReplay(const MetadataReplayConfig& config);
    ~MetadataReplay();
    MetadataRequestResult run(const MetadataRequest& request);
    MetadataReplayStats   stats() const;
    // Test the native allocation callback with every pool slot request-pinned.
    bool checkPinnedForceDrop();
    void failNextTransfer();
    void writeTreeStructure(const std::string& directory) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace rtp_llm::benchmark
