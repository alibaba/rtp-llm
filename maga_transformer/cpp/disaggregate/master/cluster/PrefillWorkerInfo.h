#include <vector>
#include <cstdint>
#include "maga_transformer/cpp/disaggregate/master/scheduler/Struct.h"

namespace rtp_llm {
namespace rtp_llm_master {

class KVCacheRadixTree {};

struct PrefillTask {
    int task_id;
    int prefix_length;
    int input_length;
};

class PrefillWorkerInfo {
public:
    PrefillWorkerInfo();
    // update from response of worker
    void updateWorkerInfo(std::vector<PrefillTask>& task_list, int64_t last_running_time);
    // update when append task
    void appendPendingTask(PrefillTask task);
    // prefix tree
    void updatePrefixTree(KVCacheRadixTree& tree);
    // get local cache size
    int matchBlockNum(std::vector<int64_t> token_block_hash);

    int64_t predictFinishTime() const;
    int64_t lastRunningTime() const;
protected:
    std::vector<PrefillTask> running_task_list;
    std::vector<PrefillTask> pending_task_list;
    int64_t last_running_time;
    int64_t predict_finish_time;
    KVCacheRadixTree tree;

};

}  // namespace rtp_llm_master
}  // namespace rtp_llm
