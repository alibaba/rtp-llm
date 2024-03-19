#include "maga_transformer/cpp/components/QueryManager.h"

using namespace std;

namespace rtp_llm {

void QueryManager::add_query(th::intrusive_ptr<MagaQuery> query) {
    lock_guard<mutex> lock(query_queue_mutex_);
    query_queue_.push(query);
}

const shared_ptr<const QueryGroup> QueryManager::get_requests() {
    return active_query_group_;
}

void QueryManager::update_group(const shared_ptr<const QueryGroup> &query_group,
                                const ModelOutput &model_output,
                                const SamplerOutput &sampler_output)
{
}

} // namespace rtp_llm

