#pragma once

#include "maga_transformer/cpp/dataclass/MergedQuery.h"

namespace rtp_llm {

// Consecutive queries of same generate config are put into this group.
class QueryGroup {
public:
    std::list<th::intrusive_ptr<MagaQuery>> queries;
    th::intrusive_ptr<GenerateConfig> generate_config;
};

// QueryManager is responsible for maintaining query queue, and generating query groups.
class QueryManager {
public:
    QueryManager() {};
    ~QueryManager() {};

    void add_query(th::intrusive_ptr<MagaQuery> query);

    const std::shared_ptr<const QueryGroup> get_requests();
    void update_group(const std::shared_ptr<const QueryGroup> &query_group,
                      const ModelOutput &model_output,
                      const SamplerOutput &sampler_output);

private:
    std::queue<th::intrusive_ptr<MagaQuery>> query_queue_;
    std::mutex query_queue_mutex_;

    // These members do not need lock, they are expected to be accessed only in the main loop thread.
    std::shared_ptr<QueryGroup> active_query_group_;

};

}
