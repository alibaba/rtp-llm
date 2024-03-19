#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/common/torch_bind.h"

using namespace std;
using namespace torch;

namespace rtp_llm {

MagaQuery::MagaQuery(const intrusive_ptr<QueryRequest> &query)
    : query(query)
    , done_(false)
    , cancelled_(false)
    , current_response_(nullptr)
{
    assert(query->input_ids.dim() == 2);
    batch_size_ = query->input_ids.size(0);
}

MagaQuery::~MagaQuery() {}

intrusive_ptr<GenerateResponse> MagaQuery::next_response() {
    unique_lock<mutex> lock(response_mutex_);
    if (!done_) {
        update_cv_.wait(lock, [this] { return done_ || cancelled_; });
    }
    const auto response = current_response_;
    return response;
}

void MagaQuery::cancel() {
    lock_guard<mutex> lock(response_mutex_);
    cancelled_ = true;
    done_ = true;
    update_cv_.notify_all();
}

void MagaQuery::push_response(const intrusive_ptr<GenerateResponse> &response) {
    lock_guard<mutex> lock(response_mutex_);
    if (cancelled_) {
        return;
    }
    current_response_ = response;
    done_ = response->all_finished;
    update_cv_.notify_all();
}

bool MagaQuery::is_done() {
    lock_guard<mutex> lock(response_mutex_);
    return done_;
}

void MagaQuery::add_cache_block(const vector<int64_t> &kv_cache_block) {}

} // namespace rtp_llm

