#include "rtp_llm/cpp/distribute/RpcCpuTpBroadcaster.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <chrono>
#include <cstring>
#include <sstream>

namespace rtp_llm {

namespace {

constexpr int kDefaultTimeoutMs = 30000;

int normalizeTimeoutMs(int timeout_ms) {
    return timeout_ms > 0 ? timeout_ms : kDefaultTimeoutMs;
}

}  // namespace

RpcCpuTpBroadcaster& RpcCpuTpBroadcaster::instance() {
    static RpcCpuTpBroadcaster i;
    return i;
}

std::size_t RpcCpuTpBroadcaster::InboxKeyHash::operator()(const InboxKey& key) const {
    std::size_t h = std::hash<std::string>{}(key.group_key);
    h ^= std::hash<uint64_t>{}(key.seq) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.dst_tp_rank) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

std::string RpcCpuTpBroadcaster::makeGroupKey(int dp_rank, int tp_size, int world_size) const {
    std::ostringstream oss;
    oss << "tp_cpu_broadcast:dp=" << dp_rank << ":tp=" << tp_size << ":world=" << world_size;
    return oss.str();
}

void RpcCpuTpBroadcaster::initialize(int                            tp_rank,
                                     int                            tp_size,
                                     int                            dp_rank,
                                     int                            world_size,
                                     const std::vector<std::string>& worker_grpc_addrs,
                                     int                            timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    timeout_ms = normalizeTimeoutMs(timeout_ms);

    if (initialized_.load(std::memory_order_acquire)) {
        const std::string new_group_key = makeGroupKey(dp_rank, tp_size, world_size);
        RTP_LLM_CHECK_WITH_INFO(tp_rank_ == tp_rank && tp_size_ == tp_size && dp_rank_ == dp_rank
                                    && world_size_ == world_size && group_key_ == new_group_key,
                                "RpcCpuTpBroadcaster re-init mismatch: was rank=%d size=%d dp=%d world=%d group=%s, "
                                "now rank=%d size=%d dp=%d world=%d group=%s",
                                tp_rank_,
                                tp_size_,
                                dp_rank_,
                                world_size_,
                                group_key_.c_str(),
                                tp_rank,
                                tp_size,
                                dp_rank,
                                world_size,
                                new_group_key.c_str());
        return;
    }

    if (tp_size <= 1) {
        tp_rank_    = tp_rank;
        tp_size_    = tp_size;
        dp_rank_    = dp_rank;
        world_size_ = world_size;
        timeout_ms_ = timeout_ms;
        group_key_  = makeGroupKey(dp_rank, tp_size, world_size);
        initialized_.store(true, std::memory_order_release);
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(tp_rank >= 0 && tp_rank < tp_size,
                            "RpcCpuTpBroadcaster bad tp_rank=%d tp_size=%d",
                            tp_rank,
                            tp_size);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int>(worker_grpc_addrs.size()) >= world_size,
                            "RpcCpuTpBroadcaster worker_grpc_addrs too small: addrs=%zu world_size=%d",
                            worker_grpc_addrs.size(),
                            world_size);

    tp_rank_    = tp_rank;
    tp_size_    = tp_size;
    dp_rank_    = dp_rank;
    world_size_ = world_size;
    timeout_ms_ = timeout_ms;
    group_key_  = makeGroupKey(dp_rank, tp_size, world_size);
    seq_.store(0, std::memory_order_release);
    inbox_.clear();
    peer_addrs_.clear();
    peer_tp_ranks_.clear();
    broadcast_manager_.reset();

    if (tp_rank_ == 0) {
        peer_addrs_.reserve(tp_size - 1);
        peer_tp_ranks_.reserve(tp_size - 1);
        for (int peer_tp_rank = 1; peer_tp_rank < tp_size; ++peer_tp_rank) {
            const int world_rank = dp_rank * tp_size + peer_tp_rank;
            RTP_LLM_CHECK_WITH_INFO(world_rank >= 0 && world_rank < static_cast<int>(worker_grpc_addrs.size()),
                                    "RpcCpuTpBroadcaster bad peer world_rank=%d addrs=%zu",
                                    world_rank,
                                    worker_grpc_addrs.size());
            peer_addrs_.push_back(worker_grpc_addrs[world_rank]);
            peer_tp_ranks_.push_back(peer_tp_rank);
        }
        broadcast_manager_ = std::make_shared<BroadcastManager>(peer_addrs_);
        RTP_LLM_CHECK_WITH_INFO(broadcast_manager_->init(),
                                "RpcCpuTpBroadcaster BroadcastManager init failed for %zu peer(s)",
                                peer_addrs_.size());
    }

    initialized_.store(true, std::memory_order_release);
    cv_.notify_all();
    RTP_LLM_LOG_INFO("Initialized RpcCpuTpBroadcaster rank=%d tp_size=%d dp_rank=%d world_size=%d peers=%zu timeout_ms=%d",
                     tp_rank_,
                     tp_size_,
                     dp_rank_,
                     world_size_,
                     peer_addrs_.size(),
                     timeout_ms_);
}

void RpcCpuTpBroadcaster::reset() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        inbox_.clear();
        peer_addrs_.clear();
        peer_tp_ranks_.clear();
        broadcast_manager_.reset();
        tp_rank_    = 0;
        tp_size_    = 1;
        dp_rank_    = 0;
        world_size_ = 1;
        timeout_ms_ = kDefaultTimeoutMs;
        group_key_.clear();
        seq_.store(0, std::memory_order_release);
        initialized_.store(false, std::memory_order_release);
    }
    cv_.notify_all();
}

uint64_t RpcCpuTpBroadcaster::nextSeq() {
    return seq_.fetch_add(1, std::memory_order_acq_rel);
}

void RpcCpuTpBroadcaster::broadcast(void* buf, std::size_t nbytes, int root) {
    RTP_LLM_CHECK_WITH_INFO(initialized_.load(std::memory_order_acquire),
                            "RpcCpuTpBroadcaster::broadcast called before initialize");
    if (tp_size_ <= 1 || nbytes == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(root == 0, "RpcCpuTpBroadcaster supports only root=0; got %d", root);

    const uint64_t seq = nextSeq();
    if (tp_rank_ == 0) {
        std::shared_ptr<BroadcastManager> manager;
        std::vector<int>                  peer_tp_ranks;
        std::string                       group_key;
        int                               timeout_ms = kDefaultTimeoutMs;
        {
            std::lock_guard<std::mutex> lock(mu_);
            manager       = broadcast_manager_;
            peer_tp_ranks = peer_tp_ranks_;
            group_key     = group_key_;
            timeout_ms    = timeout_ms_;
        }
        RTP_LLM_CHECK_WITH_INFO(manager != nullptr, "RpcCpuTpBroadcaster root has no BroadcastManager");

        std::vector<CpuTpBroadcastRequestPB> requests;
        requests.reserve(peer_tp_ranks.size());
        for (int peer_tp_rank : peer_tp_ranks) {
            CpuTpBroadcastRequestPB request;
            request.set_group_key(group_key);
            request.set_seq(seq);
            request.set_root(root);
            request.set_src_tp_rank(tp_rank_);
            request.set_dst_tp_rank(peer_tp_rank);
            request.set_nbytes(static_cast<uint64_t>(nbytes));
            request.set_payload(buf, nbytes);
            requests.push_back(std::move(request));
        }

        auto rpc_call = [](std::shared_ptr<RpcService::Stub>& stub,
                           std::shared_ptr<grpc::ClientContext>& ctx,
                           const CpuTpBroadcastRequestPB& request,
                           grpc::CompletionQueue* cq) {
            return stub->AsyncCpuTpBroadcast(ctx.get(), request, cq);
        };

        auto result = manager->broadcast<CpuTpBroadcastRequestPB, CpuTpBroadcastResponsePB>(
            requests, timeout_ms, rpc_call);
        RTP_LLM_CHECK_WITH_INFO(result != nullptr,
                                "RpcCpuTpBroadcaster broadcast setup failed seq=%lu nbytes=%zu",
                                seq,
                                nbytes);
        RTP_LLM_CHECK_WITH_INFO(result->waitDone(timeout_ms),
                                "RpcCpuTpBroadcaster broadcast wait timeout seq=%lu timeout_ms=%d",
                                seq,
                                timeout_ms);
        RTP_LLM_CHECK_WITH_INFO(result->success(), "RpcCpuTpBroadcaster broadcast RPC failed seq=%lu", seq);
        for (const auto& response : result->responses()) {
            RTP_LLM_CHECK_WITH_INFO(response.success(),
                                    "RpcCpuTpBroadcaster peer rejected seq=%lu: %s",
                                    seq,
                                    response.error_message().c_str());
        }
        return;
    }

    InboxKey    key;
    std::string payload;
    int         timeout_ms = kDefaultTimeoutMs;
    {
        std::unique_lock<std::mutex> lock(mu_);
        key        = InboxKey{group_key_, seq, tp_rank_};
        timeout_ms = timeout_ms_;
        const bool ready = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return !initialized_.load(std::memory_order_acquire) || inbox_.find(key) != inbox_.end();
        });
        RTP_LLM_CHECK_WITH_INFO(ready && initialized_.load(std::memory_order_acquire),
                                "RpcCpuTpBroadcaster receive timeout seq=%lu rank=%d timeout_ms=%d",
                                seq,
                                tp_rank_,
                                timeout_ms);
        auto it = inbox_.find(key);
        RTP_LLM_CHECK_WITH_INFO(it != inbox_.end(), "RpcCpuTpBroadcaster missing inbox payload seq=%lu", seq);
        payload = std::move(it->second);
        inbox_.erase(it);
    }

    RTP_LLM_CHECK_WITH_INFO(payload.size() == nbytes,
                            "RpcCpuTpBroadcaster size mismatch seq=%lu rank=%d expected=%zu actual=%zu",
                            seq,
                            tp_rank_,
                            nbytes,
                            payload.size());
    std::memcpy(buf, payload.data(), nbytes);
}

bool RpcCpuTpBroadcaster::handleBroadcastRequest(const CpuTpBroadcastRequestPB& request,
                                                 CpuTpBroadcastResponsePB*      response) {
    auto fail = [&](const std::string& message) {
        response->set_success(false);
        response->set_error_message(message);
        RTP_LLM_LOG_WARNING("RpcCpuTpBroadcaster rejected request: %s", message.c_str());
        return false;
    };

    std::unique_lock<std::mutex> lock(mu_);
    if (!initialized_.load(std::memory_order_acquire)) {
        cv_.wait_for(lock, std::chrono::milliseconds(kDefaultTimeoutMs), [&] {
            return initialized_.load(std::memory_order_acquire);
        });
    }
    if (!initialized_.load(std::memory_order_acquire)) {
        return fail("broadcaster is not initialized");
    }
    if (request.group_key() != group_key_) {
        return fail("group_key mismatch: got " + request.group_key() + ", expected " + group_key_);
    }
    if (request.root() != 0 || request.src_tp_rank() != 0) {
        return fail("only root tp_rank 0 is supported");
    }
    if (request.dst_tp_rank() != tp_rank_) {
        return fail("dst_tp_rank mismatch");
    }
    if (request.nbytes() != request.payload().size()) {
        return fail("payload size mismatch");
    }

    InboxKey key{request.group_key(), request.seq(), request.dst_tp_rank()};
    if (inbox_.find(key) != inbox_.end()) {
        return fail("duplicate payload");
    }
    inbox_.emplace(std::move(key), request.payload());
    response->set_success(true);
    response->clear_error_message();
    cv_.notify_all();
    return true;
}

}  // namespace rtp_llm
