#include <mutex>
#include <memory>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>

#include "autil/TimeUtility.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/NetUtil.h"
#include "maga_transformer/cpp/utils/KVCacheUtils.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/model_rpc/DecodeRpcServer.h"

using namespace std;
using namespace autil::legacy;
using namespace fastertransformer;

using grpc::ClientContext;
using grpc::Status;

const int LOAD_TIMEOUT_MS = 5000;

#define GRPC_RET_IF_ERROR(decode_context, stat, code, msg)          \
    if (!(stat)) {                                                  \
        decode_context.error_status = grpc::Status(code, msg);      \
        return;                                                     \
    }

#define EXECUTE_STAGE_FUNC(func, decode_context)                    \
    func(decode_context);                                           \
    if (!decode_context.error_status.ok()) {                        \
        return decode_context.error_status;                         \
    }

string makeRequestKey(const string& client_id, size_t request_id) {
    return client_id + "_request_id_" + std::to_string(request_id);
}

namespace rtp_llm {

class LoadStatus {
public:
    enum class Status {
        LOADING,
        SUCCESS,
        FAILED,
    };
    LoadStatus(): status_(Status::LOADING) {}

    Status waitDone() {
        std::unique_lock<std::mutex> lock(lock_);
        while (status_ == Status::LOADING) {
            cond_.wait_for(lock, std::chrono::milliseconds(5));
        }
        return status_;
    }

    void updateResult(bool success, CacheStoreErrorCode ec) {
        std::lock_guard<std::mutex> lock(lock_);
        if (success) {
            status_ = Status::SUCCESS;
        } else {
            status_ = Status::FAILED;
        }
        error_code_ = ec;
        cond_.notify_all();
    }

    inline std::string to_string() {
        switch (status_) {
            case Status::LOADING:
                return "LOADING";
            case Status::SUCCESS:
                return "SUCCESS";
            case Status::FAILED:
                return "FAILED, error code is " + CacheStoreErrorCodeToString(error_code_);
            default:
                return "UNKNOWN STATUS";
        }
    }

protected:
    Status                  status_;
    CacheStoreErrorCode     error_code_;
    std::condition_variable cond_;
    std::mutex              lock_;
};

DecodeRpcServer::~DecodeRpcServer() {
    if (thread_pool_) {
        thread_pool_->stop();
        thread_pool_.reset();
    }
}

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   py::object                                             mm_process_engine,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initThreadPool();
    return grpc::Status::OK;
}

void DecodeRpcServer::initThreadPool() {
    if (maga_init_params_.gpt_init_parameter.tp_size_ == 1) {
        return;
    }
    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        maga_init_params_.gpt_init_parameter.tp_size_ * 8, maga_init_params_.gpt_init_parameter.tp_size_ * 8, nullptr, "RemoteCacheLoadPool");
    FT_CHECK_WITH_INFO(thread_pool_->start(), "DecodeRpcServer init ThreadPool failed");
    FT_LOG_INFO("normal cache store init done");
}

void DecodeRpcServer::prepareGenerateContext(DecoderGenerateContext& decode_context) {
    auto& allocate_request = decode_context.allocate_request;
    GRPC_RET_IF_ERROR(decode_context, decode_context.rpc_context.grpc_stream->Read(&allocate_request),
                        grpc::StatusCode::INTERNAL, "failed to get message");
    GRPC_RET_IF_ERROR(decode_context, allocate_request.stage() == RemoteStage::ALLOCATE,
                      grpc::StatusCode::INTERNAL,
                      "message first status != RemoteStage::ALLOCATE");
    decode_context.request_id   = allocate_request.request_id();
    decode_context.request_key  = makeRequestKey(allocate_request.client_id(), decode_context.request_id);

    decode_context.peer_ip = extractIP(decode_context.rpc_context.context->peer());
    if (decode_context.peer_ip.empty()) {
        string error_msg = "request: [" + decode_context.request_key + "] get client ip failed, peer is "
                            + decode_context.rpc_context.context->peer();
        FT_LOG_ERROR(error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return;
    }
}

void DecodeRpcServer::allocateResource(DecoderGenerateContext& decode_context) {
    auto input            = QueryConverter::transQuery(&decode_context.allocate_request.input());
    auto generate_stream  = engine_->makeStream(input);
    decode_context.stream = generate_stream;
    decode_context.time_info.updateRequestBegineTime();
    auto stat = generate_stream->initKVBlock(0);
    if (!stat.ok()) {
        string error_msg = "request: [" + decode_context.request_key + "] malloc kv cache block failed";
        FT_LOG_ERROR(error_msg);
        decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
        return;
    }
    GRPC_RET_IF_ERROR(decode_context, decode_context.rpc_context.grpc_stream->Write(GenerateOutputsPB()),
                        grpc::StatusCode::INTERNAL, "failed to write allocate output");
}

void DecodeRpcServer::loadCacheFromPrefill(DecoderGenerateContext& decode_context) {
    auto& grpc_stream = decode_context.rpc_context.grpc_stream;
    GenerateRequestPB loadRequest;
    GRPC_RET_IF_ERROR(decode_context, grpc_stream->Read(&loadRequest),
                        grpc::StatusCode::INTERNAL, "failed to get loadReqeust");
    GRPC_RET_IF_ERROR(decode_context, loadRequest.stage() == RemoteStage::LOAD,
                        grpc::StatusCode::INTERNAL, "request stage is not RemoteStage::LOAD");
    decode_context.time_info.updateLoadBeginTime();
    auto loadStatus = loadCacheForAllRank(decode_context);
    decode_context.time_info.updateLoadEndTime();

    GRPC_RET_IF_ERROR(decode_context, loadStatus.ok(),
                        grpc::StatusCode::INTERNAL, loadStatus.ToString().c_str());
    GRPC_RET_IF_ERROR(decode_context,
                        grpc_stream->Write(GenerateOutputsPB()), grpc::StatusCode::INTERNAL, "send load response failed");
}

void DecodeRpcServer::localGenerate(DecoderGenerateContext& decode_context) {
    auto& grpc_stream = decode_context.rpc_context.grpc_stream;
    auto& generate_stream = decode_context.stream;
    GenerateRequestPB generate_request;
    GRPC_RET_IF_ERROR(decode_context, grpc_stream->Read(&generate_request),
                        grpc::StatusCode::INTERNAL, "poll generate request failed");
    GRPC_RET_IF_ERROR(decode_context, generate_request.stage() == RemoteStage::GENERATE,
                      grpc::StatusCode::INTERNAL,
                      "message first status != RemoteStage::GENERATE");
    decode_context.time_info.updateGenerateStartTime();
    generate_stream->setIsContextStream(false);
    generate_stream->step();
    auto new_tokens = engine_->getDevice()->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)generate_stream->tileNum(), (size_t)1}, ft::AllocationType::HOST}, {});
    auto data           = new_tokens->data<int32_t>();
    auto first_token_id = generate_request.first_generate_token_id();
    *data               = first_token_id;
    generate_stream->update(new_tokens, 1, nullptr, nullptr, nullptr, nullptr, nullptr, false);
    generate_stream->incLastOutputPos();
    engine_->enqueue(generate_stream);
    FT_LOG_DEBUG("request:[%s] enqueue success", decode_context.request_key.c_str());
    decode_context.error_status = pollStreamOutput(decode_context.rpc_context.context,
                                   generate_request.request_id(),
                                   dynamic_cast<grpc::internal::WriterInterface<GenerateOutputsPB>*>(grpc_stream),
                                   generate_stream);
    decode_context.time_info.updateGenerateEndTime();
}

void DecodeRpcServer::reportTime(DecoderGenerateContext& decode_context) {
    GenerateOutputsPB response;
    const auto& time_info = decode_context.time_info;
    response.set_receive_load_time(time_info.receive_load_time);
    response.set_start_load_time(time_info.start_load_time);
    response.set_receive_generate_time(time_info.receive_generate_time);
    response.set_load_done_time(time_info.load_done_time);
    response.set_begin_compute_time(time_info.begin_compute_time);
    response.set_compute_done_time(time_info.compute_done_time);
    decode_context.rpc_context.grpc_stream->Write(response);

    RPCMetricsCollector collector;
    collector.load_latency_us = time_info.load_done_time - time_info.start_load_time;
    collector.onflight_request = onflightRequestNum();
    reportMetrics(&collector);
}

RemoteLoadRequestPB DecodeRpcServer::constructRemoteLoadRequest(const LoadKVCacheContext& load_context) const {
    RemoteLoadRequestPB request;
    request.set_request_id(load_context.request_id);
    request.set_request_key(load_context.request_key);
    request.set_peer_ip(load_context.peer_ip);
    for (auto& cache_key : load_context.cache_keys) {
        request.add_cache_keys(cache_key);
    }
    for (auto& block_id : load_context.block_ids) {
        request.add_block_ids(block_id);
    }
    return request;
}

absl::Status DecodeRpcServer::loadCacheForAllRank(DecoderGenerateContext& decode_context) {
    auto& generate_stream = decode_context.stream;
    auto& cache_keys = generate_stream->cacheKeys();
    auto& block_ids  = generate_stream->kvCache().blocks(0);
    if (cache_keys.size() != block_ids.size()) {
        return absl::Status(absl::StatusCode::kInternal, "block size not equal to cache keys size");
    }

    LoadKVCacheContext load_context{decode_context.request_id, decode_context.request_key,
                                    decode_context.peer_ip, cache_keys,
                                    block_ids, generate_stream->reuseBlockSize()};

    if (maga_init_params_.gpt_init_parameter.tp_size_ == 1) {
        return loadCache(load_context);
    }
    std::vector<autil::ThreadPoolBase::Future<absl::Status>> futures;
    auto local_task = [&] {
        auto res = this->loadCache(load_context);
        return res;
    };
    futures.emplace_back(thread_pool_->async(local_task));
    RemoteLoadRequestPB load_request;
    if (maga_init_params_.gpt_init_parameter.tp_size_ > 1) {
        load_request = constructRemoteLoadRequest(load_context);
        for (int i = 0; i < workers_.size(); i++) {
            auto& worker = this->workers_[i];
            auto remote_task = [&]() {
                auto connect_status = this->rpc_pool_.getConnection(worker);
                if (!connect_status.ok()) {
                    string error_msg = "get grpc connection for ip " + worker + " failed";
                    return absl::Status(absl::StatusCode::kInternal, error_msg);
                }
                auto          stub = connect_status.value().stub.get();
                ClientContext client_context;
                EmptyPB       response;
                auto          grpc_status = stub->remote_load(&client_context, load_request, &response);
                if (!grpc_status.ok()) {
                    return absl::Status(absl::StatusCode::kInternal, grpc_status.error_message());
                }
                return absl::OkStatus();
            };
            futures.emplace_back(thread_pool_->async(remote_task));
        }
    }
    std::string err_msg = "failed to load kv cache in rank: ";
    bool        success = true;
    for (int i = 0; i < futures.size(); i++ ) {
        auto status = futures[i].get();
        if (!status.ok()) {
            // TODO(xinfei.sxf) 可以不等待其他rank的结果吗
            success = false;
            err_msg += std::to_string(i) + ": " + status.ToString() + ", ";
        }
    }
    if (!success) {
        FT_LOG_WARNING(err_msg);
        return absl::Status(absl::StatusCode::kInternal, err_msg);
    }
    return absl::OkStatus();
}

absl::Status DecodeRpcServer::loadCache(const LoadKVCacheContext& load_context) {
    const auto& request_key   = load_context.request_key;
    auto        cache_manager = engine_->resourceContext().cache_manager;
    const auto& cache_config  = cache_manager->cacheConfig();
    auto        block_size    = cache_config.kv_block_stride;
    auto        scale_block_size    = cache_config.kv_scale_block_stride;
    auto        block_num     = load_context.block_ids.size();

    auto start_load_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    auto load_cache    = std::make_shared<RequestBlockBuffer>(std::to_string(load_context.request_id));
    for (size_t layer_id = 0; layer_id < maga_init_params_.gpt_init_parameter.num_layers_; layer_id++) {
        for (size_t block_pos = load_context.reuse_block_size; block_pos < block_num; block_pos++) {
            auto                  cache_key = makeCacheKey(std::to_string(load_context.cache_keys[block_pos]), layer_id);
            auto                  block_id  = load_context.block_ids[block_pos];
            auto                  addr_info = cache_manager->convertIndexToAddr(block_id, layer_id);
            std::shared_ptr<void> k_block_addr(addr_info.k_addr, [](void* p) {});
            std::shared_ptr<void> v_block_addr(addr_info.v_addr, [](void* p) {});
            load_cache->addBlock("k_" + cache_key, k_block_addr, block_size, true, true);
            load_cache->addBlock("v_" + cache_key, v_block_addr, block_size, true, true);

            if (addr_info.k_scale_addr) {
                std::shared_ptr<void> k_scale_addr(addr_info.k_scale_addr, [](void* p) {});
                std::shared_ptr<void> v_scale_addr(addr_info.v_scale_addr, [](void* p) {});
                load_cache->addBlock("k_scale" + cache_key, k_scale_addr, scale_block_size, true, true);
                load_cache->addBlock("v_scale" + cache_key, v_scale_addr, scale_block_size, true, true);
            }
        }
    }
    auto load_status = LoadStatus();
    auto load_callback = [request_key, start_load_time_us, &load_status, this](bool success, CacheStoreErrorCode ec) {
        auto load_done_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        load_status.updateResult(success, ec);
        if (ec != CacheStoreErrorCode::None) {
            FT_LOG_WARNING("request %s all layers load finished, state:[%s], error code[%s], cost time %ldus",
                request_key.c_str(), success ? "success" : "failed",
                CacheStoreErrorCodeToString(ec).c_str(), load_done_time_us - start_load_time_us);
        } else {
            FT_LOG_DEBUG("request %s all layers load finished, state:[%s], cost time %ldus",
                request_key.c_str(), success ? "success" : "failed", load_done_time_us - start_load_time_us);
        }
    };
    auto timeout_ms = maga_init_params_.gpt_init_parameter.load_cache_timeout_ms_ ? 
                        maga_init_params_.gpt_init_parameter.load_cache_timeout_ms_ : LOAD_TIMEOUT_MS;
    cache_store_->load(load_cache, load_callback, load_context.peer_ip, timeout_ms);

    // TODO(xinfei.sxf) use coroutine or check multi load ready at one loop
    int64_t wait_load_time_us = 0;
    if (load_status.waitDone() == LoadStatus::Status::SUCCESS) {
        FT_LOG_DEBUG("request [%s] load kv cache success, wait %ldus", request_key.c_str(), wait_load_time_us);
    } else {
        // TODO(xinfei.sxf) add retry for part failed blocks.
        string error_msg =
            "request " + request_key + " , load kv cache failed, load status = " + load_status.to_string();
        FT_LOG_ERROR(error_msg);
        return absl::InternalError(error_msg);
    }
    return absl::OkStatus();
}

grpc::Status DecodeRpcServer::remote_load(grpc::ServerContext* context, const RemoteLoadRequestPB* request, EmptyPB* response) {
    std::vector<int32_t> cache_keys(request->cache_keys().begin(), request->cache_keys().end());
    std::vector<int32_t> block_ids(request->block_ids().begin(), request->block_ids().end());
    auto status = loadCache({request->request_id(), request->request_key(), request->peer_ip(),
                                cache_keys, block_ids, request->reuse_block_size()});
    if (status.ok()) {
        return grpc::Status::OK;
    } else {
        return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
    }
}

grpc::Status DecodeRpcServer::remote_generate(
    grpc::ServerContext* context, grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* grpc_stream) {
    AtomicGuard request_guard(onflight_requests_);
    DecodeRpcContext rpc_context(context, grpc_stream);
    auto decode_context = DecoderGenerateContext(rpc_context);
    EXECUTE_STAGE_FUNC(prepareGenerateContext, decode_context);
    EXECUTE_STAGE_FUNC(allocateResource, decode_context);
    EXECUTE_STAGE_FUNC(loadCacheFromPrefill, decode_context);
    EXECUTE_STAGE_FUNC(localGenerate, decode_context);
    EXECUTE_STAGE_FUNC(reportTime, decode_context);
    return grpc::Status::OK;
}

}  // namespace rtp_llm
