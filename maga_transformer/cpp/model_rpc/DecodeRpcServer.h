#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/model_rpc/RemoteRpcServer.h"

namespace rtp_llm {

struct DecodeRpcContext {
    DecodeRpcContext(grpc::ServerContext* context,
                     grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* grpc_stream)
                     : context(context), grpc_stream(grpc_stream) {}

    grpc::ServerContext* context;
    grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* grpc_stream;
};

class DecoderGenerateContext: public GenerateContext {
public:
    DecoderGenerateContext(DecodeRpcContext& rpc_context) 
                        : GenerateContext(0), rpc_context(rpc_context) {}

    struct TimeInfo {
        void updateRequestBegineTime() {
            receive_generate_time = autil::TimeUtility::currentTimeInMicroSeconds();
        }

        void updateLoadBeginTime() {
            start_load_time = autil::TimeUtility::currentTimeInMicroSeconds();
        }

        void updateLoadEndTime() {
            load_done_time = autil::TimeUtility::currentTimeInMicroSeconds();
        }

        void updateGenerateStartTime() {
            receive_generate_time = autil::TimeUtility::currentTimeInMicroSeconds();
        }

        void updateGenerateEndTime() {
            compute_done_time = autil::TimeUtility::currentTimeInMicroSeconds();
        }
        int64_t receive_load_time;
        int64_t start_load_time;
        int64_t receive_generate_time;
        int64_t load_done_time;
        int64_t begin_compute_time;
        int64_t compute_done_time;
    };

public:
    DecodeRpcContext& rpc_context;
    std::string peer_ip;
    grpc::Status error_status = grpc::Status::OK;
    std::string request_key;
    GenerateRequestPB allocate_request;
    TimeInfo time_info;
};

class LoadKVCacheContext {
public:
    LoadKVCacheContext(int64_t request_id, const std::string& request_key, const std::string& peer_ip,
                        const std::vector<int32_t>& cache_keys, const std::vector<int32_t>& block_ids,
                        int64_t reuse_block_size) :
                        request_id(request_id), request_key(request_key), peer_ip(peer_ip),
                        cache_keys(cache_keys), block_ids(block_ids), reuse_block_size(reuse_block_size) {}
    int64_t request_id;
    const std::string& request_key;
    const std::string& peer_ip;
    const std::vector<int32_t>& cache_keys;
    const std::vector<int32_t>& block_ids;
    int64_t reuse_block_size;
};

class DecodeRpcServer: public RemoteRpcServer {
public:
    DecodeRpcServer() {}
    ~DecodeRpcServer();
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);
    
    grpc::Status remote_generate(grpc::ServerContext*                                            context,
                                 grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream);

    grpc::Status
    remote_load(grpc::ServerContext* context, const RemoteLoadRequestPB* request, EmptyPB* response);

    bool ready() {
        return true;
    }

private:
    void initThreadPool();
    void prepareGenerateContext(DecoderGenerateContext& decode_context);
    void allocateResource(DecoderGenerateContext& decode_context);
    void loadCacheFromPrefill(DecoderGenerateContext& decode_context);
    void localGenerate(DecoderGenerateContext& decode_context);
    void reportTime(DecoderGenerateContext& decode_context);

    absl::Status loadCache(const LoadKVCacheContext& load_context);
    absl::Status loadCacheForAllRank(DecoderGenerateContext& decode_context);
    RemoteLoadRequestPB constructRemoteLoadRequest(const LoadKVCacheContext& load_context) const;

private:
    autil::ThreadPoolBasePtr thread_pool_;
};

}
