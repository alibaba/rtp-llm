#include "rtp_llm/cpp/cache/connector/p2p/DecodeLoadHelper.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/unknown_field_set.h>
#include <grpc++/grpc++.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <utility>

namespace rtp_llm {

namespace {

using google::protobuf::UnknownField;
using google::protobuf::UnknownFieldSet;

struct WorkerAddrParts {
    std::string host;
    int32_t     cache_store_port = 0;
};

bool parseWorkerAddrPort(const std::string& port_text, int32_t* port) {
    if (!port || port_text.empty()) {
        return false;
    }
    char* end   = nullptr;
    errno       = 0;
    auto value  = std::strtoll(port_text.c_str(), &end, 10);
    const bool parsed_all = end == port_text.c_str() + port_text.size();
    if (errno != 0 || !parsed_all || value < 1 || value > 65535) {
        return false;
    }
    *port = static_cast<int32_t>(value);
    return true;
}

bool parseWorkerAddr(const std::string& worker_addr, WorkerAddrParts* parts) {
    if (!parts || worker_addr.empty()) {
        return false;
    }

    std::string host;
    std::string cache_store_port_text;
    std::string grpc_port_text;
    if (worker_addr.front() == '[') {
        const auto close_pos = worker_addr.find(']');
        if (close_pos == std::string::npos || close_pos + 1 >= worker_addr.size()
            || worker_addr[close_pos + 1] != ':') {
            return false;
        }
        const auto cache_port_begin = close_pos + 2;
        const auto cache_port_end   = worker_addr.find(':', cache_port_begin);
        if (cache_port_end == std::string::npos || cache_port_end + 1 >= worker_addr.size()) {
            return false;
        }
        host                  = worker_addr.substr(1, close_pos - 1);
        cache_store_port_text = worker_addr.substr(cache_port_begin, cache_port_end - cache_port_begin);
        grpc_port_text        = worker_addr.substr(cache_port_end + 1);
    } else {
        const auto grpc_col = worker_addr.rfind(':');
        const auto cache_col = (grpc_col == std::string::npos || grpc_col == 0)
                                   ? std::string::npos
                                   : worker_addr.rfind(':', grpc_col - 1);
        if (cache_col == std::string::npos || cache_col == 0 || cache_col + 1 >= grpc_col
            || grpc_col + 1 >= worker_addr.size()) {
            return false;
        }
        host                  = worker_addr.substr(0, cache_col);
        cache_store_port_text = worker_addr.substr(cache_col + 1, grpc_col - cache_col - 1);
        grpc_port_text        = worker_addr.substr(grpc_col + 1);
        if (host.find(':') != std::string::npos && host.find('.') != std::string::npos) {
            return false;
        }
    }

    int32_t cache_store_port = 0;
    int32_t grpc_port        = 0;
    if (host.empty() || !parseWorkerAddrPort(cache_store_port_text, &cache_store_port)
        || !parseWorkerAddrPort(grpc_port_text, &grpc_port)) {
        return false;
    }
    parts->host             = std::move(host);
    parts->cache_store_port = cache_store_port;
    return true;
}

bool parsePackedInt32s(const std::string& bytes, std::vector<int32_t>& out) {
    google::protobuf::io::CodedInputStream input(reinterpret_cast<const uint8_t*>(bytes.data()),
                                                 static_cast<int>(bytes.size()));
    uint32_t                               value = 0;
    while (input.ReadVarint32(&value)) {
        out.push_back(static_cast<int32_t>(value));
    }
    return input.ConsumedEntireMessage();
}

bool extractLegacyStartLoadPayload(const P2PConnectorStartLoadResponsePB& response,
                                   P2PSideChannelPayload&                 side_channel_payload) {
    const UnknownFieldSet& unknown_fields     = response.GetReflection()->GetUnknownFields(response);
    bool                   found_legacy_field = false;
    bool                   has_first_token    = false;

    for (int i = 0; i < unknown_fields.field_count(); ++i) {
        const UnknownField& field = unknown_fields.field(i);
        switch (field.number()) {
            case 1:
                if (field.type() == UnknownField::TYPE_VARINT) {
                    side_channel_payload.first_token_id = static_cast<int64_t>(field.varint());
                    has_first_token                     = true;
                    found_legacy_field                  = true;
                }
                break;
            case 2:
                if (field.type() == UnknownField::TYPE_VARINT) {
                    side_channel_payload.total_reuse_len = static_cast<int32_t>(field.varint());
                    found_legacy_field                   = true;
                }
                break;
            case 3:
                if (field.type() == UnknownField::TYPE_VARINT) {
                    side_channel_payload.local_reuse_len = static_cast<int32_t>(field.varint());
                    found_legacy_field                   = true;
                }
                break;
            case 4:
                if (field.type() == UnknownField::TYPE_VARINT) {
                    side_channel_payload.remote_reuse_len = static_cast<int32_t>(field.varint());
                    found_legacy_field                    = true;
                }
                break;
            case 5:
                if (field.type() == UnknownField::TYPE_LENGTH_DELIMITED) {
                    found_legacy_field = true;
                    parsePackedInt32s(field.length_delimited(), side_channel_payload.propose_tokens);
                } else if (field.type() == UnknownField::TYPE_VARINT) {
                    found_legacy_field = true;
                    side_channel_payload.propose_tokens.push_back(static_cast<int32_t>(field.varint()));
                }
                break;
            case 6:
                if (field.type() == UnknownField::TYPE_LENGTH_DELIMITED) {
                    found_legacy_field = true;
                    side_channel_payload.propose_probs.ParseFromString(field.length_delimited());
                }
                break;
            case 7:
                if (field.type() == UnknownField::TYPE_LENGTH_DELIMITED) {
                    found_legacy_field = true;
                    side_channel_payload.propose_hidden.ParseFromString(field.length_delimited());
                }
                break;
            case 8:
                if (field.type() == UnknownField::TYPE_LENGTH_DELIMITED) {
                    found_legacy_field = true;
                    parsePackedInt32s(field.length_delimited(), side_channel_payload.position_ids);
                } else if (field.type() == UnknownField::TYPE_VARINT) {
                    found_legacy_field = true;
                    side_channel_payload.position_ids.push_back(static_cast<int32_t>(field.varint()));
                }
                break;
            case 11:
                if (field.type() == UnknownField::TYPE_VARINT) {
                    side_channel_payload.memory_reuse_len = static_cast<int32_t>(field.varint());
                    found_legacy_field                    = true;
                }
                break;
            default:
                break;
        }
    }

    side_channel_payload.has_first_token = has_first_token;
    side_channel_payload.has_data        = found_legacy_field;
    return found_legacy_field;
}

}  // namespace

DecodeLoadHelper::DecodeLoadHelper(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {
    rpc_pool_ = std::make_shared<RPCPool>();

    // worker_addrs entries are host:cache_store_port:grpc_port or [IPv6]:cache_store_port:grpc_port.
    for (const auto& worker_addr : worker_addrs_) {
        WorkerAddrParts parts;
        if (!parseWorkerAddr(worker_addr, &parts)) {
            RTP_LLM_FAIL("DecodeLoadHelper: invalid worker addr format [%s], expected "
                         "host:cache_store_port:grpc_port or [IPv6]:cache_store_port:grpc_port",
                         worker_addr.c_str());
            continue;
        }
        TPWorkerInfoPB tp_worker;
        tp_worker.set_ip(parts.host);
        tp_worker.set_cache_store_port(parts.cache_store_port);
        tp_worker_infos_.push_back(tp_worker);
    }
}

std::shared_ptr<DecodeLoadHelper::Result> DecodeLoadHelper::load(int64_t            request_id,
                                                                   const std::string& prefill_ip,
                                                                   uint32_t           prefill_port,
                                                                   const std::string& unique_key,
                                                                   int64_t            request_deadline_ms,
                                                                   int64_t            transfer_deadline_ms,
                                 int64_t            load_timeout_ms,
                                                                   bool               no_transfer,
                                                                   uint64_t           plan_digest,
                                                                   const std::vector<int>& active_route_ids) {
    auto result        = std::make_shared<Result>();
    result->request_id = request_id;
    result->unique_key = unique_key;
    if (!rpc_pool_) {
        result->status = grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::GET_CONNECTION_FAILED, "StartLoad RPC pool is null key=" + unique_key));
        result->complete(false);
        return result;
    }
    result->server_addr = formatGrpcHostPort(prefill_ip, prefill_port);
    if (result->server_addr.empty()) {
        result->status = grpcStatusFromErrorInfo(ErrorInfo(ErrorCode::INVALID_PARAMS,
                                                           "StartLoad invalid peer=" + prefill_ip + ":"
                                                               + std::to_string(prefill_port) + " key=" + unique_key));
        result->complete(false);
        return result;
    }

    // [PD-DIAG] Measure getConnection separately. RpcPool::getConnection holds a
    // pool-wide mutex and may synchronously trigger gRPC channel reconnection
    // (channel->GetState(true)) when the cached connection is in TRANSIENT_FAILURE,
    // which can serialize all callers behind the slow reconnect of one peer.
    const int64_t get_conn_start_us = currentTimeUs();
    auto          conn_status       = rpc_pool_->getConnection(result->server_addr);
    const int64_t get_conn_cost_us  = currentTimeUs() - get_conn_start_us;
    if (get_conn_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] DecodeLoadHelper::load slow getConnection, addr: %s, cost_us=%ld, unique_key: %s",
            result->server_addr.c_str(),
            get_conn_cost_us,
            unique_key.c_str());
    }
    if (!conn_status.ok()) {
        result->status =
            grpcStatusFromErrorInfo(ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                              "StartLoad getConnection peer=" + result->server_addr
                                                  + " key=" + unique_key + ": " + conn_status.status().ToString()));
        result->complete(false);
        return result;
    }

    result->stub = conn_status.value().stub;
    if (!result->stub) {
        result->status = grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                      "StartLoad stub is null peer=" + result->server_addr + " key=" + unique_key));
        result->complete(false);
        return result;
    }

    result->request_id    = request_id;
    result->unique_key    = unique_key;
    result->start_time_us = currentTimeUs();

    const int64_t build_rpc_start_us = currentTimeUs();
    if (!buildAndStartAsyncRpc(result,
                               unique_key,
                               request_deadline_ms,
                               transfer_deadline_ms,
                               load_timeout_ms,
                               request_id,
                               no_transfer,
                               plan_digest,
                               active_route_ids)) {
        if (result->status.ok())
            result->status = grpcStatusFromErrorInfo(
                ErrorInfo(ErrorCode::RPC_FINISH_FAILED,
                          "StartLoad async reader creation failed peer=" + result->server_addr + " key=" + unique_key));
        result->complete(false);
        return result;
    }
    const int64_t build_rpc_cost_us = currentTimeUs() - build_rpc_start_us;
    if (build_rpc_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] DecodeLoadHelper::load slow buildAndStartAsyncRpc, addr: %s, cost_us=%ld, unique_key: %s",
            result->server_addr.c_str(),
            build_rpc_cost_us,
            unique_key.c_str());
    }

    RTP_LLM_LOG_DEBUG("DecodeLoadHelper load started, unique_key: %s, addr: %s, request_deadline_ms: %lld, "
                      "transfer_deadline_ms: %lld, timeout ms: %d",
                      unique_key.c_str(),
                      result->server_addr.c_str(),
                      request_deadline_ms,
                      transfer_deadline_ms,
                      result->timeout_ms);
    return result;
}

bool DecodeLoadHelper::buildAndStartAsyncRpc(const std::shared_ptr<Result>& result,
                                              const std::string&             unique_key,
                                              int64_t                        request_deadline_ms,
                                              int64_t                        transfer_deadline_ms,
                               int64_t                        load_timeout_ms,
                                              int64_t                        request_id,
                                              bool                           no_transfer,
                                              uint64_t                       plan_digest,
                                              const std::vector<int>&        active_route_ids) {
    result->request.set_unique_key(unique_key);
    result->request.set_timeout_ms(load_timeout_ms);
    result->request.set_no_transfer(no_transfer);
    result->request.set_plan_digest(plan_digest);
    for (int route_id : active_route_ids) {
        result->request.add_active_route_ids(route_id);
    }

    for (const auto& tp_worker : tp_worker_infos_) {
        auto tp_worker_info = result->request.add_workers();
        tp_worker_info->set_ip(tp_worker.ip());
        tp_worker_info->set_cache_store_port(tp_worker.cache_store_port());
    }

    result->client_context   = std::make_shared<grpc::ClientContext>();

    const int64_t now_ms       = currentTimeMs();
    if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()
        || transfer_deadline_ms <= now_ms || transfer_deadline_ms > request_deadline_ms) {
        result->status = grpcStatusFromErrorInfo(ErrorInfo(
            transfer_deadline_ms <= now_ms ? ErrorCode::GENERATE_TIMEOUT : ErrorCode::INVALID_PARAMS,
            "StartLoad invalid deadline key=" + unique_key + " request_deadline=" + std::to_string(request_deadline_ms)
                + " transfer_deadline=" + std::to_string(transfer_deadline_ms)));
        return false;
    }
    result->timeout_ms = static_cast<int>(std::min<int64_t>(
        transfer_deadline_ms - now_ms, std::numeric_limits<int>::max()));
    result->client_context->set_deadline(
        std::chrono::system_clock::time_point(std::chrono::milliseconds(transfer_deadline_ms)));

    return RpcCompletionQueue::instance().submit(
        result->client_context,
        [&](grpc::CompletionQueue* cq, void* tag) {
            result->reader = result->stub->PrepareAsyncStartLoad(result->client_context.get(), result->request, cq);
            if (!result->reader) {
                return false;
            }
            result->reader->StartCall();
            result->reader->Finish(&result->response, &result->status, tag);
            return true;
        },
        [result](bool ok) { result->complete(ok); });
}

void DecodeLoadHelper::Result::setDoneCallback(std::function<void()> callback) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_) {
            done_callback_ = std::move(callback);
            return;
        }
    }
    if (callback) {
        callback();
    }
}

void DecodeLoadHelper::Result::cancel() {
    cancel_requested_.store(true, std::memory_order_release);
    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return;
        }
        if (client_context) {
            client_context->TryCancel();
        }
        // Logical cancellation is immediate; the CQ tag retains response, reader
        // and ClientContext until gRPC delivers Finish. Do not drain on this thread.
        done_         = true;
        success_      = false;
        error_code    = ErrorCode::CANCELLED;
        error_message = "StartLoad cancelled key=" + unique_key + " peer=" + server_addr;
        first_error_.record(ErrorInfo(error_code, error_message));
        total_cost_time_us = currentTimeUs() - start_time_us;
        callback           = std::move(done_callback_);
    }
    if (callback) {
        callback();
    }
}

void DecodeLoadHelper::Result::updateStreamFromResponse() {
    if (response.has_payload()) {
        const auto& payload = response.payload();
        side_channel_payload.has_first_token =
            payload.has_first_generate_token() || payload.first_generate_token_id() != 0;
        side_channel_payload.first_token_id   = payload.first_generate_token_id();
        side_channel_payload.total_reuse_len  = payload.total_reuse_len();
        side_channel_payload.local_reuse_len  = payload.local_reuse_len();
        side_channel_payload.remote_reuse_len = payload.remote_reuse_len();
        side_channel_payload.memory_reuse_len = payload.memory_reuse_len();
        side_channel_payload.disk_reuse_len   = payload.disk_reuse_len();
        side_channel_payload.has_data         = true;

        // Extract tensors from the payload map
        auto it_propose = payload.tensors().find("propose_tokens");
        if (it_propose != payload.tensors().end() && it_propose->second.has_tensor()) {
            const auto& tensor_pb = it_propose->second.tensor();
            if (tensor_pb.data_type() == TensorPB::INT32 && !tensor_pb.int32_data().empty()) {
                const auto* data  = reinterpret_cast<const int*>(tensor_pb.int32_data().data());
                size_t      count = tensor_pb.int32_data().size() / sizeof(int);
                side_channel_payload.propose_tokens.assign(data, data + count);
            }
        }

        auto it_probs = payload.tensors().find("propose_probs");
        if (it_probs != payload.tensors().end() && it_probs->second.has_tensor()) {
            side_channel_payload.propose_probs.CopyFrom(it_probs->second.tensor());
        }

        auto it_hidden = payload.tensors().find("propose_hidden");
        if (it_hidden != payload.tensors().end() && it_hidden->second.has_tensor()) {
            side_channel_payload.propose_hidden.CopyFrom(it_hidden->second.tensor());
        }

        auto it_pos = payload.tensors().find("position_ids");
        if (it_pos != payload.tensors().end() && it_pos->second.has_tensor()) {
            const auto& tensor_pb = it_pos->second.tensor();
            if (tensor_pb.data_type() == TensorPB::INT32 && !tensor_pb.int32_data().empty()) {
                const auto* data  = reinterpret_cast<const int32_t*>(tensor_pb.int32_data().data());
                size_t      count = tensor_pb.int32_data().size() / sizeof(int32_t);
                side_channel_payload.position_ids.assign(data, data + count);
            }
        }
    } else {
        extractLegacyStartLoadPayload(response, side_channel_payload);
    }

    RTP_LLM_LOG_DEBUG("DecodeLoadHelper::Result: parsed side-channel payload, first_token: %ld, total_reuse: %d",
                      side_channel_payload.first_token_id,
                      side_channel_payload.total_reuse_len);
}

void DecodeLoadHelper::Result::complete(bool ok) {
    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return;  // A late Finish must not overwrite logical cancellation.
        }
        success_ = false;
        error_code = ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED;
        if (!ok || !status.ok()) {
            const auto error = errorInfoFromGrpcStatus(
                status.ok() ? grpc::Status(grpc::StatusCode::INTERNAL, "Finish event failed") : status,
                "StartLoad peer=" + server_addr + " key=" + unique_key);
            error_code    = error.code();
            error_message = error.ToString();
        } else if (response.error_code() != ErrorCodePB::NONE_ERROR) {
            error_code    = transRPCErrorCode(response.error_code());
            error_message = response.error_message();
        } else {
            updateStreamFromResponse();
            if (!side_channel_payload.has_first_token) {
                error_message = "StartLoad response is missing the required first token";
            } else {
                success_   = true;
                error_code = ErrorCode::NONE_ERROR;
            }
        }
        first_error_.record(ErrorInfo(error_code, error_message));
        done_              = true;
        total_cost_time_us = currentTimeUs() - start_time_us;
        callback           = std::move(done_callback_);
    }
    if (callback) {
        callback();
    }
}

}  // namespace rtp_llm
