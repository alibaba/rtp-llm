#include "rtp_llm/cpp/api_server/ConstraintTreeService.h"

#include <cstdint>
#include <string>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/models/logits_processor/ConstraintTreeCsr.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

class ConstraintTreeUpdateResponse: public autil::legacy::Jsonizable {
public:
    std::string status;
    uint64_t    version           = 0;
    uint64_t    requested_version = 0;
    std::string message;
    bool        initialized  = false;
    uint64_t    prefix_count = 0;
    uint64_t    edge_count   = 0;
    std::string mapping_fingerprint;
    std::string content_sha256;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("status", status, status);
        json.Jsonize("version", version, version);
        json.Jsonize("requested_version", requested_version, requested_version);
        json.Jsonize("message", message, message);
        json.Jsonize("initialized", initialized, initialized);
        json.Jsonize("prefix_count", prefix_count, prefix_count);
        json.Jsonize("edge_count", edge_count, edge_count);
        json.Jsonize("mapping_fingerprint", mapping_fingerprint, mapping_fingerprint);
        json.Jsonize("content_sha256", content_sha256, content_sha256);
    }
};

void prepareJsonResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
}

ConstraintTreeUpdateResponse makeResponse(std::string status, uint64_t requested_version, std::string message) {
    const auto snapshot = ConstraintTreeCsrManager::instance()->snapshot();

    ConstraintTreeUpdateResponse response;
    response.status              = std::move(status);
    response.requested_version   = requested_version;
    response.message             = std::move(message);
    response.initialized         = snapshot != nullptr;
    response.version             = snapshot ? snapshot->version() : 0;
    response.prefix_count        = snapshot ? snapshot->stateCount() : 0;
    response.edge_count          = snapshot ? snapshot->edgeCount() : 0;
    response.mapping_fingerprint = snapshot ? snapshot->mappingFingerprint() : "";
    response.content_sha256      = snapshot ? snapshot->contentSha256() : "";
    return response;
}

}  // namespace

ConstraintTreeService::ConstraintTreeService(DeviceBase* device, std::string mapping_json):
    latest_requested_version_(ConstraintTreeCsrManager::instance()->currentVersion()),
    mapping_json_(std::move(mapping_json)),
    device_(device) {
    // Tokenizer access happened before this service was created. HTTP/background
    // threads only read an immutable manifest and never acquire the Python GIL.
    class MappingStatus: public autil::legacy::Jsonizable {
    public:
        std::string mapping_fingerprint;
        int         vocab_size = 0, start_token_id = 0, end_token_id = 0;
        void        Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
            json.Jsonize("mapping_fingerprint", mapping_fingerprint, mapping_fingerprint);
            json.Jsonize("vocab_size", vocab_size, vocab_size);
            json.Jsonize("start_token_id", start_token_id, start_token_id);
            json.Jsonize("end_token_id", end_token_id, end_token_id);
        }
    } mapping;
    autil::legacy::FromJsonString(mapping, mapping_json_);
    mapping_fingerprint_ = mapping.mapping_fingerprint;
    mapping_status_json_ = autil::legacy::ToJsonString(mapping, true);
    update_thread_       = std::thread([this]() { updateLoop(); });
}

void ConstraintTreeService::constraintTreeMapping(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                                  bool                                                    full) {
    prepareJsonResponse(writer);
    if (mapping_fingerprint_.empty()) {
        writer->SetStatus(503, "Service Unavailable");
        writer->Write("{\"error\":\"Worker has no C-token SID mapping\"}");
        return;
    }
    writer->Write(full ? mapping_json_ : mapping_status_json_);
}

ConstraintTreeService::~ConstraintTreeService() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        pending_update_.reset();
    }
    condition_.notify_one();
    if (update_thread_.joinable()) {
        update_thread_.join();
    }
}

void ConstraintTreeService::updateConstraintTree(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                                 const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    std::string body              = request.GetBody();
    uint64_t    requested_version = 0;
    std::string requested_mapping, requested_content;
    const auto  header_result =
        ConstraintTreeCsrManager::peekVersion(body, requested_version, &requested_mapping, &requested_content);
    if (!header_result.ok()) {
        writer->SetStatus(400, "Bad Request");
        writer->Write(autil::legacy::ToJsonString(makeResponse("invalid_request", 0, header_result.message), true));
        return;
    }
    if (requested_mapping != mapping_fingerprint_) {
        writer->SetStatus(409, "Conflict");
        writer->Write(autil::legacy::ToJsonString(
            makeResponse("mapping_mismatch", requested_version, "Worker SID mapping fingerprint mismatch"), true));
        return;
    }

    const auto     active         = ConstraintTreeCsrManager::instance()->snapshot();
    const uint64_t active_version = active ? active->version() : 0;
    std::string    response_status;
    std::string    response_message;
    int            response_code = 200;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (requested_version < active_version || requested_version < latest_requested_version_) {
            response_status  = "stale_version";
            response_message = "a newer tree version is active or already queued";
            response_code    = 409;
        } else if (requested_version == latest_requested_version_ && !latest_requested_content_sha256_.empty()
                   && requested_content != latest_requested_content_sha256_) {
            response_status  = "version_conflict";
            response_message = "same version has different content";
            response_code    = 409;
        } else if (requested_version == active_version) {
            const bool same = active && active->mappingFingerprint() == requested_mapping
                              && active->contentSha256() == requested_content;
            response_status  = same ? "already_current" : "version_conflict";
            response_message = same ? "tree version is already active" : "same version has different content";
            response_code    = same ? 200 : 409;
        } else if (requested_version == latest_requested_version_
                   && (update_state_ == "queued" || update_state_ == "loading")) {
            const bool same = latest_requested_content_sha256_ == requested_content;
            response_status = same ? "already_accepted" : "version_conflict";
            response_message =
                same ? "tree version is already queued or loading" : "same version has different content";
            response_code = same ? 200 : 409;
        } else {
            latest_requested_version_        = requested_version;
            latest_requested_content_sha256_ = requested_content;
            pending_update_                  = PendingUpdate{requested_version, std::move(body)};
            update_state_                    = "queued";
            update_message_                  = "tree update queued";
            response_status                  = "accepted";
            response_message                 = "tree update accepted for background loading";
        }
    }

    if (response_code != 200) {
        writer->SetStatus(response_code, "Conflict");
    }
    writer->Write(
        autil::legacy::ToJsonString(makeResponse(response_status, requested_version, response_message), true));
    if (response_status == "accepted") {
        condition_.notify_one();
    }
}

void ConstraintTreeService::constraintTreeStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                                 const http_server::HttpRequest&) {
    prepareJsonResponse(writer);
    std::string status;
    std::string message;
    uint64_t    requested_version;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status            = update_state_;
        message           = update_message_;
        requested_version = latest_requested_version_;
    }
    if (status == "idle" && ConstraintTreeCsrManager::instance()->snapshot()) {
        status  = "ready";
        message = "constraint tree is ready";
    }
    writer->Write(autil::legacy::ToJsonString(makeResponse(status, requested_version, message), true));
}

void ConstraintTreeService::updateLoop() {
    while (true) {
        PendingUpdate update;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() { return stopping_ || pending_update_.has_value(); });
            if (stopping_) {
                return;
            }
            update = std::move(*pending_update_);
            pending_update_.reset();
            update_state_   = "loading";
            update_message_ = "parsing and loading constraint tree snapshot";
        }

        ConstraintTreeCsrUpdateResult result;
        try {
            // The loader owns a dedicated thread. CUDA/ROCm device selection and
            // the framework's current stream are thread-local, so initialize them
            // here before allocating or copying device buffers.
            if (device_ != nullptr) {
                device_->preRun();
            }
            result = ConstraintTreeCsrManager::instance()->updateFromBinary(update.body, device_, mapping_fingerprint_);
        } catch (const std::exception& e) {
            result = {ConstraintTreeCsrUpdateCode::RESOURCE_ERROR,
                      ConstraintTreeCsrManager::instance()->currentVersion(),
                      std::string("unexpected CSR load failure: ") + e.what()};
        } catch (...) {
            result = {ConstraintTreeCsrUpdateCode::RESOURCE_ERROR,
                      ConstraintTreeCsrManager::instance()->currentVersion(),
                      "unexpected non-standard CSR load failure"};
        }
        update.body.clear();
        update.body.shrink_to_fit();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_update_.has_value()) {
                update_state_   = "queued";
                update_message_ = "a newer tree update is queued";
            } else if (result.ok()) {
                update_state_   = "ready";
                update_message_ = result.message;
            } else if (result.code == ConstraintTreeCsrUpdateCode::STALE_VERSION) {
                update_state_   = "stale_version";
                update_message_ = result.message;
            } else {
                update_state_   = "failed";
                update_message_ = result.message;
            }
        }
        RTP_LLM_LOG_INFO(
            "constraint tree background update finished requested_version=[%llu], active_version=[%llu], status=[%s]",
            static_cast<unsigned long long>(update.version),
            static_cast<unsigned long long>(result.current_version),
            constraintTreeCsrUpdateCodeName(result.code));
    }
}

}  // namespace rtp_llm
