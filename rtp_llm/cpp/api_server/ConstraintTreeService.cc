#include "rtp_llm/cpp/api_server/ConstraintTreeService.h"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
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

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("status", status, status);
        json.Jsonize("version", version, version);
        json.Jsonize("requested_version", requested_version, requested_version);
        json.Jsonize("message", message, message);
        json.Jsonize("initialized", initialized, initialized);
        json.Jsonize("prefix_count", prefix_count, prefix_count);
    }
};

void prepareJsonResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
}

ConstraintTreeUpdateResponse makeResponse(std::string status, uint64_t requested_version, std::string message) {
    const auto snapshot = PrefixToCandidateTokens::instance()->snapshot();

    ConstraintTreeUpdateResponse response;
    response.status            = std::move(status);
    response.requested_version = requested_version;
    response.message           = std::move(message);
    response.initialized       = snapshot != nullptr;
    response.version           = snapshot ? snapshot->version() : 0;
    response.prefix_count      = snapshot ? snapshot->prefixCount() : 0;
    return response;
}

std::optional<uint64_t> extractVersion(std::string_view body) {
    const auto key = body.find("\"version\"");
    if (key == std::string_view::npos) {
        return std::nullopt;
    }
    auto cursor = body.find(':', key + sizeof("\"version\"") - 1);
    if (cursor == std::string_view::npos) {
        return std::nullopt;
    }
    ++cursor;
    while (cursor < body.size() && std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    const char* begin   = body.data() + cursor;
    const char* end     = body.data() + body.size();
    uint64_t    version = 0;
    const auto  parsed  = std::from_chars(begin, end, version);
    if (parsed.ec != std::errc() || parsed.ptr == begin || version == 0) {
        return std::nullopt;
    }
    return version;
}

}  // namespace

ConstraintTreeService::ConstraintTreeService():
    latest_requested_version_(PrefixToCandidateTokens::instance()->currentVersion()),
    update_thread_([this]() { updateLoop(); }) {}

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
    const auto  requested_version = extractVersion(body);
    if (!requested_version.has_value()) {
        writer->SetStatus(400, "Bad Request");
        writer->Write(autil::legacy::ToJsonString(
            makeResponse("invalid_request", 0, "request must contain a positive integer version"), true));
        return;
    }

    const uint64_t active_version = PrefixToCandidateTokens::instance()->currentVersion();
    std::string    response_status;
    std::string    response_message;
    int            response_code = 200;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (*requested_version < active_version || *requested_version < latest_requested_version_) {
            response_status  = "stale_version";
            response_message = "a newer tree version is active or already queued";
            response_code    = 409;
        } else if (*requested_version == active_version) {
            response_status  = "already_current";
            response_message = "tree version is already active";
        } else if (*requested_version == latest_requested_version_
                   && (update_state_ == "queued" || update_state_ == "loading")) {
            response_status  = "already_accepted";
            response_message = "tree version is already queued or loading";
        } else {
            latest_requested_version_ = *requested_version;
            pending_update_           = PendingUpdate{*requested_version, std::move(body)};
            update_state_             = "queued";
            update_message_           = "tree update queued";
            response_status           = "accepted";
            response_message          = "tree update accepted for background loading";
        }
    }

    if (response_code != 200) {
        writer->SetStatus(response_code, "Conflict");
    }
    writer->Write(
        autil::legacy::ToJsonString(makeResponse(response_status, *requested_version, response_message), true));
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
    if (status == "idle" && PrefixToCandidateTokens::instance()->initSuccess()) {
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

        const auto result = PrefixToCandidateTokens::instance()->updatePrefixDictFromJson(update.body);
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
            } else if (result.code == PrefixTreeUpdateCode::STALE_VERSION) {
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
            prefixTreeUpdateCodeName(result.code));
    }
}

}  // namespace rtp_llm
