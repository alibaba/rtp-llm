#include "rtp_llm/cpp/cache/events/CurlKVCacheEventReporter.h"

#include <algorithm>
#include <atomic>
#include <curl/curl.h>
#include <limits>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::detail {
namespace {

struct CurlResponseBuffer {
    std::string* response = nullptr;
    bool         exceeded = false;
};

size_t appendCurlResponse(char* data, size_t size, size_t count, void* user_data) {
    auto* buffer = static_cast<CurlResponseBuffer*>(user_data);
    if (size != 0 && count > std::numeric_limits<size_t>::max() / size) {
        buffer->exceeded = true;
        return 0;
    }
    const size_t bytes = size * count;
    if (buffer->response == nullptr || buffer->response->size() > kKVCacheEventMaxResponseBytes
        || bytes > kKVCacheEventMaxResponseBytes - buffer->response->size()) {
        buffer->exceeded = true;
        return 0;
    }
    try {
        buffer->response->append(data, bytes);
        return bytes;
    } catch (...) {
        return 0;
    }
}

class CurlKVCacheEventReporter final: public KVCacheEventReporter {
public:
    CurlKVCacheEventReporter(std::string endpoint, int request_timeout_ms):
        endpoint_(std::move(endpoint)), request_timeout_ms_(std::max(request_timeout_ms, 1)) {
        // Publishers are constructed synchronously before their worker starts.
        // Keep process-wide initialization behind call_once; cleanup belongs
        // to process teardown so another dependency's curl state stays valid.
        static std::once_flag curl_init_once;
        static bool           curl_initialized = false;
        std::call_once(curl_init_once, [] { curl_initialized = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK; });
        initialized_ = curl_initialized;
    }

    bool post(const std::string& route, const std::string& request, std::string& response) noexcept override {
        if (!initialized_ || cancelled_.load(std::memory_order_acquire)) {
            return false;
        }
        try {
            response.clear();
            return postImpl(route, request, response);
        } catch (...) {
            return false;
        }
    }

    void cancel() noexcept override {
        cancelled_.store(true, std::memory_order_release);
    }

private:
    static int abortCancelledTransfer(void* user_data, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
        return static_cast<std::atomic<bool>*>(user_data)->load(std::memory_order_acquire) ? 1 : 0;
    }

    bool postImpl(const std::string& route, const std::string& request, std::string& response) {
        if (request.size() > static_cast<size_t>(std::numeric_limits<curl_off_t>::max())) {
            RTP_LLM_LOG_WARNING(
                "KVCM event request exceeds libcurl size range, route=%s bytes=%zu", route.c_str(), request.size());
            return false;
        }

        const std::string url  = endpoint_ + route;
        CURL*             curl = curl_easy_init();
        if (curl == nullptr) {
            return false;
        }

        char error_buffer[CURL_ERROR_SIZE] = {0};
        auto headers                       = curl_slist_append(nullptr, "Content-Type: application/json");
        if (headers == nullptr) {
            curl_easy_cleanup(curl);
            return false;
        }
        auto headers_with_accept = curl_slist_append(headers, "Accept: application/json");
        if (headers_with_accept == nullptr) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return false;
        }
        headers = headers_with_accept;

        CurlResponseBuffer response_buffer{&response, false};
        const bool         options_configured =
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str()) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_POST, 1L) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.data()) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.size())) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(request_timeout_ms_)) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(request_timeout_ms_)) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, appendCurlResponse) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, abortCancelledTransfer) == CURLE_OK
            && curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &cancelled_) == CURLE_OK;
        if (!options_configured) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            RTP_LLM_LOG_WARNING("KVCM event request could not configure required libcurl options, route=%s",
                                route.c_str());
            return false;
        }

        const CURLcode result      = curl_easy_perform(curl);
        long           status_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (response_buffer.exceeded) {
            RTP_LLM_LOG_WARNING(
                "KVCM event response exceeded %zu bytes, route=%s", kKVCacheEventMaxResponseBytes, route.c_str());
        }
        if (result != CURLE_OK || status_code < 200 || status_code >= 300) {
            if (cancelled_.load(std::memory_order_acquire)) {
                return false;
            }
            RTP_LLM_LOG_WARNING("KVCM event request failed, route=%s curl_code=%d http_status=%ld error=%s",
                                route.c_str(),
                                static_cast<int>(result),
                                status_code,
                                error_buffer);
            return false;
        }
        return true;
    }

private:
    std::string       endpoint_;
    int               request_timeout_ms_;
    bool              initialized_{false};
    std::atomic<bool> cancelled_{false};
};

}  // namespace

std::shared_ptr<KVCacheEventReporter> makeCurlKVCacheEventReporter(std::string endpoint, int request_timeout_ms) {
    return std::make_shared<CurlKVCacheEventReporter>(std::move(endpoint), request_timeout_ms);
}

}  // namespace rtp_llm::detail
