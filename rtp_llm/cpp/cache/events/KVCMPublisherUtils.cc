#include "rtp_llm/cpp/cache/events/KVCMPublisherUtils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <charconv>
#include <limits>
#include <rapidjson/document.h>
#include <rapidjson/memorystream.h>
#include <rapidjson/reader.h>
#include <string_view>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm {
namespace {

constexpr size_t kMaxKVCMResponseNestingDepth = 64;

class KVCMResponseNestingGuard final: public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, KVCMResponseNestingGuard> {
public:
    bool StartObject() {
        return enterContainer();
    }

    bool EndObject(rapidjson::SizeType) {
        return leaveContainer();
    }

    bool StartArray() {
        return enterContainer();
    }

    bool EndArray(rapidjson::SizeType) {
        return leaveContainer();
    }

private:
    bool enterContainer() noexcept {
        if (depth_ >= kMaxKVCMResponseNestingDepth) {
            return false;
        }
        ++depth_;
        return true;
    }

    bool leaveContainer() noexcept {
        if (depth_ == 0) {
            return false;
        }
        --depth_;
        return true;
    }

private:
    size_t depth_{0};
};

struct KVCMResponseCodeMapping {
    int64_t                  numeric;
    std::string_view         name;
    detail::KVCMResponseCode code;
};

constexpr std::array<KVCMResponseCodeMapping, 17> kKVCMResponseCodeMappings = {{
    {0, "UNSPECIFIED", detail::KVCMResponseCode::UNSPECIFIED},
    {1, "OK", detail::KVCMResponseCode::OK},
    {2, "UNSUPPORTED", detail::KVCMResponseCode::UNSUPPORTED},
    {3, "INTERNAL_ERROR", detail::KVCMResponseCode::INTERNAL_ERROR},
    {4, "SERVICE_NOT_READY", detail::KVCMResponseCode::SERVICE_NOT_READY},
    {5, "INVALID_ARGUMENT", detail::KVCMResponseCode::INVALID_ARGUMENT},
    {6, "DUPLICATE_ENTITY", detail::KVCMResponseCode::DUPLICATE_ENTITY},
    {7, "REACH_MAX_ENTITY_CAPACITY", detail::KVCMResponseCode::REACH_MAX_ENTITY_CAPACITY},
    {8, "INSTANCE_NOT_EXIST", detail::KVCMResponseCode::INSTANCE_NOT_EXIST},
    {9, "SERVER_NOT_LEADER", detail::KVCMResponseCode::SERVER_NOT_LEADER},
    {10, "NODE_NOT_REGISTERED", detail::KVCMResponseCode::NODE_NOT_REGISTERED},
    {11, "SNAPSHOT_IN_PROGRESS", detail::KVCMResponseCode::SNAPSHOT_IN_PROGRESS},
    {13, "SNAPSHOT_RATE_LIMITED", detail::KVCMResponseCode::SNAPSHOT_RATE_LIMITED},
    {14, "SNAPSHOT_REQUIRED", detail::KVCMResponseCode::SNAPSHOT_REQUIRED},
    {20, "IO_ERROR", detail::KVCMResponseCode::IO_ERROR},
    {100, "UNKNOWN_ERROR", detail::KVCMResponseCode::UNKNOWN_ERROR},
    {65535, "ERROR_MAX", detail::KVCMResponseCode::ERROR_MAX},
}};

detail::KVCMResponseCode numericResponseCode(int64_t numeric) noexcept {
    const auto mapping = std::find_if(kKVCMResponseCodeMappings.begin(),
                                      kKVCMResponseCodeMappings.end(),
                                      [numeric](const auto& entry) { return entry.numeric == numeric; });
    return mapping == kKVCMResponseCodeMappings.end() ? detail::KVCMResponseCode::UNRECOGNIZED : mapping->code;
}

detail::KVCMResponseCode stringResponseCode(std::string_view text) noexcept {
    const auto mapping = std::find_if(kKVCMResponseCodeMappings.begin(),
                                      kKVCMResponseCodeMappings.end(),
                                      [text](const auto& entry) { return entry.name == text; });
    if (mapping != kKVCMResponseCodeMappings.end()) {
        return mapping->code;
    }

    int64_t numeric         = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), numeric);
    return error == std::errc{} && end == text.data() + text.size() ? numericResponseCode(numeric) :
                                                                      detail::KVCMResponseCode::UNRECOGNIZED;
}

struct ParsedResponseCode {
    bool                     structurally_valid = false;
    detail::KVCMResponseCode code               = detail::KVCMResponseCode::UNRECOGNIZED;
};

ParsedResponseCode jsonResponseCode(const rapidjson::Value& code) noexcept {
    if (code.IsString()) {
        return {true, stringResponseCode({code.GetString(), code.GetStringLength()})};
    }
    if (code.IsInt64()) {
        return {true, numericResponseCode(code.GetInt64())};
    }
    if (code.IsUint64()) {
        return {true,
                code.GetUint64() <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ?
                    numericResponseCode(static_cast<int64_t>(code.GetUint64())) :
                    detail::KVCMResponseCode::UNRECOGNIZED};
    }
    return {};
}

const rapidjson::Value*
findMember(const rapidjson::Value& object, const char* snake_case, const char* camel_case) noexcept {
    if (object.HasMember(snake_case)) {
        return &object[snake_case];
    }
    return object.HasMember(camel_case) ? &object[camel_case] : nullptr;
}

bool parseUint64(const rapidjson::Value& value, uint64_t& result) noexcept {
    if (value.IsUint64()) {
        result = value.GetUint64();
        return true;
    }
    if (!value.IsString()) {
        return false;
    }
    const std::string_view text(value.GetString(), value.GetStringLength());
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    return error == std::errc{} && end == text.data() + text.size();
}

}  // namespace

namespace detail {

bool isValidSnapshotVersionToken(std::string_view token) noexcept {
    if (token.size() != 32) {
        return false;
    }
    return std::all_of(token.begin(), token.end(), [](unsigned char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F');
    });
}

bool KVCMResponseInfo::ok() const noexcept {
    if (!parsed || header_code != KVCMResponseCode::OK) {
        return false;
    }
    return std::all_of(
        item_results.begin(), item_results.end(), [](KVCMResponseCode code) { return code == KVCMResponseCode::OK; });
}

KVCMResponseCode KVCMResponseInfo::firstFailure() const noexcept {
    if (!parsed || header_code != KVCMResponseCode::OK) {
        return header_code;
    }
    const auto failure = std::find_if(
        item_results.begin(), item_results.end(), [](KVCMResponseCode code) { return code != KVCMResponseCode::OK; });
    return failure == item_results.end() ? KVCMResponseCode::OK : *failure;
}

bool KVCMResponseInfo::hasCode(KVCMResponseCode code) const noexcept {
    if (!parsed) {
        return false;
    }
    return header_code == code || std::find(item_results.begin(), item_results.end(), code) != item_results.end();
}

bool KVCMResponseInfo::hasPermanentFailure() const noexcept {
    // A batch can contain both transient and deterministic item failures. Do
    // not let an earlier transient item hide a later configuration/protocol
    // error and create an unbounded authoritative-snapshot retry loop.
    return hasCode(KVCMResponseCode::UNSUPPORTED) || hasCode(KVCMResponseCode::INVALID_ARGUMENT)
           || hasCode(KVCMResponseCode::DUPLICATE_ENTITY) || hasCode(KVCMResponseCode::REACH_MAX_ENTITY_CAPACITY)
           || hasCode(KVCMResponseCode::INSTANCE_NOT_EXIST) || hasCode(KVCMResponseCode::ERROR_MAX);
}

bool KVCMResponseInfo::requiresRegistration() const noexcept {
    // ReportEvent can return INSTANCE_NOT_EXIST after KVCM loses or removes
    // the instance. RegisterInstance is idempotent, so report traffic should
    // recreate the instance before retrying. The same code returned by
    // RegisterInstance itself remains a permanent configuration failure; that
    // call site intentionally does not consult this predicate.
    // Inspect every item: an earlier transient result must not hide the lost
    // reporter lifecycle of a later event in the same batch.
    return hasCode(KVCMResponseCode::INSTANCE_NOT_EXIST) || hasCode(KVCMResponseCode::NODE_NOT_REGISTERED);
}

bool KVCMResponseInfo::requestsSnapshot() const noexcept {
    return snapshot_required || hasCode(KVCMResponseCode::SNAPSHOT_REQUIRED);
}

KVCMResponseInfo parseKVCMResponse(const std::string& response) noexcept {
    KVCMResponseInfo result;
    // Keep the parser safe when called through an injected or future
    // reporter that does not share CurlKVCacheEventReporter's receive cap.
    // This check runs before RapidJSON can allocate from untrusted input.
    if (response.size() > kKVCacheEventMaxResponseBytes || response.find('\0') != std::string::npos) {
        return result;
    }
    try {
        // RapidJSON's default recursive parser can exhaust a worker thread's
        // stack on a small but pathologically deep response. Validate syntax
        // and depth with the iterative SAX parser before constructing a DOM;
        // the same depth bound also keeps recursive DOM destruction safe.
        rapidjson::MemoryStream      input(response.data(), response.size());
        rapidjson::Reader            reader;
        KVCMResponseNestingGuard     nesting_guard;
        constexpr unsigned           kParseFlags = rapidjson::kParseIterativeFlag;
        const rapidjson::ParseResult shape       = reader.Parse<kParseFlags>(input, nesting_guard);
        if (!shape) {
            return result;
        }

        rapidjson::Document document;
        document.Parse<kParseFlags>(response.data(), response.size());
        if (document.HasParseError() || !document.IsObject() || !document.HasMember("header")) {
            return result;
        }
        const auto& header = document["header"];
        if (!header.IsObject() || !header.HasMember("status")) {
            return result;
        }
        const auto& status = header["status"];
        if (!status.IsObject() || !status.HasMember("code")) {
            return result;
        }
        const auto parsed_header_code = jsonResponseCode(status["code"]);
        if (!parsed_header_code.structurally_valid) {
            return result;
        }
        result.header_code = parsed_header_code.code;
        if (result.header_code == KVCMResponseCode::UNRECOGNIZED) {
            result.has_unrecognized_code = true;
        }

        if (const auto* item_results = findMember(document, "item_results", "itemResults")) {
            if (!item_results->IsArray()) {
                return result;
            }
            result.item_results.reserve(item_results->Size());
            for (const auto& item : item_results->GetArray()) {
                const auto parsed_code = jsonResponseCode(item);
                if (!parsed_code.structurally_valid) {
                    return result;
                }
                if (parsed_code.code == KVCMResponseCode::UNRECOGNIZED) {
                    result.has_unrecognized_code = true;
                }
                result.item_results.push_back(parsed_code.code);
            }
        }

        if (const auto* version = findMember(document, "committed_snapshot_version", "committedSnapshotVersion")) {
            if (!version->IsString()) {
                return result;
            }
            result.committed_snapshot_version.assign(version->GetString(), version->GetStringLength());
        }
        if (const auto* retry_after = findMember(document, "retry_after_ms", "retryAfterMs")) {
            if (!parseUint64(*retry_after, result.retry_after_ms)) {
                return result;
            }
        }
        if (const auto* snapshot_required = findMember(document, "snapshot_required", "snapshotRequired")) {
            if (!snapshot_required->IsBool()) {
                return result;
            }
            result.snapshot_required = snapshot_required->GetBool();
        }
        result.parsed = true;
    } catch (...) {
        return {};
    }
    return result;
}

std::string normalizeKVCacheEventEndpoint(std::string endpoint) {
    const auto scheme_end      = endpoint.find("://");
    const auto authority_start = scheme_end == std::string::npos ? endpoint.size() : scheme_end + 3;
    // Keep the scheme delimiter intact for malformed values such as
    // "http://" so validation can reject them deterministically.
    while (endpoint.size() > authority_start && endpoint.back() == '/') {
        endpoint.pop_back();
    }
    return endpoint;
}

namespace {

bool isValidAuthority(std::string_view authority, bool allow_bracketed_ipv6) noexcept {
    if (authority.empty() || authority.find_first_of("/?#@\\%") != std::string_view::npos
        || std::any_of(
            authority.begin(), authority.end(), [](unsigned char value) { return value <= 0x20 || value == 0x7f; })) {
        return false;
    }

    std::string_view port;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (!allow_bracketed_ipv6 || close == std::string_view::npos || close == 1
            || authority.find('[', 1) != std::string_view::npos
            || authority.find(']', close + 1) != std::string_view::npos) {
            return false;
        }
        const auto host = authority.substr(1, close - 1);
        // Brackets are reserved for IPv6 literals. Validate the complete
        // address instead of accepting any text containing two colons; a
        // malformed literal would otherwise pass startup and fail forever in
        // libcurl's retry loop.
        in6_addr          parsed_ipv6{};
        const std::string host_text(host);
        if (inet_pton(AF_INET6, host_text.c_str(), &parsed_ipv6) != 1) {
            return false;
        }
        const auto suffix = authority.substr(close + 1);
        if (!suffix.empty()) {
            if (suffix.front() != ':' || suffix.size() == 1) {
                return false;
            }
            port = suffix.substr(1);
        }
    } else {
        if (authority.find_first_of("[]") != std::string_view::npos) {
            return false;
        }
        const auto colon = authority.rfind(':');
        if (colon != std::string_view::npos) {
            if (authority.find(':') != colon || colon == 0 || colon + 1 == authority.size()) {
                return false;
            }
            port = authority.substr(colon + 1);
        }
        const auto host = authority.substr(0, authority.size() - (port.empty() ? 0 : port.size() + 1));
        if (host.empty() || host.front() == '.' || host.back() == '.' || host.find("..") != std::string_view::npos) {
            return false;
        }
    }

    if (!port.empty()) {
        uint32_t   parsed_port = 0;
        const auto parsed      = std::from_chars(port.data(), port.data() + port.size(), parsed_port);
        if (parsed.ec != std::errc{} || parsed.ptr != port.data() + port.size() || parsed_port == 0
            || parsed_port > 65535) {
            return false;
        }
    }
    return true;
}

bool hasValidPercentEncoding(std::string_view value) noexcept {
    constexpr std::string_view kHexDigits = "0123456789abcdefABCDEF";
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%') {
            continue;
        }
        if (i + 2 >= value.size() || kHexDigits.find(value[i + 1]) == std::string_view::npos
            || kHexDigits.find(value[i + 2]) == std::string_view::npos) {
            return false;
        }
        i += 2;
    }
    return true;
}

}  // namespace

bool isValidKVCacheEventHostIpPort(std::string_view host_ip_port) noexcept {
    // Keep equivalent to Python's _valid_host_ip_port(); both tests consume
    // config/test/kv_cache_event_validation_cases.inc.
    // KVCM parses location specs with StandardUri, whose authority parser does
    // not support bracketed IPv6. Accepting it here would make every ADD fail
    // remotely even though the generated URI looks syntactically plausible.
    // This value is also embedded in a URI authority and a KVCM location-id
    // component. Keep it ASCII and reject escapes so both parsers receive the
    // exact same identity bytes.
    return isValidAuthority(host_ip_port, /*allow_bracketed_ipv6=*/false)
           && std::all_of(host_ip_port.begin(), host_ip_port.end(), [](unsigned char value) { return value < 0x7f; })
           && host_ip_port.find('%') == std::string_view::npos;
}

bool isValidKVCacheEventIdentity(std::string_view identity) noexcept {
    // KVCM instance/group identities are protocol identifiers, not display
    // labels. Keeping them non-empty printable ASCII prevents invisible byte
    // differences and makes Python startup validation exactly reproducible in
    // direct C++ construction paths.
    return !identity.empty() && std::all_of(identity.begin(), identity.end(), [](unsigned char value) {
        return value > 0x20 && value < 0x7f;
    });
}

bool isValidKVCacheEventEndpoint(std::string_view endpoint) noexcept {
    // Keep equivalent to Python's _valid_manager_endpoint().
    constexpr std::string_view kHttpScheme     = "http://";
    constexpr std::string_view kHttpsScheme    = "https://";
    size_t                     authority_start = 0;
    if (endpoint.compare(0, kHttpScheme.size(), kHttpScheme) == 0) {
        authority_start = kHttpScheme.size();
    } else if (endpoint.compare(0, kHttpsScheme.size(), kHttpsScheme) == 0) {
        authority_start = kHttpsScheme.size();
    } else {
        return false;
    }

    if (authority_start >= endpoint.size() || endpoint.find_first_of("?#\\", authority_start) != std::string_view::npos
        || std::any_of(
            endpoint.begin(), endpoint.end(), [](unsigned char value) { return value <= 0x20 || value >= 0x7f; })
        || !hasValidPercentEncoding(endpoint.substr(authority_start))) {
        return false;
    }

    const auto authority_end = endpoint.find('/', authority_start);
    return isValidAuthority(
        endpoint.substr(authority_start,
                        (authority_end == std::string_view::npos ? endpoint.size() : authority_end) - authority_start),
        /*allow_bracketed_ipv6=*/true);
}

}  // namespace detail

}  // namespace rtp_llm
