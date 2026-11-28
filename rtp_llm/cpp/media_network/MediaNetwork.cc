#include "rtp_llm/cpp/media_network/ContentDecoder.h"

#include <arpa/inet.h>
#include <ares.h>
#include <cerrno>
#include <poll.h>
#include <curl/curl.h>
#include <charconv>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstring>
#include <string_view>
#include <strings.h>
#include <deque>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
namespace {
using Clock   = std::chrono::steady_clock;
using Headers = std::map<std::string, std::string>;
using Bytes   = std::shared_ptr<const std::string>;

std::string lower(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
struct Failure: std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Policy {
    std::unordered_set<std::string>            denied;
    Headers                                    group_hosts;
    std::vector<std::pair<uint32_t, uint32_t>> blocked_networks;
    void                                       set_networks(const std::vector<std::string>& cidrs) {
        blocked_networks.clear();
        for (const auto& cidr : cidrs) {
            auto slash = cidr.find('/');
            if (slash == std::string::npos)
                throw std::invalid_argument("invalid download CIDR");
            auto    prefix = std::stoi(cidr.substr(slash + 1));
            in_addr address{};
            if (prefix < 0 || prefix > 32 || !inet_aton(cidr.substr(0, slash).c_str(), &address))
                throw std::invalid_argument("invalid download CIDR");
            auto mask = prefix == 0 ? 0u : UINT32_MAX << (32 - prefix);
            blocked_networks.emplace_back(ntohl(address.s_addr) & mask, mask);
        }
    }
    std::unordered_set<std::string> regions;
    bool                            region_list = false;
    bool                            internal    = false;
    bool                            rewrite_oss = false;
    std::string                     ca;
    size_t                          range_chunk_bytes = 0;
    bool                            reserve_response_body = false;
};
struct Endpoint {
    std::string origin, host;
    size_t      start, length;
};
Endpoint endpoint(const std::string& url, const Policy& policy, bool media) {
    if (url.find_first_of("\r\n\\") != std::string::npos || url.find('\0') != std::string::npos)
        throw Failure("invalid_url");
    auto start = url.find("://");
    if (start == std::string::npos || (lower(url.substr(0, start)) != "http" && lower(url.substr(0, start)) != "https"))
        throw Failure("invalid_url");
    start += 3;
    auto end       = url.find_first_of("/?#", start);
    auto authority = url.substr(start, end - start);
    if (authority.empty() || authority.find_first_of("@% \t") != std::string::npos)
        throw Failure("invalid_url");
    auto host_end = authority.find(':');
    if (authority.front() == '[') {
        auto bracket = authority.find(']');
        if (media || bracket == std::string::npos)
            throw Failure("invalid_url");
        host_end = bracket + 1;
    }
    auto spelling = authority.substr(0, host_end);
    auto host     = lower(spelling);
    while (!host.empty() && host.back() == '.')
        host.pop_back();
    if (host.empty() || std::any_of(host.begin(), host.end(), [](unsigned char c) { return c >= 128 || c < 33; }))
        throw Failure("invalid_url");
    if (media && policy.denied.count(host))
        throw Failure("blocked");
    in_addr ip{};
    if (media && policy.internal && inet_aton(host.c_str(), &ip)) {
        const uint32_t value = ntohl(ip.s_addr);
        for (auto [network, mask] : policy.blocked_networks)
            if ((value & mask) == network)
                throw Failure("blocked");
    }
    return {lower(url.substr(0, start) + authority), host, start, spelling.size()};
}
bool ends(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
std::string rewrite(std::string url, const Policy& policy) {
    auto e = endpoint(url, policy, true);
    if (!policy.rewrite_oss)
        return url;
    if (policy.region_list) {
        const std::string suffix = "-internal.aliyuncs.com";
        if (!ends(e.host, suffix))
            return url;
        auto label    = e.host.rfind('.', e.host.size() - suffix.size() - 1);
        label         = label == std::string::npos ? 0 : label + 1;
        auto regional = e.host.substr(label, e.host.size() - suffix.size() - label);
        if (regional.rfind("oss-", 0) == 0 && regional.substr(4).find('-') != std::string::npos
            && !policy.regions.count(regional.substr(4)))
            url.erase(e.start + e.host.size() - suffix.size(), 9);
    } else {
        for (const auto& [domain, group] : policy.group_hosts) {
            if (e.host == domain || ends(e.host, "." + domain)) {
                url.replace(e.start + e.host.size() - domain.size(), domain.size(), group);
                break;
            }
        }
    }
    return url;
}
struct Request {
    std::string method = "GET", url, proxy, cache_key, body;
    Headers     headers, proxy_headers;
    size_t      limit      = 0;
    long        connect_ms = 3000, idle_ms = 20000, budget_ms = 30000;
    int         attempts = 1;
    bool        media    = true;
    bool        observe  = false;
};
struct RangeMetadata {
    std::string url, etag;
    size_t      length;
    bool        strip_headers;
};
bool strong_etag(const std::string& tag);

struct QueueSignal {
    std::mutex              mu;
    std::condition_variable cv;
    size_t                  revision = 0;
    std::atomic<bool>       cancelled{false};

    void notify() {
        std::lock_guard<std::mutex> lock(mu);
        ++revision;
        cv.notify_one();
    }
};

struct Job {
    Request           req;
    Clock::time_point submitted = Clock::now(), deadline, finished;
    std::atomic<bool> canceled{false}, completed{false};
    std::string       error;
    Bytes             body;
    std::optional<RangeMetadata> ranges;
    bool                         range_attempted = false;
    bool                         observe         = false;
    std::array<int64_t, 11>      phases_us{};
    long              status  = 0;
    bool              hit     = false;
    int               retries = 0, connect_retries = 0, proxy_attempts = 0, fallbacks = 0;
    // One subscription per caller. Cancellation of the owner only stops the
    // shared transfer when every other caller has also cancelled.
    std::shared_ptr<std::atomic<size_t>> subscribers;
    std::weak_ptr<QueueSignal>           queue_signal;
    std::vector<std::shared_ptr<Job>>    followers;
    std::atomic<bool>                    released{false};
    bool                                 owner = false, joined = false;
    Clock::time_point                    queued_since = submitted;
    Clock::duration                      queue_time{};
    void                                 release_subscription() {
        if (subscribers && !released.exchange(true))
            subscribers->fetch_sub(1);
    }

    void cancel() {
        if (canceled.exchange(true))
            return;
        release_subscription();
        if (auto signal = queue_signal.lock()) {
            signal->cancelled = true;
            signal->notify();
        }
    }
    bool stop_transfer() const {
        return owner ? subscribers->load() == 0 : canceled.load();
    }
};
struct Transfer {
    explicit Transfer(Job& job): job(job) {}
    Job&                                            job;
    std::string                                     body;
    Clock::time_point                               last_io           = Clock::now();
    Clock::time_point                               submitted, admitted;
    bool                                            large             = false;
    bool                                            received_response = false;
    long                                            status            = 0;
    std::string                                     content_encoding;
    bool                                            has_encoding_header = false;
    bool                                            headers_complete    = false;
    size_t                                          content_length      = 0;
    bool                                            reserve_response_body = false;
    std::unique_ptr<rtp_llm::media::ContentDecoder> decoder;

    size_t                  range_chunk_bytes = 0;
    bool                    accepts_ranges = false, split_body = false;
    const std::atomic_bool* stop_ranges = nullptr;
    std::string             expected_range, content_range, etag;
    size_t                  range_length = 0;

    void append(std::string_view bytes) {
        if (range_length && bytes.size() > range_length - body.size())
            throw Failure("invalid_range");
        if (job.req.limit && bytes.size() > job.req.limit - body.size()) {
            large = true;
            throw Failure("too_large");
        }
        body.append(bytes);
    }

    bool canceled() const {
        return job.stop_transfer() || (stop_ranges && stop_ranges->load());
    }

    std::string failure() const {
        if (canceled())
            return "cancelled";
        auto now = Clock::now();
        if (now >= job.deadline || now - last_io >= std::chrono::milliseconds(job.req.idle_ms))
            return "timeout";
        return {};
    }
};
size_t write_body(char* data, size_t sz, size_t n, void* ptr) noexcept {
    auto&  t     = *static_cast<Transfer*>(ptr);
    size_t count = sz * n;
    t.last_io    = Clock::now();
    if (t.range_length && (t.status != 206 || t.stop_ranges->load()))
        return 0;
    if (t.job.req.media && t.status >= 300)
        return count;
    try {
        if (t.decoder)
            t.decoder->write({data, count});
        else {
            constexpr size_t reserve_after = 128 * 1024;
            constexpr size_t max_reserve    = 16 * 1024 * 1024;
            if (t.reserve_response_body && t.status == 200 && t.content_length <= max_reserve
                && t.content_length > t.body.capacity() && t.body.size() < reserve_after
                && count >= reserve_after - t.body.size()) {
                try {
                    t.body.reserve(t.content_length);
                } catch (const std::bad_alloc&) {
                    // The existing incremental path can still complete.
                }
            }
            t.append({data, count});
        }
    } catch (...) {
        return 0;
    }
    return count;
}
size_t read_header(char* bytes, size_t size, size_t count, void* ptr) noexcept {
    auto& transfer          = *static_cast<Transfer*>(ptr);
    transfer.last_io        = Clock::now();
    const size_t     length = size * count;
    std::string_view line(bytes, length);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
        line.remove_suffix(1);
    try {
        if (line.substr(0, 5) == "HTTP/") {
            auto index = line.find(' ');
            if (index != line.npos) {
                std::from_chars(bytes + index + 1, bytes + line.size(), transfer.status);
                transfer.received_response = transfer.status >= 200;
            }
            transfer.content_encoding.clear();
            transfer.has_encoding_header = false;
            transfer.headers_complete    = false;
            transfer.content_length      = 0;
            transfer.decoder.reset();
            transfer.etag.clear();
            transfer.content_range.clear();
            transfer.accepts_ranges = false;
        } else if (transfer.range_chunk_bytes && !transfer.headers_complete && line.size() >= 14
                   && !strncasecmp(bytes, "Accept-Ranges:", 14)) {
            const auto start = line.find_first_not_of(" \t", 14);
            if (start != line.npos) {
                const auto end                 = line.find_last_not_of(" \t");
                transfer.accepts_ranges        = lower(std::string(line.substr(start, end - start + 1))) == "bytes";
            }
        } else if ((transfer.range_chunk_bytes || transfer.range_length) && !transfer.headers_complete
                   && line.size() >= 5 && !strncasecmp(bytes, "ETag:", 5)) {
            transfer.etag = line.substr(5);
            transfer.etag.erase(0, transfer.etag.find_first_not_of(" \t"));
            transfer.etag.erase(transfer.etag.find_last_not_of(" \t") + 1);
        } else if (transfer.range_length && !transfer.headers_complete) {
            if (line.empty() && transfer.status >= 200) {
                transfer.headers_complete = true;
                if (transfer.status != 206 || transfer.content_range != transfer.expected_range)
                    return 0;
            } else if (line.size() >= 14 && !strncasecmp(bytes, "Content-Range:", 14)) {
                auto pos = line.find_first_not_of(" \t", 14);
                if (pos != line.npos)
                    transfer.content_range.append(line.substr(pos));
            } else if (line.size() >= 17 && !strncasecmp(bytes, "Content-Encoding:", 17)) {
                auto pos = line.find_first_not_of(" \t", 17);
                if (pos != line.npos && lower(std::string(line.substr(pos))) != "identity")
                    return 0;
            }
        } else if (transfer.job.req.media && transfer.status == 200 && !transfer.headers_complete) {
            if (line.empty()) {
                transfer.headers_complete = true;
                if (transfer.range_chunk_bytes && transfer.accepts_ranges
                    && transfer.content_length > transfer.range_chunk_bytes
                    && (!transfer.has_encoding_header || lower(transfer.content_encoding) == "identity")
                    && strong_etag(transfer.etag)) {
                    if (transfer.job.req.limit && transfer.content_length > transfer.job.req.limit)
                        transfer.large = true;
                    else
                        transfer.split_body = true;
                    // Small and unsupported responses continue on the original GET.
                    return 0;
                }
                if (transfer.has_encoding_header) {
                    transfer.decoder = rtp_llm::media::ContentDecoder::create(
                        lower(transfer.content_encoding),
                        [&transfer](std::string_view decoded) { transfer.append(decoded); },
                        [&transfer] {
                            if (transfer.job.stop_transfer() || Clock::now() >= transfer.job.deadline)
                                throw Failure("download stopped");
                        },
                        transfer.job.req.limit);
                }
                // A compressed Content-Length describes wire bytes. The media
                // limit is enforced on the output, including for chunked bodies.
                if (!transfer.decoder && transfer.job.req.limit && transfer.content_length > transfer.job.req.limit) {
                    transfer.large = true;
                    return 0;
                }
            } else if (line.size() >= 17 && !strncasecmp(bytes, "Content-Encoding:", 17)) {
                if (transfer.has_encoding_header)
                    transfer.content_encoding += ", ";
                transfer.has_encoding_header = true;
                auto value                   = line.find_first_not_of(" \t", 17);
                if (value != line.npos)
                    transfer.content_encoding.append(line.substr(value));
            } else if ((transfer.job.req.limit || transfer.range_chunk_bytes || transfer.reserve_response_body)
                       && line.size() >= 15
                       && !strncasecmp(bytes, "Content-Length:", 15)) {
                auto pos = line.find_first_not_of(" \t", 15);
                if (pos != line.npos)
                    std::from_chars(bytes + pos, bytes + line.size(), transfer.content_length);
            }
        }
    } catch (...) {
        return 0;
    }
    return length;
}
int progress(void* ptr, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
    auto& t = *static_cast<Transfer*>(ptr);
    return t.canceled() || Clock::now() >= t.job.deadline
           || Clock::now() - t.last_io >= std::chrono::milliseconds(t.job.req.idle_ms);
}
struct HeaderList {
    curl_slist* head = nullptr;
    ~HeaderList() {
        curl_slist_free_all(head);
    }
    void append(const std::string& header) {
        auto* next = curl_slist_append(head, header.c_str());
        if (!next)
            throw std::bad_alloc();
        head = next;
    }
};
bool safe_on_redirect(const std::string& name) {
    auto n = lower(name);
    return n == "user-agent" || n == "accept" || n == "accept-encoding" || n == "accept-language";
}
// A forward proxy resolves the origin itself. Preserve the internal policy's
// local hostname check, accepting the first successful A or AAAA answer.
void validate_proxied_media_host(const Endpoint& dest, Job& job) {
    in_addr literal{};
    if (dest.host.front() == '[' || inet_aton(dest.host.c_str(), &literal))
        return;
    ares_channel raw;
    if (ares_init(&raw) != ARES_SUCCESS)
        throw Failure("dns");
    struct Lookup {
        int  pending = 2;
        bool found   = false;
    } lookup;
    // Destroy the channel before lookup. Destruction completes pending callbacks.
    std::unique_ptr<ares_channeldata, decltype(&ares_destroy)> channel(raw, ares_destroy);
    auto completed = [](void* data, int status, int, struct hostent*) {
        auto& lookup = *static_cast<Lookup*>(data);
        lookup.found |= status == ARES_SUCCESS;
        --lookup.pending;
    };
    ares_gethostbyname(raw, dest.host.c_str(), AF_INET6, completed, &lookup);
    ares_gethostbyname(raw, dest.host.c_str(), AF_INET, completed, &lookup);
    while (!lookup.found && lookup.pending) {
        if (job.stop_transfer())
            throw Failure("cancelled");
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(job.deadline - Clock::now()).count();
        if (remaining <= 0)
            throw Failure("timeout");
        ares_socket_t sockets[ARES_GETSOCK_MAXNUM];
        pollfd        descriptors[ARES_GETSOCK_MAXNUM];
        int           bits  = ares_getsock(raw, sockets, ARES_GETSOCK_MAXNUM);
        int           count = 0;
        for (int i = 0; i < ARES_GETSOCK_MAXNUM; ++i) {
            short events = 0;
            if (ARES_GETSOCK_READABLE(bits, i))
                events |= POLLIN;
            if (ARES_GETSOCK_WRITABLE(bits, i))
                events |= POLLOUT;
            if (events)
                descriptors[count++] = {sockets[i], events, 0};
        }
        int ready = poll(descriptors, count, static_cast<int>(std::min<int64_t>(100, remaining)));
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready < 0)
            throw Failure("dns");
        if (!ready)
            ares_process_fd(raw, ARES_SOCKET_BAD, ARES_SOCKET_BAD);
        for (int i = 0; i < count && !lookup.found && lookup.pending; ++i) {
            const auto& fd = descriptors[i];
            if (fd.revents)
                ares_process_fd(raw,
                                fd.revents & (POLLIN | POLLERR | POLLHUP) ? fd.fd : ARES_SOCKET_BAD,
                                fd.revents & POLLOUT ? fd.fd : ARES_SOCKET_BAD);
        }
    }
    if (!lookup.found)
        throw Failure("dns");
}
// Only the I/O thread drives the multi handle. Workers lend it their configured
// easy handles for one transfer; the multi handle retains connections between transfers.
class CurlConnectionPool {
public:
    using Completion = std::function<void(CURLcode)>;

    explicit CurlConnectionPool(long capacity): capacity_(capacity), multi_(curl_multi_init(), curl_multi_cleanup) {
        if (!multi_ || curl_multi_setopt(multi_.get(), CURLMOPT_MAXCONNECTS, capacity) != CURLM_OK)
            throw Failure("curl connection pool initialization failed");
        io_ = std::thread([this] { drive(); });
    }
    ~CurlConnectionPool() {
        stop();
    }

    std::future<CURLcode> submit(CURL* easy, Transfer& transfer) {
        auto done   = std::make_shared<std::promise<CURLcode>>();
        auto result = done->get_future();
        submit(easy, transfer, [done](CURLcode code) { done->set_value(code); });
        return result;
    }

    void submit(CURL* easy, Transfer& transfer, Completion done) {
        transfer.submitted = Clock::now();
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!stopping_) {
                pending_.push_back({easy, transfer, std::move(done)});
                curl_multi_wakeup(multi_.get());
                return;
            }
        }
        done(CURLE_FAILED_INIT);
    }

    void stop() {
        if (!io_.joinable())
            return;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        curl_multi_wakeup(multi_.get());
        io_.join();
    }

private:
    struct Pending {
        CURL*                  easy;
        Transfer&              transfer;
        Completion             done;
    };
    struct Active {
        Transfer&  transfer;
        Completion done;
    };
    using Transfers = std::map<CURL*, Active>;
    void drive() {
        Transfers active;
        for (;;) {
            int poll_ms = 60000;
            std::vector<std::pair<Completion, CURLcode>> rejected;
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_)
                    break;
                const auto now = Clock::now();
                for (auto it = pending_.begin(); it != pending_.end();) {
                    auto& [easy, transfer, done] = *it;
                    const auto remaining =
                        std::chrono::ceil<std::chrono::milliseconds>(transfer.job.deadline - now).count();
                    if (transfer.canceled()) {
                        rejected.emplace_back(std::move(done), CURLE_ABORTED_BY_CALLBACK);
                    } else if (remaining <= 0) {
                        rejected.emplace_back(std::move(done), CURLE_OPERATION_TIMEDOUT);
                    } else if (active.size() >= capacity_) {
                        // A queued fragment still consumes its parent request's deadline.
                        poll_ms = static_cast<int>(std::min<int64_t>(poll_ms, remaining));
                        ++it;
                        continue;
                    } else {
                        transfer.last_io = transfer.admitted = now;
                        curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, static_cast<long>(remaining));
                        if (curl_multi_add_handle(multi_.get(), easy) == CURLM_OK)
                            active.emplace(easy, Active{transfer, std::move(done)});
                        else
                            rejected.emplace_back(std::move(done), CURLE_FAILED_INIT);
                    }
                    it = pending_.erase(it);
                }
            }
            for (auto& [done, code] : rejected)
                done(code);
            int running = 0;
            if (curl_multi_perform(multi_.get(), &running) != CURLM_OK)
                break;
            int  remaining = 0;
            bool completed = false;
            while (auto* msg = curl_multi_info_read(multi_.get(), &remaining)) {
                if (msg->msg == CURLMSG_DONE) {
                    completed     = true;
                    auto transfer = active.extract(msg->easy_handle);
                    auto result   = msg->data.result;
                    curl_multi_remove_handle(multi_.get(), transfer.key());
                    transfer.mapped().done(result);
                }
            }
            if (completed) {
                // A rejected range can cancel siblings holding the remaining
                // handles. Release those handles before waiting for more I/O.
                for (auto it = active.begin(); it != active.end();) {
                    auto entry = it++;
                    if (!entry->second.transfer.canceled())
                        continue;
                    auto transfer = active.extract(entry);
                    curl_multi_remove_handle(multi_.get(), transfer.key());
                    transfer.mapped().done(CURLE_ABORTED_BY_CALLBACK);
                }
                continue;
            }
            if (curl_multi_poll(multi_.get(), nullptr, 0, active.empty() ? poll_ms : std::min(poll_ms, 1000), nullptr)
                != CURLM_OK)
                break;
        }

        std::list<Pending> pending;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
            pending.swap(pending_);
        }
        for (auto& [easy, item] : active) {
            curl_multi_remove_handle(multi_.get(), easy);
            item.done(CURLE_FAILED_INIT);
        }
        for (auto& item : pending)
            item.done(CURLE_FAILED_INIT);
    }

    size_t                                                capacity_;
    std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)> multi_;
    std::mutex                                            mu_;
    std::list<Pending>                                    pending_;
    bool                                                  stopping_ = false;
    std::thread                                           io_;
};

void configure_transfer(CURL*              curl,
                        Transfer&          transfer,
                        HeaderList&        headers,
                        const Policy&      policy,
                        const std::string& url,
                        const std::string& proxy,
                        bool               strip) {
    auto&       job = transfer.job;
    const auto& req = job.req;
    auto        ms  = std::chrono::duration_cast<std::chrono::milliseconds>(job.deadline - Clock::now()).count();
    if (job.stop_transfer())
        throw Failure("cancelled");
    if (ms <= 0)
        throw Failure("timeout");
    transfer.reserve_response_body = policy.reserve_response_body && req.media;
    curl_easy_reset(curl);
    for (const auto& [name, value] : (proxy.empty() ? req.headers : req.proxy_headers)) {
        if (name.empty() || name.find_first_of(" :\r\n\t") != std::string::npos || name.find('\0') != std::string::npos
            || value.find_first_of("\r\n") != std::string::npos || value.find('\0') != std::string::npos)
            throw Failure("invalid_header");
        if (transfer.range_length && lower(name) == "accept-encoding")
            continue;
        if (!strip || safe_on_redirect(name))
            headers.append(name + ": " + value);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    if (req.media) {
        curl_easy_setopt(curl, CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS, 150L);
        // Let curl finish a redirect at the headers. Authorize every next hop here.
        if (curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 0L) != CURLE_OK
            || curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L) != CURLE_OK)
            throw Failure("transport");
    }
    curl_easy_setopt(curl, CURLOPT_PROXY, proxy.c_str());
    if (!proxy.empty())
        curl_easy_setopt(curl, CURLOPT_NOPROXY, "");
    if (!policy.ca.empty())
        curl_easy_setopt(curl, CURLOPT_CAINFO, policy.ca.c_str());
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, std::min<long>(ms, req.connect_ms));
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(ms));
    if (transfer.range_length || transfer.range_chunk_bytes)
        curl_easy_setopt(curl, CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
    if (transfer.range_length) {
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    } else {
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, req.media ? "gzip, deflate, br, zstd" : "");
    }
    if (req.media)
        curl_easy_setopt(curl, CURLOPT_HTTP_CONTENT_DECODING, 0L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers.head);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, read_header);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &transfer);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &transfer);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    if (req.method != "GET") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(req.body.size()));
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req.body.data());
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, req.method.c_str());
    }
}

enum Phase {
    HTTP_ADMISSION,
    DNS_VALIDATION,
    CURL_DNS,
    TCP_CONNECT,
    TLS_CONNECT,
    FIRST_BYTE,
    RESPONSE_BODY,
    RETRY_DELAY,
    RANGE_DOWNLOAD,
    RESULT_DELIVERY,
    RESULT_COPY
};

int64_t elapsed_us(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void record_http_phases(CURL* curl, Transfer& transfer) {
    if (!transfer.job.observe)
        return;
    auto& phases = transfer.job.phases_us;
    auto  get    = [curl](CURLINFO info) {
        curl_off_t value = 0;
        curl_easy_getinfo(curl, info, &value);
        return value;
    };
    if (transfer.admitted == Clock::time_point{}) {
        phases[HTTP_ADMISSION] += elapsed_us(transfer.submitted);
        return;
    }
    phases[HTTP_ADMISSION] += elapsed_us(transfer.submitted, transfer.admitted);
    const auto dns   = get(CURLINFO_NAMELOOKUP_TIME_T);
    const auto tcp   = get(CURLINFO_CONNECT_TIME_T);
    const auto tls   = get(CURLINFO_APPCONNECT_TIME_T);
    const auto ready = get(CURLINFO_PRETRANSFER_TIME_T);
    const auto first = get(CURLINFO_STARTTRANSFER_TIME_T);
    const auto total = get(CURLINFO_TOTAL_TIME_T);
    phases[CURL_DNS] += dns;
    phases[TCP_CONNECT] += std::max<curl_off_t>(0, tcp - dns);
    phases[TLS_CONNECT] += std::max<curl_off_t>(0, tls - tcp);
    // libcurl reports no start-transfer time if no response arrived.
    if (ready > 0)
        phases[FIRST_BYTE] += std::max<curl_off_t>(0, (first > 0 ? first : total) - ready);
    if (first > 0)
        phases[RESPONSE_BODY] += std::max<curl_off_t>(0, total - first);
}

std::string finish_exchange(CURL* curl, Transfer& transfer, CURLcode rc, const Policy& policy) {
    auto&       job = transfer.job;
    const auto& req = job.req;
    record_http_phases(curl, transfer);
    if (job.stop_transfer())
        throw Failure("cancelled");
    if (transfer.large)
        throw Failure("too_large");
    if (transfer.split_body && rc == CURLE_WRITE_ERROR) {
        job.status = transfer.status;
        return {};
    }
    if (rc != CURLE_OK && !(req.media && rc == CURLE_TOO_MANY_REDIRECTS)) {
        auto stopped = transfer.failure();
        if (!stopped.empty())
            throw Failure(stopped);
        if (rc == CURLE_OPERATION_TIMEDOUT) {
            double pretransfer = 0;
            curl_easy_getinfo(curl, CURLINFO_PRETRANSFER_TIME, &pretransfer);
            throw Failure(!transfer.received_response && pretransfer == 0 ? "connect_timeout" : "timeout");
        }
        if (rc == CURLE_URL_MALFORMAT || rc == CURLE_UNSUPPORTED_PROTOCOL)
            throw Failure("invalid_url");
        if (rc == CURLE_COULDNT_RESOLVE_HOST && req.media && policy.internal)
            throw Failure("dns");
        if (transfer.received_response)
            throw Failure("body_transport");
        throw Failure("transport");
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &job.status);
    if (req.media
        && (job.status == 301 || job.status == 302 || job.status == 303 || job.status == 307 || job.status == 308)) {
        char* redirect = nullptr;
        curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &redirect);
        if (redirect) {
            return redirect;
        }
    }
    if (transfer.decoder) {
        try {
            transfer.decoder->finish();
        } catch (const std::exception&) {
            throw Failure("body_transport");
        }
    }
    job.body = std::make_shared<const std::string>(std::move(transfer.body));
    return {};
}

void exchange(CurlConnectionPool& connections,
              CURL*               curl,
              Job&                job,
              const Policy&       policy,
              const std::string&  proxy,
              size_t              range_chunk_bytes = 0) {
    auto&       req   = job.req;
    std::string url   = req.media ? rewrite(req.url, policy) : req.url;
    const auto  first = endpoint(url, policy, req.media);
    bool        strip = false;
    for (int hop = 0; hop <= 30; ++hop) {
        auto dest = endpoint(url, policy, req.media);
        strip     = strip || dest.origin != first.origin;
        if (req.media && policy.internal && !proxy.empty()) {
            const auto start = Clock::now();
            try {
                validate_proxied_media_host(dest, job);
            } catch (...) {
                if (job.observe)
                    job.phases_us[DNS_VALIDATION] += elapsed_us(start);
                throw;
            }
            if (job.observe)
                job.phases_us[DNS_VALIDATION] += elapsed_us(start);
        }
        Transfer transfer{job};
        transfer.range_chunk_bytes = range_chunk_bytes;
        HeaderList headers;
        configure_transfer(curl, transfer, headers, policy, url, proxy, strip);
        auto rc = connections.submit(curl, transfer).get();
        auto redirect = finish_exchange(curl, transfer, rc, policy);
        if (redirect.empty()) {
            if (transfer.split_body) {
                job.range_attempted = true;
                job.ranges.emplace(
                    RangeMetadata{std::move(url), std::move(transfer.etag), transfer.content_length, strip});
            }
            return;
        }
        url = std::move(redirect);
    }
    throw Failure("redirect_limit");
}

// The I/O loop owns each easy handle until its completion future is ready.
struct RangePart {
    explicit RangePart(Job& job): transfer{job} {
        if (!curl)
            throw Failure("curl initialization failed");
    }
    ~RangePart() {
        if (done.valid())
            done.wait();
    }
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl{curl_easy_init(), curl_easy_cleanup};
    Transfer                                            transfer;
    HeaderList                                          headers;
    std::future<CURLcode>                               done;
};

bool valid_range(const Transfer& response, CURLcode code, const std::string& etag) {
    return code == CURLE_OK && response.status == 206 && response.headers_complete
           && response.body.size() == response.range_length && (response.etag.empty() || response.etag == etag);
}

bool strong_etag(const std::string& tag) {
    return tag.size() >= 2 && tag.front() == '"' && tag.back() == '"'
           && std::all_of(
               tag.begin() + 1, tag.end() - 1, [](unsigned char c) { return c != '"' && c > 0x20 && c != 0x7f; });
}

struct DirectDownload {
    explicit DirectDownload(std::shared_ptr<Job> job): job(std::move(job)) {}
    std::shared_ptr<Job>                                job;
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl{curl_easy_init(), curl_easy_cleanup};
    std::optional<Transfer>                             transfer;
    std::optional<HeaderList>                           headers;
    std::string                                         url, initial_url, origin;
    int                                                 hops = 0, connect_retries = 0;
    bool                                                strip = false;
};

class Network {
public:
    Network(int concurrency, size_t capacity, Policy policy):
        capacity_(capacity), concurrency_(concurrency), policy_(std::move(policy)) {
        if (concurrency < 1)
            throw std::invalid_argument("expected positive network concurrency");
        static const auto init = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (init != CURLE_OK)
            throw Failure("curl initialization failed");
        connections_ = std::make_unique<CurlConnectionPool>(concurrency);
        try {
            deadline_thread_ = std::thread([this] { expire_queued(); });
            dispatcher_      = std::thread([this] {
                while (auto job = take_ready(false))
                    start_direct(job);
            });
            // These workers only handle proxy DNS and segmented responses.
            for (int i = 0; i < std::min(concurrency, 8); ++i)
                workers_.emplace_back([this] { work(); });
        } catch (...) {
            close();
            throw;
        }
    }
    ~Network() {
        close();
    }
    std::shared_ptr<Job> submit(Request req) {
        if (req.attempts < 1 || req.attempts > 3 || req.connect_ms <= 0 || req.idle_ms <= 0
            || (req.media ? req.method != "GET" : (req.method != "GET" && req.method != "POST" && req.method != "PUT")))
            throw std::invalid_argument("invalid network request options");
        auto job      = std::make_shared<Job>();
        job->queue_signal     = queue_signal_;
        job->deadline         = job->submitted + std::chrono::milliseconds(req.budget_ms);
        job->observe          = req.observe;
        job->req      = std::move(req);
        const auto& request   = job->req;
        const bool  shareable = request.media && !request.cache_key.empty();
        if (shareable) {
            try {
                endpoint(rewrite(request.url, policy_), policy_, true);
            } catch (const Failure& error) {
                job->error = error.what();
            }
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (closed_)
            throw Failure("network is closed");
        if (shareable) {
            if (Clock::now() >= job->deadline)
                job->error = "timeout";
            if (!job->error.empty() || cached(*job)) {
                finish(job);
                return job;
            }
        }
        if (queue_.size() + blocking_queue_.size() + follower_count_ >= 4096) {
            discard_cancelled();
            if (queue_.size() + blocking_queue_.size() + follower_count_ >= 4096)
                throw Failure("network queue full");
        }
        if (shareable) {
            const auto [begin, end] = flights_.equal_range(request.cache_key);
            for (auto flight = begin; flight != end; ++flight) {
                if (!compatible(*flight->second, *job))
                    continue;
                auto& owner = *flight->second;
                // Increment only while a live caller still owns this transfer.
                auto live = owner.subscribers->load();
                while (live && !owner.subscribers->compare_exchange_weak(live, live + 1)) {}
                if (live) {
                    job->subscribers = owner.subscribers;
                    job->joined      = true;
                    try {
                        owner.followers.push_back(job);
                    } catch (...) {
                        job->release_subscription();
                        throw;
                    }
                    ++follower_count_;
                    return job;
                }
            }
            job->subscribers = std::make_shared<std::atomic<size_t>>(1);
            job->owner       = true;
        }
        enqueue(job);
        return job;
    }
    std::shared_ptr<Job> next() {
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this] { return !done_.empty() || drained(); });
        if (done_.empty())
            return {};
        auto job = done_.front();
        done_.pop_front();
        return job;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_)
                return;
            closed_ = true;
            for (const auto& entry : queue_)
                cancel_group(*entry.second);
            for (const auto& entry : blocking_queue_)
                cancel_group(*entry.second);
            for (const auto& job : running_)
                cancel_group(*job);
        }
        cv_.notify_all();
        blocking_cv_.notify_all();
        queue_signal_->notify();
        if (deadline_thread_.joinable())
            deadline_thread_.join();
        connections_->stop();
        if (dispatcher_.joinable())
            dispatcher_.join();
        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
        connections_.reset();
        done_cv_.notify_all();
    }

private:
    using Entry = std::pair<std::string, Bytes>;
    // Fresh requests and retries compete by the time they became eligible.
    using Pending = std::multimap<Clock::time_point, std::shared_ptr<Job>>;
    static bool blocking(const Job& job) {
        return !job.req.proxy.empty() || job.ranges.has_value();
    }
    bool drained() const {
        return closed_ && active_ == 0 && queue_.empty() && blocking_queue_.empty();
    }
    Pending::iterator queue_job(const std::shared_ptr<Job>& job) {
        auto&      queue            = blocking(*job) ? blocking_queue_ : queue_;
        const auto entry            = queue.emplace(job->queued_since, job);
        const bool earlier_deadline = queued_deadlines_.empty() || job->deadline < queued_deadlines_.begin()->first;
        try {
            queued_deadlines_.emplace(job->deadline, entry);
        } catch (...) {
            queue.erase(entry);
            throw;
        }
        if (earlier_deadline)
            queue_signal_->notify();
        (blocking(*job) ? blocking_cv_ : cv_).notify_one();
        return entry;
    }
    void remove_queued(Pending::iterator entry) {
        const auto [begin, end] = queued_deadlines_.equal_range(entry->second->deadline);
        const auto expiry       = std::find_if(begin, end, [&](const auto& pair) { return pair.second == entry; });
        queued_deadlines_.erase(expiry);
        auto& queue = blocking(*entry->second) ? blocking_queue_ : queue_;
        queue.erase(entry);
    }
    void enqueue(const std::shared_ptr<Job>& job) {
        const auto entry = queue_job(job);
        if (job->owner) {
            try {
                flights_.emplace(job->req.cache_key, job);
            } catch (...) {
                remove_queued(entry);
                throw;
            }
        }
    }
    static bool compatible(const Job& owner, const Job& caller) {
        const auto& a = owner.req;
        const auto& b = caller.req;
        // An earlier deadline gets independent admission and timeout handling.
        return owner.deadline <= caller.deadline && a.url == b.url && a.headers == b.headers && a.proxy == b.proxy
               && a.proxy_headers == b.proxy_headers && a.limit == b.limit && a.connect_ms == b.connect_ms
               && a.idle_ms == b.idle_ms && a.attempts == b.attempts;
    }
    static void cancel_group(Job& owner) {
        owner.cancel();
        for (const auto& follower : owner.followers)
            follower->cancel();
    }
    // Cache publication, in-flight lookup and completion all use mu_.
    void finish(const std::shared_ptr<Job>& job) {
        job->release_subscription();
        job->finished = Clock::now();
        job->req      = Request{};
        job->completed.store(true);
        done_.push_back(job);
        done_cv_.notify_one();
    }
    bool cached(Job& job) {
        if (!capacity_ || job.req.cache_key.empty())
            return false;
        auto it = entries_.find(job.req.cache_key);
        if (it == entries_.end())
            return false;
        lru_.splice(lru_.begin(), lru_, it->second);
        job.body = it->second->second;
        if (job.req.limit && job.body->size() > job.req.limit) {
            job.error = "too_large";
            job.body.reset();
            return true;
        }
        job.hit    = true;
        job.status = 200;
        return true;
    }
    void remember(const Job& job) {
        if (!capacity_ || job.req.cache_key.empty() || job.status != 200)
            return;
        if (entries_.count(job.req.cache_key))
            return;
        lru_.emplace_front(job.req.cache_key, job.body);
        try {
            entries_.emplace(lru_.front().first, lru_.begin());
        } catch (...) {
            lru_.pop_front();
            throw;
        }
        if (entries_.size() > capacity_) {
            entries_.erase(lru_.back().first);
            lru_.pop_back();
        }
    }
    bool read_ranges(Job& job, const RangeMetadata& meta, const std::string& proxy) {
        const size_t                           total  = meta.length;
        const size_t                           window = std::min<size_t>(4, concurrency_);
        std::atomic_bool                       stop{false};
        std::deque<std::unique_ptr<RangePart>> pending;
        std::string                            body;
        size_t                                 next = 0;
        try {
            if (job.req.limit)
                body.reserve(total);
            while (next < total || !pending.empty()) {
                while (next < total && pending.size() < window) {
                    auto&        part            = *pending.emplace_back(std::make_unique<RangePart>(job));
                    const size_t length          = std::min(policy_.range_chunk_bytes, total - next);
                    const auto   interval        = std::to_string(next) + "-" + std::to_string(next + length - 1);
                    part.transfer.range_length   = length;
                    part.transfer.expected_range = "bytes " + interval + "/" + std::to_string(total);
                    part.transfer.stop_ranges    = &stop;
                    part.headers.append("Range: bytes=" + interval);
                    part.headers.append("If-Match: " + meta.etag);
                    // HEAD already authorized this exact URL. A range redirect triggers
                    // a normal GET, which validates every hop and filters its headers.
                    configure_transfer(
                        part.curl.get(), part.transfer, part.headers, policy_, meta.url, proxy, meta.strip_headers);
                    auto done       = std::make_shared<std::promise<CURLcode>>();
                    auto completion = done->get_future();
                    connections_->submit(part.curl.get(), part.transfer, [&part, &meta, &stop, done](CURLcode code) {
                        if (!valid_range(part.transfer, code, meta.etag))
                            stop = true;
                        done->set_value(code);
                    });
                    part.done = std::move(completion);
                    next += length;
                }
                auto& part     = *pending.front();
                auto& response = part.transfer;
                if (!valid_range(response, part.done.get(), meta.etag) || stop.load()) {
                    stop = true;
                    return false;
                }
                body.append(response.body);
                pending.pop_front();
            }
        } catch (...) {
            stop = true;
            throw;
        }
        job.body   = std::make_shared<const std::string>(std::move(body));
        job.status = 200;
        return true;
    }

    size_t eligible_range_chunk(const Job& job, const std::string& proxy) const {
        if (!policy_.range_chunk_bytes || !job.req.media || job.range_attempted)
            return 0;
        const auto& headers = proxy.empty() ? job.req.headers : job.req.proxy_headers;
        for (const auto& [name, value] : headers) {
            auto normalized = lower(name);
            if (normalized == "range" || normalized.rfind("if-", 0) == 0)
                return 0;
        }
        return policy_.range_chunk_bytes;
    }
    void download(CURL* curl, Job& job, const std::string& proxy) {
        if (!job.ranges)
            exchange(*connections_, curl, job, policy_, proxy, eligible_range_chunk(job, proxy));
        if (job.ranges) {
            const auto meta = std::move(*job.ranges);
            job.ranges.reset();
            const auto started = Clock::now();
            const bool read    = read_ranges(job, meta, proxy);
            if (job.observe)
                job.phases_us[RANGE_DOWNLOAD] += elapsed_us(started);
            if (!read)
                exchange(*connections_, curl, job, policy_, proxy);
        }
    }

    void perform(CURL* curl, Job& job) {
        const auto& req = job.req;
        for (int retry = 0;; ++retry) {
            try {
                if (!req.proxy.empty())
                    ++job.proxy_attempts;
                try {
                    download(curl, job, req.proxy);
                } catch (const Failure& e) {
                    if (req.proxy.empty() || std::string(e.what()) != "transport")
                        throw;
                    ++job.fallbacks;
                    download(curl, job, "");
                }
                return;
            } catch (const Failure& e) {
                if (std::string(e.what()) != "connect_timeout")
                    throw;
                if (!req.media || retry >= 2 || Clock::now() >= job.deadline)
                    throw Failure("timeout");
                ++job.connect_retries;
            }
        }
    }

    static bool retry_status(const Job& job) {
        return job.req.media
               && (job.status == 408 || job.status == 429 || job.status == 500 || job.status == 502 || job.status == 503
                   || job.status == 504);
    }
    static bool retry_error(const std::string& reason) {
        return reason == "transport" || reason == "timeout" || reason == "body_transport" || reason == "dns";
    }
    static bool can_retry(const Job& job) {
        return job.retries + 1 < job.req.attempts;
    }

    // Run one outer attempt. A retryable result yields the worker back to the queue.
    bool run_attempt(CURL* curl, Job& job) {
        const auto& req = job.req;
        if (job.stop_transfer())
            throw Failure("cancelled");
        if (Clock::now() >= job.deadline)
            throw Failure("timeout");
        if (req.media && !job.owner)
            endpoint(rewrite(req.url, policy_), policy_, true);
        try {
            perform(curl, job);
            return can_retry(job) && retry_status(job);
        } catch (const Failure& e) {
            if (!can_retry(job) || !retry_error(e.what()))
                throw;
            return true;
        }
    }
    void schedule_retry(const std::shared_ptr<Job>& job, Clock::time_point now) {
        job->queued_since = now + std::chrono::milliseconds(200 << job->retries);
        if (job->observe)
            job->phases_us[RETRY_DELAY] += elapsed_us(now, std::min(job->deadline, job->queued_since));
        queue_job(job);
        ++job->retries;
    }
    void complete_followers(const std::shared_ptr<Job>& job, Clock::time_point now) {
        const auto [begin, end] = flights_.equal_range(job->req.cache_key);
        auto flight             = std::find_if(begin, end, [&](const auto& pair) { return pair.second == job; });
        flights_.erase(flight);
        follower_count_ -= job->followers.size();
        std::vector<std::shared_ptr<Job>> remaining;
        const bool exhausted = job->error == "timeout" && now + std::chrono::milliseconds(1) >= job->deadline;
        for (auto& waiter : job->followers) {
            if (waiter->canceled.load())
                waiter->error = "cancelled";
            else if (now >= waiter->deadline)
                waiter->error = "timeout";
            else if (exhausted) {
                waiter->queued_since = now;
                remaining.push_back(std::move(waiter));
                continue;
            } else {
                waiter->body   = job->body;
                waiter->error  = job->error;
                waiter->status = job->status;
            }
            finish(waiter);
        }
        job->followers.clear();
        if (remaining.empty())
            return;
        // The physical request used its own deadline. Keep the live callers
        // together and retry within the earliest remaining caller's budget.
        auto first = std::min_element(
            remaining.begin(), remaining.end(), [](const auto& a, const auto& b) { return a->deadline < b->deadline; });
        auto next = *first;
        remaining.erase(first);
        next->owner     = true;
        next->joined    = false;
        next->followers = std::move(remaining);
        follower_count_ += next->followers.size();
        enqueue(next);
    }
    void complete(const std::shared_ptr<Job>& job, Clock::time_point now) {
        if (job->error.empty() && !job->stop_transfer()) {
            try {
                remember(*job);
            } catch (const std::bad_alloc&) {
                // Cache publication is optional once the body has been downloaded.
            }
        }
        if (job->owner)
            complete_followers(job, now);
        if (job->canceled.load())
            job->error = "cancelled";
        finish(job);
    }
    static void account_queue(Job& job, Clock::time_point now) {
        if (now <= job.queued_since)
            return;
        job.queue_time += now - job.queued_since;
        for (const auto& waiter : job.followers)
            waiter->queue_time += now - std::max(job.queued_since, waiter->queued_since);
    }
    void discard_cancelled() {
        auto now = Clock::now();
        for (auto* queue : {&queue_, &blocking_queue_}) {
            for (auto it = queue->begin(); it != queue->end();) {
                auto entry = it++;
                auto job   = entry->second;
                if (job->stop_transfer()) {
                    remove_queued(entry);
                    account_queue(*job, now);
                    job->error = "cancelled";
                    complete(job, now);
                }
            }
        }
        for (auto& flight : flights_) {
            auto&      followers = flight.second->followers;
            const auto remaining = std::remove_if(followers.begin(), followers.end(), [&](const auto& caller) {
                if (!caller->canceled.load())
                    return false;
                caller->error = "cancelled";
                finish(caller);
                --follower_count_;
                return true;
            });
            followers.erase(remaining, followers.end());
        }
    }
    // Initial requests and retries both expire while all transfer slots are busy.
    void expire_queued() {
        auto& signal = *queue_signal_;
        for (;;) {
            size_t revision;
            {
                std::lock_guard<std::mutex> lock(signal.mu);
                revision = signal.revision;
            }
            auto next = Clock::time_point::max();
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (closed_)
                    return;
                if (signal.cancelled.exchange(false))
                    discard_cancelled();
                while (!queued_deadlines_.empty()) {
                    auto       first = queued_deadlines_.begin();
                    const auto now   = Clock::now();
                    if (now < first->first) {
                        next = first->first;
                        break;
                    }
                    auto entry = first->second;
                    auto job   = entry->second;
                    queued_deadlines_.erase(first);
                    auto& queue = blocking(*job) ? blocking_queue_ : queue_;
                    queue.erase(entry);
                    account_queue(*job, now);
                    job->error = job->stop_transfer() ? "cancelled" : "timeout";
                    complete(job, now);
                }
            }
            std::unique_lock<std::mutex> lock(signal.mu);
            auto                         changed = [&] { return revision != signal.revision; };
            if (next == Clock::time_point::max())
                signal.cv.wait(lock, changed);
            else
                signal.cv.wait_until(lock, next, changed);
        }
    }
    std::shared_ptr<Job> take_ready(bool blocking_worker) {
        auto&                        queue = blocking_worker ? blocking_queue_ : queue_;
        auto&                        cv    = blocking_worker ? blocking_cv_ : cv_;
        std::unique_lock<std::mutex> lock(mu_);
        for (;;) {
            if (queue.empty()) {
                if (closed_)
                    return {};
                cv.wait(lock);
                continue;
            }
            if (!closed_ && active_ >= concurrency_) {
                cv.wait(lock);
                continue;
            }
            auto       entry    = queue.begin();
            const auto eligible = entry->first;
            const auto now      = Clock::now();
            if (!closed_ && now < eligible) {
                cv.wait_until(lock, eligible);
                continue;
            }
            auto job = entry->second;
            remove_queued(entry);
            account_queue(*job, now);
            running_.insert(job);
            ++active_;
            return job;
        }
    }
    void finish_attempt(const std::shared_ptr<Job>& job, bool retry) {
        std::lock_guard<std::mutex> lock(mu_);
        running_.erase(job);
        --active_;
        cv_.notify_one();
        blocking_cv_.notify_one();
        const auto now = Clock::now();
        if (retry && !job->stop_transfer()) {
            if (now >= job->deadline)
                job->error = "timeout";
            else {
                try {
                    schedule_retry(job, now);
                    return;
                } catch (const std::exception& e) {
                    job->error = e.what();
                }
            }
        }
        complete(job, now);
        if (drained())
            done_cv_.notify_all();
    }
    void direct_error(const std::shared_ptr<Job>& job, const std::string& reason) {
        const bool retry = can_retry(*job) && retry_error(reason);
        if (!retry)
            job->error = reason;
        finish_attempt(job, retry);
    }
    void submit_direct(const std::shared_ptr<DirectDownload>& state) {
        auto& job = *state->job;
        try {
            if (state->hops > 30)
                throw Failure("redirect_limit");
            auto dest    = endpoint(state->url, policy_, job.req.media);
            state->strip = state->strip || dest.origin != state->origin;
            state->transfer.emplace(job);
            state->transfer->range_chunk_bytes = eligible_range_chunk(job, "");
            state->headers.emplace();
            configure_transfer(
                state->curl.get(), *state->transfer, *state->headers, policy_, state->url, "", state->strip);
            connections_->submit(
                state->curl.get(), *state->transfer, [this, state](CURLcode code) { direct_completed(state, code); });
        } catch (const std::exception& error) {
            direct_error(state->job, error.what());
        }
    }
    void direct_completed(const std::shared_ptr<DirectDownload>& state, CURLcode code) {
        auto& job = *state->job;
        try {
            auto redirect = finish_exchange(state->curl.get(), *state->transfer, code, policy_);
            if (!redirect.empty()) {
                state->url = std::move(redirect);
                ++state->hops;
                submit_direct(state);
                return;
            }
            if (state->transfer->split_body) {
                job.range_attempted = true;
                job.ranges.emplace(RangeMetadata{std::move(state->url),
                                                 std::move(state->transfer->etag),
                                                 state->transfer->content_length,
                                                 state->strip});
                std::lock_guard<std::mutex> lock(mu_);
                if (closed_ || job.stop_transfer())
                    throw Failure("cancelled");
                job.queued_since = Clock::now();
                queue_job(state->job);
                running_.erase(state->job);
                --active_;
                cv_.notify_one();
                return;
            }
        } catch (const std::exception& error) {
            std::string reason = error.what();
            if (reason == "connect_timeout") {
                if (job.req.media && state->connect_retries < 2 && Clock::now() < job.deadline) {
                    ++state->connect_retries;
                    ++job.connect_retries;
                    state->url   = state->initial_url;
                    state->hops  = 0;
                    state->strip = false;
                    submit_direct(state);
                    return;
                }
                reason = "timeout";
            }
            direct_error(state->job, reason);
            return;
        }
        finish_attempt(state->job, can_retry(job) && retry_status(job));
    }
    void start_direct(const std::shared_ptr<Job>& job) {
        try {
            auto state = std::make_shared<DirectDownload>(job);
            if (!state->curl)
                throw Failure("curl initialization failed");
            state->url         = job->req.media ? rewrite(job->req.url, policy_) : job->req.url;
            state->initial_url = state->url;
            state->origin      = endpoint(state->url, policy_, job->req.media).origin;
            submit_direct(state);
        } catch (const std::exception& error) {
            direct_error(job, error.what());
        }
    }
    void work() {
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
        while (auto job = take_ready(true)) {
            bool retry = false;
            try {
                if (!curl)
                    throw Failure("curl initialization failed");
                retry = run_attempt(curl.get(), *job);
            } catch (const std::exception& e) {
                job->error = e.what();
            }
            finish_attempt(job, retry);
        }
    }
    size_t capacity_, concurrency_, active_ = 0, follower_count_ = 0;
    Policy                                                      policy_;
    std::unordered_multimap<std::string, std::shared_ptr<Job>>  flights_;
    bool                                                        closed_ = false;
    std::mutex                                                  mu_;
    std::condition_variable                                     cv_, blocking_cv_, done_cv_;
    std::shared_ptr<QueueSignal>                                queue_signal_ = std::make_shared<QueueSignal>();
    Pending                                                     queue_, blocking_queue_;
    std::multimap<Clock::time_point, Pending::iterator>         queued_deadlines_;
    std::thread                                                 deadline_thread_;
    std::thread                                                 dispatcher_;
    std::deque<std::shared_ptr<Job>>                            done_;
    std::unordered_set<std::shared_ptr<Job>>                    running_;
    std::unique_ptr<CurlConnectionPool>                         connections_;
    std::vector<std::thread>                                    workers_;
    std::list<Entry>                                            lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> entries_;
};
}  // namespace

PYBIND11_MODULE(_media_network, m) {
    py::class_<Policy>(m, "Policy")
        .def(py::init<>())
        .def("set_networks", &Policy::set_networks)
        .def_readwrite("denied", &Policy::denied)
        .def_readwrite("group_hosts", &Policy::group_hosts)
        .def_readwrite("regions", &Policy::regions)
        .def_readwrite("region_list", &Policy::region_list)
        .def_readwrite("internal", &Policy::internal)
        .def_readwrite("rewrite_oss", &Policy::rewrite_oss)
        .def_readwrite("ca", &Policy::ca)
        .def_readwrite("range_chunk_bytes", &Policy::range_chunk_bytes)
        .def_readwrite("reserve_response_body", &Policy::reserve_response_body);
    py::class_<Request>(m, "Request")
        .def(py::init<>())
        .def_readwrite("url", &Request::url)
        .def_readwrite("method", &Request::method)
        .def_readwrite("headers", &Request::headers)
        .def_readwrite("proxy", &Request::proxy)
        .def_readwrite("proxy_headers", &Request::proxy_headers)
        .def_readwrite("cache_key", &Request::cache_key)
        .def("set_body",
             [](Request& req, py::bytes bytes) {
                 char*      data;
                 Py_ssize_t length;
                 if (PyBytes_AsStringAndSize(bytes.ptr(), &data, &length))
                     throw py::error_already_set();
                 std::string owned;
                 {
                     py::gil_scoped_release unlocked;
                     owned.assign(data, length);
                 }
                 req.body = std::move(owned);
             })
        .def_readwrite("limit", &Request::limit)
        .def_readwrite("connect_ms", &Request::connect_ms)
        .def_readwrite("idle_ms", &Request::idle_ms)
        .def_readwrite("budget_ms", &Request::budget_ms)
        .def_readwrite("attempts", &Request::attempts)
        .def_readwrite("observe", &Request::observe)
        .def_readwrite("media", &Request::media);
    py::class_<Job, std::shared_ptr<Job>>(m, "Job").def("cancel", &Job::cancel).def("result", [](const Job& j) {
        if (!j.completed.load())
            throw std::logic_error("transfer still in progress");
        auto   delivery_started = j.observe ? Clock::now() : Clock::time_point{};
        size_t size   = j.body && j.error.empty() ? j.body->size() : 0;
        auto*  buffer = PyBytes_FromStringAndSize(nullptr, size);
        if (!buffer)
            throw py::error_already_set();
        auto bytes = py::reinterpret_steal<py::bytes>(buffer);
        if (size) {
            char* destination = PyBytes_AS_STRING(buffer);
            if (size >= 65536) {
                py::gil_scoped_release unlock;
                std::memcpy(destination, j.body->data(), size);
            } else {
                std::memcpy(destination, j.body->data(), size);
            }
        }
        py::tuple phases;
        if (j.observe) {
            auto values             = j.phases_us;
            values[RESULT_DELIVERY] = elapsed_us(j.finished, delivery_started);
            values[RESULT_COPY]     = elapsed_us(delivery_started);
            phases                  = py::tuple(py::cast(values));
        }
        return py::make_tuple(
            j.status,
            j.error,
            std::move(bytes),
            j.hit,
            j.retries,
            py::make_tuple(j.proxy_attempts, j.fallbacks, j.connect_retries),
            std::chrono::duration<double, std::milli>(j.finished - j.submitted - j.queue_time).count(),
            std::chrono::duration<double, std::milli>(j.queue_time).count(),
            j.joined,
            std::move(phases));
    });
    py::class_<Network>(m, "Network")
        .def(py::init<int, size_t, Policy>())
        .def("submit", [](Network& net, Request& req) { return net.submit(std::move(req)); })
        .def("next", &Network::next, py::call_guard<py::gil_scoped_release>())
        .def("close", &Network::close, py::call_guard<py::gil_scoped_release>());
}
