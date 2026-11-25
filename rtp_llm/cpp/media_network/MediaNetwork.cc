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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstring>
#include <string_view>
#include <strings.h>
#include <deque>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
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
};
struct Job {
    Request           req;
    Clock::time_point deadline, started, finished;
    std::atomic<bool> canceled{false}, completed{false};
    std::string       error;
    Bytes             body;
    long              status  = 0;
    bool              hit     = false;
    int               retries = 0, connect_retries = 0, proxy_attempts = 0, fallbacks = 0;
};
bool is_media_redirect(long status) {
    return status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
}
struct Transfer {
    Job&                                            job;
    std::string                                     body;
    Clock::time_point                               last_io           = Clock::now();
    bool                                            large             = false;
    bool                                            received_response = false;
    long                                            status            = 0;
    std::string                                     content_encoding;
    bool                                            has_encoding_header      = false;
    bool                                            headers_complete         = false;
    bool                                            has_redirect_location    = false;
    bool                                            stopped_at_final_headers = false;
    size_t                                          content_length           = 0;
    std::unique_ptr<rtp_llm::media::ContentDecoder> decoder;

    void append(std::string_view bytes) {
        if (job.req.limit && bytes.size() > job.req.limit - body.size()) {
            large = true;
            throw Failure("too_large");
        }
        body.append(bytes);
    }

    std::string failure() const {
        if (job.canceled.load())
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
    if (t.job.req.media && t.status != 200)
        return count;
    try {
        if (t.decoder)
            t.decoder->write({data, count});
        else
            t.append({data, count});
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
            transfer.has_encoding_header   = false;
            transfer.headers_complete      = false;
            transfer.has_redirect_location = false;
            transfer.content_length        = 0;
            transfer.decoder.reset();
        } else if (transfer.job.req.media && !transfer.headers_complete) {
            if (line.empty()) {
                transfer.headers_complete = true;
                if (transfer.status != 200) {
                    // Curl resolves Location for redirects that exchange() can follow.
                    if (transfer.status >= 200 && !transfer.has_redirect_location) {
                        transfer.stopped_at_final_headers = true;
                        return 0;
                    }
                    return length;
                }
                if (transfer.has_encoding_header) {
                    transfer.decoder = rtp_llm::media::ContentDecoder::create(
                        lower(transfer.content_encoding),
                        [&transfer](std::string_view decoded) { transfer.append(decoded); },
                        [&transfer] {
                            if (transfer.job.canceled.load() || Clock::now() >= transfer.job.deadline)
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
            } else if (transfer.status != 200) {
                if (is_media_redirect(transfer.status) && line.size() > 9 && !strncasecmp(bytes, "Location:", 9)
                    && line.find_first_not_of(" \t", 9) != line.npos)
                    transfer.has_redirect_location = true;
            } else if (line.size() >= 17 && !strncasecmp(bytes, "Content-Encoding:", 17)) {
                if (transfer.has_encoding_header)
                    transfer.content_encoding += ", ";
                transfer.has_encoding_header = true;
                auto value                   = line.find_first_not_of(" \t", 17);
                if (value != line.npos)
                    transfer.content_encoding.append(line.substr(value));
            } else if (transfer.job.req.limit && line.size() >= 15 && !strncasecmp(bytes, "Content-Length:", 15)) {
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
    return t.job.canceled.load() || Clock::now() >= t.job.deadline
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
        if (job.canceled.load())
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
    explicit CurlConnectionPool(long capacity): multi_(curl_multi_init(), curl_multi_cleanup) {
        if (!multi_ || curl_multi_setopt(multi_.get(), CURLMOPT_MAXCONNECTS, capacity) != CURLM_OK)
            throw Failure("curl connection pool initialization failed");
        io_ = std::thread([this] { drive(); });
    }
    ~CurlConnectionPool() {
        stop();
    }

    CURLcode perform(CURL* easy) {
        std::future<CURLcode> result;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_)
                return CURLE_FAILED_INIT;
            result = pending_.try_emplace(easy).first->second.get_future();
        }
        curl_multi_wakeup(multi_.get());
        return result.get();
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
    using Transfers = std::map<CURL*, std::promise<CURLcode>>;
    void drive() {
        Transfers active;
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stopping_)
                    break;
                while (!pending_.empty()) {
                    auto transfer = pending_.extract(pending_.begin());
                    if (curl_multi_add_handle(multi_.get(), transfer.key()) == CURLM_OK)
                        active.insert(std::move(transfer));
                    else
                        transfer.mapped().set_value(CURLE_FAILED_INIT);
                }
            }
            int running = 0;
            if (curl_multi_perform(multi_.get(), &running) != CURLM_OK)
                break;
            int remaining = 0;
            while (auto* msg = curl_multi_info_read(multi_.get(), &remaining)) {
                if (msg->msg == CURLMSG_DONE) {
                    auto transfer = active.extract(msg->easy_handle);
                    auto result   = msg->data.result;
                    curl_multi_remove_handle(multi_.get(), transfer.key());
                    transfer.mapped().set_value(result);
                }
            }
            if (curl_multi_poll(multi_.get(), nullptr, 0, active.empty() ? 60000 : 1000, nullptr) != CURLM_OK)
                break;
        }

        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
        for (auto& [easy, result] : active) {
            curl_multi_remove_handle(multi_.get(), easy);
            result.set_value(CURLE_FAILED_INIT);
        }
        for (auto& pending : pending_)
            pending.second.set_value(CURLE_FAILED_INIT);
        pending_.clear();
    }

    std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)> multi_;
    std::mutex                                            mu_;
    Transfers                                             pending_;
    bool                                                  stopping_ = false;
    std::thread                                           io_;
};

void exchange(CurlConnectionPool& connections, CURL* curl, Job& job, const Policy& policy, const std::string& proxy) {
    auto&       req   = job.req;
    std::string url   = req.media ? rewrite(req.url, policy) : req.url;
    const auto  first = endpoint(url, policy, req.media);
    bool        strip = false;
    for (int hop = 0; hop <= 30; ++hop) {
        auto dest = endpoint(url, policy, req.media);
        strip     = strip || dest.origin != first.origin;
        if (req.media && policy.internal && !proxy.empty())
            validate_proxied_media_host(dest, job);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(job.deadline - Clock::now()).count();
        if (job.canceled.load())
            throw Failure("cancelled");
        if (ms <= 0)
            throw Failure("timeout");
        curl_easy_reset(curl);
        Transfer   transfer{job};
        HeaderList headers;
        for (const auto& [name, value] : (proxy.empty() ? req.headers : req.proxy_headers)) {
            if (name.empty() || name.find_first_of(" :\r\n\t") != std::string::npos
                || name.find('\0') != std::string::npos || value.find_first_of("\r\n") != std::string::npos
                || value.find('\0') != std::string::npos)
                throw Failure("invalid_header");
            if ((hop == 0 || lower(name) != "cookie") && (!strip || safe_on_redirect(name)))
                headers.append(name + ": " + value);
        }
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        if (req.media) {
            curl_easy_setopt(curl, CURLOPT_HAPPY_EYEBALLS_TIMEOUT_MS, 150L);
            curl_easy_setopt(curl, CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
            // Keep origin cookies across authorized redirect hops of this job.
            curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "");
            // Let curl resolve the redirect, draining the body if it can reuse
            // the connection. Authorize each next hop before sending it.
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
        curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, req.media ? "gzip, deflate, br, zstd" : "");
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
        auto rc = connections.perform(curl);
        if (job.canceled.load())
            throw Failure("cancelled");
        if (transfer.large)
            throw Failure("too_large");
        if (rc == CURLE_WRITE_ERROR && transfer.stopped_at_final_headers) {
            job.status = transfer.status;
            job.body.reset();
            return;
        }
        if (rc != CURLE_OK && !(req.media && rc == CURLE_TOO_MANY_REDIRECTS)) {
            long connect_status = 0;
            if (req.media && !proxy.empty())
                curl_easy_getinfo(curl, CURLINFO_HTTP_CONNECTCODE, &connect_status);
            const bool received_response = transfer.received_response || connect_status >= 200;
            auto stopped = transfer.failure();
            if ((rc == CURLE_OPERATION_TIMEDOUT || (rc == CURLE_ABORTED_BY_CALLBACK && stopped == "timeout"))
                && Clock::now() < job.deadline) {
                // libcurl can stamp PRETRANSFER when an unconnected transfer
                // jumps directly to completion. A reused TCP socket keeps its
                // primary IP even when its connect time for this request is zero.
                curl_off_t connected = 0;
                char* primary_ip = nullptr;
                curl_easy_getinfo(curl, CURLINFO_CONNECT_TIME_T, &connected);
                curl_easy_getinfo(curl, CURLINFO_PRIMARY_IP, &primary_ip);
                const bool unconnected = !received_response && connected == 0 && (!primary_ip || !*primary_ip);
                throw Failure(unconnected ? "connect_timeout" : "timeout");
            }
            if (!stopped.empty())
                throw Failure(stopped);
            if (rc == CURLE_OPERATION_TIMEDOUT)
                throw Failure("timeout");
            if (rc == CURLE_URL_MALFORMAT || rc == CURLE_UNSUPPORTED_PROTOCOL)
                throw Failure("invalid_url");
            if (rc == CURLE_COULDNT_RESOLVE_HOST && req.media && policy.internal)
                throw Failure("dns");
            if (received_response)
                throw Failure("body_transport");
            throw Failure("transport");
        }
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &job.status);
        if (req.media && is_media_redirect(job.status)) {
            char* redirect = nullptr;
            curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &redirect);
            if (redirect) {
                url = redirect;
                continue;
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
        return;
    }
    throw Failure("redirect_limit");
}

class Network {
public:
    Network(int workers, size_t capacity, Policy policy): capacity_(capacity), policy_(std::move(policy)) {
        if (workers < 1)
            throw std::invalid_argument("expected at least one network worker");
        static const auto init = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (init != CURLE_OK)
            throw Failure("curl initialization failed");
        connections_ = std::make_unique<CurlConnectionPool>(workers);
        try {
            for (int i = 0; i < workers; ++i)
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
        job->deadline = Clock::now() + std::chrono::milliseconds(req.budget_ms);
        job->req      = std::move(req);
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_)
                throw Failure("network is closed");
            if (queue_.size() >= 4096)
                throw Failure("network queue full");
            queue_.push_back(job);
        }
        cv_.notify_one();
        return job;
    }
    std::shared_ptr<Job> next() {
        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [this] { return !done_.empty() || (closed_ && active_ == 0 && queue_.empty()); });
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
            for (const auto& job : queue_)
                job->canceled = true;
            for (const auto& job : running_)
                job->canceled = true;
        }
        cv_.notify_all();
        connections_->stop();
        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
        connections_.reset();
        done_cv_.notify_all();
    }

private:
    using Entry = std::pair<std::string, Bytes>;
    bool cached(Job& job) {
        if (!capacity_ || job.req.cache_key.empty())
            return false;
        std::lock_guard<std::mutex> lock(cache_mu_);
        auto                        it = entries_.find(job.req.cache_key);
        if (it == entries_.end())
            return false;
        lru_.splice(lru_.begin(), lru_, it->second);
        job.hit  = true;
        job.body = it->second->second;
        if (job.req.limit && job.body->size() > job.req.limit)
            throw Failure("too_large");
        job.status = 200;
        return true;
    }
    void remember(const Job& job) {
        if (!capacity_ || job.req.cache_key.empty() || job.status != 200)
            return;
        std::lock_guard<std::mutex> lock(cache_mu_);
        if (entries_.count(job.req.cache_key))
            return;
        lru_.emplace_front(job.req.cache_key, job.body);
        entries_.emplace(lru_.front().first, lru_.begin());
        if (entries_.size() > capacity_) {
            entries_.erase(lru_.back().first);
            lru_.pop_back();
        }
    }
    void perform(CURL* curl, Job& job) {
        const auto& req = job.req;
        for (int retry = 0;; ++retry) {
            try {
                if (!req.proxy.empty())
                    ++job.proxy_attempts;
                try {
                    exchange(*connections_, curl, job, policy_, req.proxy);
                } catch (const Failure& e) {
                    if (req.proxy.empty() || std::string(e.what()) != "transport")
                        throw;
                    ++job.fallbacks;
                    exchange(*connections_, curl, job, policy_, "");
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

    void run(CURL* curl, Job& job) {
        auto& req = job.req;
        if (job.canceled.load())
            throw Failure("cancelled");
        if (Clock::now() >= job.deadline)
            throw Failure("timeout");
        if (req.media)
            endpoint(rewrite(req.url, policy_), policy_, true);
        if (cached(job))
            return;
        for (int attempt = 0;; ++attempt) {
            bool transient = false;
            try {
                perform(curl, job);
                transient = req.media
                            && (job.status == 408 || job.status == 429 || job.status == 500 || job.status == 502
                                || job.status == 503 || job.status == 504);
                if (!transient || attempt + 1 >= req.attempts) {
                    remember(job);
                    return;
                }
            } catch (const Failure& e) {
                if ((std::string(e.what()) != "transport" && std::string(e.what()) != "timeout"
                     && std::string(e.what()) != "body_transport" && std::string(e.what()) != "dns")
                    || attempt + 1 >= req.attempts)
                    throw;
            }
            if (Clock::now() >= job.deadline)
                throw Failure("timeout");
            ++job.retries;
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_until(lock,
                           std::min(job.deadline, Clock::now() + std::chrono::milliseconds(200 * (1 << attempt))),
                           [&job] { return job.canceled.load(); });
        }
    }
    void work() {
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
                if (queue_.empty())
                    return;
                job = std::move(queue_.front());
                queue_.pop_front();
                running_.insert(job);
                ++active_;
            }
            job->started = Clock::now();
            try {
                if (!curl)
                    throw Failure("curl initialization failed");
                // easy_reset preserves cookies. A worker reuses the handle, so
                // start each independent request with a fresh cookie jar.
                curl_easy_setopt(curl.get(), CURLOPT_COOKIELIST, "ALL");
                run(curl.get(), *job);
            } catch (const std::exception& e) {
                job->error = e.what();
            }
            job->finished = Clock::now();
            job->req      = Request{};
            job->completed.store(true);
            {
                std::lock_guard<std::mutex> lock(mu_);
                running_.erase(job);
                --active_;
                done_.push_back(std::move(job));
            }
            done_cv_.notify_one();
        }
    }
    size_t                                                      capacity_, active_ = 0;
    Policy                                                      policy_;
    bool                                                        closed_ = false;
    std::mutex                                                  mu_, cache_mu_;
    std::condition_variable                                     cv_, done_cv_;
    std::deque<std::shared_ptr<Job>>                            queue_, done_;
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
        .def_readwrite("ca", &Policy::ca);
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
        .def_readwrite("media", &Request::media);
    py::class_<Job, std::shared_ptr<Job>>(m, "Job")
        .def("cancel", [](Job& j) { j.canceled = true; })
        .def("result", [](const Job& j) {
            if (!j.completed.load())
                throw std::logic_error("transfer still in progress");
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
            return py::make_tuple(j.status,
                                  j.error,
                                  std::move(bytes),
                                  j.hit,
                                  j.retries,
                                  py::make_tuple(j.proxy_attempts, j.fallbacks, j.connect_retries),
                                  std::chrono::duration<double, std::milli>(j.finished - j.started).count());
        });
    py::class_<Network>(m, "Network")
        .def(py::init<int, size_t, Policy>())
        .def("submit", [](Network& net, Request& req) { return net.submit(std::move(req)); })
        .def("next", &Network::next, py::call_guard<py::gil_scoped_release>())
        .def("close", &Network::close, py::call_guard<py::gil_scoped_release>());
}
