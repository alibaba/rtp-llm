// OTLP/HTTP DNS resolver regression tests against the repo-locked @curl.
//
// Background (OTEL_DNS_HANDOFF.md): the curl 7.60.0 synchronous-DNS build
// cannot bound name resolution by the configured HTTP timeout, and its
// alarm()+siglongjmp deadline path uses a process-directed SIGALRM plus a
// global jmp_buf, so delivery to any other thread longjmps across stacks
// (reproduced as __longjmp_chk SIGABRT on B300). The fix enables the c-ares
// asynchronous resolver (USE_ARES in 3rdparty/curl/curl.BUILD).
//
// These checks run hermetically: loopback UDP sockets on port 53 replace the
// DNS server (CURLOPT_DNS_SERVERS "127.0.0.1" without a port, so the c-ares
// channel target is exactly 127.0.0.1:53), so no real resolver or network is
// involved. Binding port 53 requires root/CAP_NET_BIND_SERVICE; the Bazel
// test environment (GPU container) runs as root, and the check fails loudly
// with an explicit message when the bind is not possible.
//
//   1. curl_version_info reports CURL_VERSION_ASYNCHDNS.
//   2. Black-hole DNS (bound port 53, never answered): the transfer deadline
//      (CURLOPT_TIMEOUT_MS) bounds the wall clock; no signal involved.
//   3. NXDOMAIN responder: definite resolution failure returns promptly
//      with CURLE_COULDNT_RESOLVE_HOST.
//   4. Export alongside busy worker threads with SIGALRM blocked in the
//      export thread: on the synchronous-DNS build the alarm()-based
//      deadline raises a process-directed SIGALRM that must land on another
//      thread and longjmps cross-stack through the global curl_jmpenv.
//   5. Crash-tier red test, gated to the synchronous-DNS build: a forked
//      child redirects /etc/resolv.conf at the black-hole server and runs
//      the same export scenario; the alarm()+siglongjmp cross-stack jump
//      kills the child by signal (SIGABRT/SIGSEGV, the online rank-1 core).
//      The parent restores /etc/resolv.conf unconditionally and reports the
//      signal as failure. With c-ares the check is skipped: no alarm path is
//      compiled in, so the scenario cannot crash by construction.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <curl/curl.h>

namespace {

constexpr int    kDnsPort          = 53;  // DNS well-known port
constexpr long   kDeadlineMs       = 800;
constexpr double kDeadlineSlackMs  = 1200;  // deadline + scheduling tolerance
constexpr long   kExportDeadlineMs = 3000;  // OTel http_timeout_ms default

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Bind 127.0.0.1:53 as the fake DNS server endpoint. The c-ares channel
// targets "127.0.0.1" without a port (this vendored snapshot's
// ares_set_servers_csv drops ports anyway, curl's CURLOPT_DNS_SERVERS call
// included), so the well-known port is where the queries land. Requires
// root/CAP_NET_BIND_SERVICE; the GPU-container Bazel environment runs as
// root. Returns -1 with a diagnostic on failure.
int bind_dns_port(int rcv_timeout_ms) {
    const int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (rcv_timeout_ms > 0) {
        struct timeval tv{static_cast<time_t>(rcv_timeout_ms / 1000),
                          static_cast<suseconds_t>((rcv_timeout_ms % 1000) * 1000)};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(kDnsPort);
    if (bind(fd, reinterpret_cast<const struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "cannot bind 127.0.0.1:%d (errno=%d; needs root in the GPU container)\n", kDnsPort, errno);
        close(fd);
        return -1;
    }
    return fd;
}

struct NxResponder {
    int               fd = -1;
    std::atomic<bool> stop{false};
    std::thread       thread;

    void start() {
        thread = std::thread([this] {
            // 12-byte header: [0..1] id, [2] flags high, [3] flags low (rcode).
            char               buf[512];
            struct sockaddr_in from{};
            while (!stop.load(std::memory_order_relaxed)) {
                socklen_t     flen = sizeof(from);
                const ssize_t n = recvfrom(fd, buf, sizeof(buf), 0, reinterpret_cast<struct sockaddr*>(&from), &flen);
                if (n < 12) {
                    continue;  // timeout or malformed; loop re-checks stop
                }
                buf[2] = static_cast<char>(0x80 | (buf[2] & 0x7f));  // QR=1, keep RD
                buf[3] = 0x03;                                       // rcode = NXDOMAIN
                // AN/NS/AR counts are already zero in the query; echo question.
                sendto(
                    fd, buf, static_cast<size_t>(n), 0, reinterpret_cast<const struct sockaddr*>(&from), sizeof(from));
            }
        });
    }
};

// dns_servers == nullptr skips CURLOPT_DNS_SERVERS entirely: the synchronous
// build does not support the option (CURLE_NOT_BUILT_IN) and resolves through
// the system resolver, i.e. /etc/resolv.conf.
long perform_once(const char* dns_servers, CURLcode* code, double* wall_ms, long deadline_ms) {
    CURL* handle = curl_easy_init();
    if (!handle) {
        return CURLE_FAILED_INIT;
    }
    curl_easy_setopt(handle, CURLOPT_URL, "http://collector.otel-dns-regression.test:4318/v1/traces");
    curl_easy_setopt(handle, CURLOPT_PROXY, "");
    // OTel C++ v1.21 http client does not set CURLOPT_NOSIGNAL; these tests
    // assert the resolver is signal-free by construction instead.
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, deadline_ms);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, "");
    if (dns_servers != nullptr) {
        const CURLcode dns_rc = curl_easy_setopt(handle, CURLOPT_DNS_SERVERS, dns_servers);
        if (dns_rc != CURLE_OK) {
            std::fprintf(stderr, "CURLOPT_DNS_SERVERS rejected: %d\n", static_cast<int>(dns_rc));
            curl_easy_cleanup(handle);
            return dns_rc;
        }
    }
    const double t0 = now_ms();
    *code           = curl_easy_perform(handle);
    *wall_ms        = now_ms() - t0;
    curl_easy_cleanup(handle);
    return CURLE_OK;
}

// The export scenario from OTEL_DNS_HANDOFF.md, shared by checks 4 and 5.
// Two busy worker threads (the threads a process-directed SIGALRM can land
// on), then an export thread mimicking the OTel OTLP/HTTP call: black-hole
// DNS, OTel-default 3000 ms deadline (the alarm path has second granularity,
// so a sub-second deadline would never raise it), SIGALRM blocked in the
// export thread so a deadline signal (if the resolver used one) must be
// delivered to another thread. Returns 0 when the export is bounded and
// signal-free; the whole process is expected to survive to this return.
int run_export_with_workers(const char* dns_servers) {
    std::atomic<bool>        stop{false};
    std::atomic<long>        spins{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 2; ++i) {
        workers.emplace_back([&stop, &spins] {
            // Busy inference-like load: the threads SIGALRM can land on.
            while (!stop.load(std::memory_order_relaxed)) {
                spins.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    CURLcode    code    = CURLE_OK;
    double      wall_ms = 0.0;
    std::thread exporter([&code, &wall_ms, dns_servers] {
        sigset_t blocked;
        sigemptyset(&blocked);
        sigaddset(&blocked, SIGALRM);
        pthread_sigmask(SIG_BLOCK, &blocked, nullptr);
        const int rc = static_cast<int>(perform_once(dns_servers, &code, &wall_ms, kExportDeadlineMs));
        if (rc != 0) {
            code = CURLE_FAILED_INIT;
        }
    });
    exporter.join();
    stop.store(true, std::memory_order_relaxed);
    for (auto& w : workers) {
        w.join();
    }

    std::fprintf(stderr,
                 "export alongside workers: curl_code=%d wall_ms=%.1f spins=%ld\n",
                 static_cast<int>(code),
                 wall_ms,
                 spins.load());
    if (code != CURLE_OPERATION_TIMEDOUT && code != CURLE_COULDNT_RESOLVE_HOST) {
        std::fprintf(stderr, "expected timeout or resolve failure, got %d\n", static_cast<int>(code));
        return 1;
    }
    if (wall_ms > kExportDeadlineMs + kDeadlineSlackMs) {
        std::fprintf(stderr,
                     "wall %.1f ms exceeds %.1f ms deadline budget\n",
                     wall_ms,
                     static_cast<double>(kExportDeadlineMs + kDeadlineSlackMs));
        return 1;
    }
    if (spins.load() <= 0) {
        std::fputs("worker threads did not run alongside the export\n", stderr);
        return 1;
    }
    return 0;
}

int check_capability() {
    const auto* version = curl_version_info(CURLVERSION_NOW);
    if (!(version->features & CURL_VERSION_ASYNCHDNS)) {
        std::fputs("OTLP HTTP requires asynchronous DNS for bounded, signal-free name resolution.\n", stderr);
        return 1;
    }
    return 0;
}

int check_blackhole_deadline() {
    const int fd = bind_dns_port(0);
    if (fd < 0) {
        return 1;
    }
    CURLcode  code    = CURLE_OK;
    double    wall_ms = 0.0;
    const int rc      = static_cast<int>(perform_once("127.0.0.1", &code, &wall_ms, kDeadlineMs));
    close(fd);
    if (rc != 0) {
        return 1;
    }
    std::fprintf(stderr,
                 "blackhole DNS: curl_code=%d wall_ms=%.1f (deadline=%ld)\n",
                 static_cast<int>(code),
                 wall_ms,
                 kDeadlineMs);
    if (code != CURLE_OPERATION_TIMEDOUT && code != CURLE_COULDNT_RESOLVE_HOST) {
        std::fprintf(stderr, "expected timeout or resolve failure, got %d\n", static_cast<int>(code));
        return 1;
    }
    if (wall_ms > kDeadlineMs + kDeadlineSlackMs) {
        std::fprintf(stderr, "wall %.1f ms exceeds %.1f ms deadline budget\n", wall_ms, kDeadlineMs + kDeadlineSlackMs);
        return 1;
    }
    return 0;
}

int check_nxdomain_fast_fail() {
    // 100ms receive timeout so the responder loop can observe stop.
    const int fd = bind_dns_port(100);
    if (fd < 0) {
        return 1;
    }

    NxResponder responder;
    responder.fd = fd;
    responder.start();

    CURLcode  code    = CURLE_OK;
    double    wall_ms = 0.0;
    const int rc      = static_cast<int>(perform_once("127.0.0.1", &code, &wall_ms, kDeadlineMs));
    responder.stop.store(true, std::memory_order_relaxed);
    responder.thread.join();
    close(fd);
    if (rc != 0) {
        return 1;
    }
    std::fprintf(stderr, "nxdomain DNS: curl_code=%d wall_ms=%.1f\n", static_cast<int>(code), wall_ms);
    if (code != CURLE_COULDNT_RESOLVE_HOST) {
        std::fprintf(stderr, "expected CURLE_COULDNT_RESOLVE_HOST(6), got %d\n", static_cast<int>(code));
        return 1;
    }
    if (wall_ms > kDeadlineMs + kDeadlineSlackMs) {
        std::fprintf(stderr, "definite resolution failure took %.1f ms; resolver stalled\n", wall_ms);
        return 1;
    }
    return 0;
}

int check_export_signal_freedom() {
    const int fd = bind_dns_port(0);
    if (fd < 0) {
        return 1;
    }
    const int rc = run_export_with_workers("127.0.0.1");
    close(fd);
    return rc;
}

// Crash-tier red test for the synchronous-DNS build, gated so it is skipped
// once the c-ares fix is in (CURL_VERSION_ASYNCHDNS). The forked child points
// /etc/resolv.conf at the black-hole server and resolves through the system
// resolver (CURLOPT_DNS_SERVERS is not available on the sync build); curl's
// alarm()-based resolve deadline then fires a process-directed SIGALRM that
// the export thread blocks, so it lands on a worker thread and siglongjmps
// through the global curl_jmpenv across stacks — the online rank-1 failure
// tier. The parent never lets the crash escape: it reaps the child, restores
// /etc/resolv.conf unconditionally, and reports WIFSIGNALED as a failure.
int check_sync_dns_abort_scenario() {
    const auto* version = curl_version_info(CURLVERSION_NOW);
    if (version->features & CURL_VERSION_ASYNCHDNS) {
        std::fputs("sync-DNS abort scenario: skipped (asynchronous resolver present)\n", stderr);
        return 0;
    }

    // Save the container's resolver configuration; restored below no matter
    // how the child exits.
    char  saved[4096];
    FILE* in = std::fopen("/etc/resolv.conf", "r");
    if (in == nullptr) {
        std::fprintf(stderr, "cannot read /etc/resolv.conf (errno=%d)\n", errno);
        return 1;
    }
    const size_t saved_len = std::fread(saved, 1, sizeof(saved) - 1, in);
    std::fclose(in);

    // The black-hole DNS server: parent binds port 53 and never answers, so
    // the child's resolver query waits until curl's alarm deadline.
    const int fd = bind_dns_port(0);
    if (fd < 0) {
        return 1;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed (errno=%d)\n", errno);
        close(fd);
        return 1;
    }
    if (pid == 0) {
        // Child: crash containment. Point the system resolver at the
        // black-hole and run the export scenario; the synchronous-DNS curl
        // is expected to kill this process (alarm+siglongjmp cross-stack).
        FILE* out = std::fopen("/etc/resolv.conf", "w");
        if (out != nullptr) {
            std::fputs("nameserver 127.0.0.1\n", out);
            std::fclose(out);
        }
        const int rc = run_export_with_workers(nullptr);
        std::_Exit(rc);
    }

    // Parent: reap the child, then restore the resolver configuration
    // unconditionally (also after a crash).
    int         wstatus = 0;
    const pid_t got     = waitpid(pid, &wstatus, 0);
    FILE*       out     = std::fopen("/etc/resolv.conf", "w");
    if (out != nullptr) {
        std::fwrite(saved, 1, saved_len, out);
        std::fclose(out);
    }
    close(fd);
    if (got != pid) {
        std::fprintf(stderr, "waitpid failed (errno=%d)\n", errno);
        return 1;
    }
    if (WIFSIGNALED(wstatus)) {
        std::fprintf(stderr,
                     "sync-DNS export killed by signal %d: alarm()+siglongjmp cross-stack abort "
                     "(the online rank-1 core reproduced)\n",
                     WTERMSIG(wstatus));
        return 1;
    }
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
        std::fprintf(stderr, "sync-DNS export child failed without the expected crash (wstatus=%d)\n", wstatus);
        return 1;
    }
    // The child survived the sync-DNS scenario: not the expected crash tier,
    // but not a regression to report either (assertion-level failures are
    // covered by the checks above).
    std::fputs("sync-DNS abort scenario: child exited cleanly (crash not reproduced)\n", stderr);
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += check_capability();
    failures += check_blackhole_deadline();
    failures += check_nxdomain_fast_fail();
    failures += check_export_signal_freedom();
    failures += check_sync_dns_abort_scenario();
    if (failures == 0) {
        std::fputs("curl_dns_config_test: all checks passed\n", stderr);
        return 0;
    }
    return 1;
}
