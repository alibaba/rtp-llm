#include "rtp_llm/cpp/utils/cuda_call_trace/CudaCallTrace.h"

#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

namespace trace = rtp_llm::cuda_trace;
namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::abort();
    }
}
std::string read(const std::filesystem::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
template<typename F>
bool eventually(F predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    do {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
std::filesystem::path find(const std::filesystem::path& directory, const std::string& suffix) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().string();
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return entry.path();
        }
    }
    return {};
}
uint64_t counter(const std::string& data, const std::string& name) {
    const auto key = "\"" + name + "\":\"";
    const auto at  = data.find(key);
    return at == std::string::npos ? 0 : std::stoull(data.substr(at + key.size()));
}
}  // namespace

int main(int argc, char** argv) {
    check(argc == 2, "supply the fake CUPTI test DSO path");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    check(library != nullptr, "load fake CUPTI");
    auto api           = reinterpret_cast<void (*)(bool)>(dlsym(library, "fakeApi"));
    auto flush_entered = reinterpret_cast<bool (*)()>(dlsym(library, "fakeFlushEntered"));
    auto release       = reinterpret_cast<void (*)()>(dlsym(library, "fakeReleaseFlush"));
    auto version_calls = reinterpret_cast<unsigned (*)()>(dlsym(library, "fakeVersionCalls"));
    auto flood         = reinterpret_cast<void (*)(unsigned, bool)>(dlsym(library, "fakeFlood"));
    auto untracked     = reinterpret_cast<void (*)()>(dlsym(library, "fakeUntrackedApi"));
    auto fail_required = reinterpret_cast<void (*)()>(dlsym(library, "fakeFailRequiredCallback"));
    check(api && flush_entered && release && version_calls && flood && untracked && fail_required, "resolve fake API");

    setenv("RTP_LLM_GPU_STAGE_TRACE", "0", 1);
    trace::start(0);
    check(!trace::active() && !trace::ownsCupti() && trace::newId() == 0 && version_calls() == 0,
          "disabled path must not initialize CUPTI");
    {
        trace::Scope disabled("disabled scope must not build metadata", 1);
        trace::mark("disabled mark", 1);
        check(trace::capture().scope_id == 0 && version_calls() == 0, "disabled markers are inert");
    }

    char temp[] = "/tmp/rtp_cuda_trace_host_test.XXXXXX";
    check(mkdtemp(temp) != nullptr, "make isolated test output directory");
    const std::filesystem::path directory(temp);
    const std::string           prefix = (directory / "trace").string();
    setenv("RTP_LLM_GPU_STAGE_TRACE_FILE", prefix.c_str(), 1);
    setenv("RTP_LLM_GPU_STAGE_TRACE", "1", 1);
    // Before this fork there are no recorder threads or CUDA state. A required
    // callback failure must fail closed, even when optional new APIs may be absent.
    const auto child = fork();
    check(child >= 0, "fork isolated startup-failure test");
    if (child == 0) {
        const auto failure_prefix = (directory / "required-failure" / "trace").string();
        setenv("RTP_LLM_GPU_STAGE_TRACE_FILE", failure_prefix.c_str(), 1);
        fail_required();
        trace::start(7);
        _exit(!trace::active() && !trace::ownsCupti() ? 0 : 1);
    }
    int child_status = 0;
    check(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
          "required callback failure rolls back and cannot report active");
    trace::start(7);
    check(trace::active() && trace::ownsCupti(), "recorder started");
    const trace::Context empty = trace::capture();
    {
        trace::Scope scope("test.mm.d2h", INT64_C(3611341739697827841), 27, 1, 4759552);
        const auto   parent = trace::capture();
        {
            trace::Scope inherited("test.nested");
            const auto   child = trace::capture();
            check(child.request_id == parent.request_id && child.step == 27 && child.scope_id != parent.scope_id,
                  "nested scope inherits request and step");
        }
        check(trace::capture().scope_id == parent.scope_id, "scope restores context");
        api(false);  // Deliberately no EXIT while the flusher is also blocked.
        check(eventually([&] { return flush_entered(); }), "flusher entered test block");
        trace::mark("test.writer_alive_during_blocked_flush", 1234);
        check(eventually([&] {
                  const auto path = find(directory, ".open.json");
                  if (path.empty()) {
                      return false;
                  }
                  const auto data = read(path);
                  return data.find("cuMemcpyDtoHAsync_v2") != std::string::npos
                         && data.find("\"request_id\":\"3611341739697827841\"") != std::string::npos
                         && data.find("\"stream\":\"0x300000\"") != std::string::npos
                         && data.find("\"bytes\":\"4759552\"") != std::string::npos
                         && data.find("test.mm.d2h") != std::string::npos;
              }),
              "unreturned API and scope persist despite blocked CUPTI flush");
        check(eventually([&] {
                  const auto path = find(directory, ".jsonl");
                  return !path.empty()
                         && read(path).find("test.writer_alive_during_blocked_flush") != std::string::npos;
              }),
              "independent writer persisted marker while flusher cannot return");
        check(eventually([&] {
                  const auto path = find(directory, ".incident");
                  return !path.empty()
                         && read(path / "manifest.json").find("\"pin_complete\":true") != std::string::npos;
              }),
              "long demand call pins a sealed pre-tail while CUPTI flush remains blocked");
        api(true);
        check(eventually([&] {
                  const auto path = find(directory, ".open.json");
                  return !path.empty() && read(path).find("\"open_calls\":[]") != std::string::npos;
              }),
              "matching EXIT removes exactly its open API");
    }
    check(trace::capture().scope_id == empty.scope_id, "outer scope restores context");
    flood(100000, true);
    untracked();
    std::vector<std::thread> producers;
    std::atomic<bool>        stop_flood{false};
    std::atomic<unsigned>    running_flood{0};
    for (unsigned i = 0; i != 4; ++i) {
        producers.emplace_back([&] {
            running_flood.fetch_add(1, std::memory_order_release);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (!stop_flood.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
                flood(1000, false);
            }
            running_flood.fetch_sub(1, std::memory_order_release);
        });
    }
    check(eventually([&] {
              const auto snapshot = read(find(directory, ".open.json"));
              return running_flood.load(std::memory_order_acquire) > 0 && counter(snapshot, "bulk_dropped") > 0;
          }),
          "bulk lane is saturated before injecting a dependency record");
    uint64_t dependency_scope = 0;
    {
        trace::Scope scope("test.critical_during_bulk", 42);
        dependency_scope = trace::capture().scope_id;
        api(false);
        api(true);
    }
    check(eventually([&] {
              const auto snapshot = read(find(directory, ".open.json"));
              if (!counter(snapshot, "bulk_dropped") || running_flood.load(std::memory_order_acquire) == 0)
                  return false;
              const auto key = "\"scope_id\":\"" + std::to_string(dependency_scope) + "\"";
              for (const auto& file : std::filesystem::directory_iterator(directory)) {
                  if (file.path().extension() != ".jsonl")
                      continue;
                  std::istringstream lines(read(file.path()));
                  std::string        line;
                  while (std::getline(lines, line)) {
                      if (line.find(key) != std::string::npos && line.find("cuMemcpyDtoHAsync_v2") != std::string::npos
                          && line.find("\"stream\":\"0x300000\"") != std::string::npos
                          && line.find("\"bytes\":\"4759552\"") != std::string::npos)
                          return true;
                  }
              }
              return false;
          }),
          "critical copy parameters persist while launch flood is still dropping bulk history");
    stop_flood.store(true, std::memory_order_release);
    for (auto& producer : producers)
        producer.join();
    // Deliberately overflow the critical history too. Scope/API EXIT must still
    // retire their independent slots; a later real active call must be visible.
    for (unsigned i = 0; i != 30000; ++i) {
        trace::Scope transient("test.transient", 42);
        api(false);
        api(true);
    }
    check(eventually([&] {
              const auto path = find(directory, ".open.json");
              const auto data = path.empty() ? std::string{} : read(path);
              return data.find("\"open_calls\":[]") != std::string::npos
                     && data.find("\"open_scopes\":[]") != std::string::npos
                     && data.find("\"incident_active\":false") != std::string::npos && counter(data, "bulk_dropped") > 0
                     && counter(data, "critical_dropped") > 0 && counter(data, "untracked_api_calls") == 1
                     && counter(data, "optional_callback_unavailable") == 1
                     && data.find("cuGraphCreateWithArguments") != std::string::npos
                     && counter(data, "active_state_overflow") == 0 && counter(data, "close_mismatches") == 0;
          }),
          "lossy history never leaves stale API/scope entries or a permanently armed incident");
    {
        trace::Scope second("test.second_after_overflow", INT64_C(3611341739697827842), 28);
        api(false);
        check(eventually([&] {
                  const auto path = find(directory, ".open.json");
                  const auto data = path.empty() ? std::string{} : read(path);
                  return data.find("test.second_after_overflow") != std::string::npos
                         && data.find("cuMemcpyDtoHAsync_v2") != std::string::npos
                         && counter(data, "incident_count") == 2;
              }),
              "a later long demand call re-arms and pins after queue overflow");
        check(eventually([&] {
                  const auto manifest = read(find(directory, ".incident") / "manifest.json");
                  return manifest.find("3611341739697827842") != std::string::npos;
              }),
              "second incident replaces the completed first candidate");
        api(true);
    }
    check(eventually([&] {
              const auto data = read(find(directory, ".open.json"));
              return data.find("\"open_calls\":[]") != std::string::npos
                     && data.find("\"open_scopes\":[]") != std::string::npos;
          }),
          "second candidate closes while CUPTI flush remains blocked");
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (file.path().extension() == ".jsonl") {
            const auto data = read(file.path());
            check(data.find("cuCtxGetDevice_v2") == std::string::npos
                      && data.find("cuGraphExecGetId") == std::string::npos,
                  "filtered query flood does not enter either history queue");
        }
    }
    release();
    std::fprintf(stdout, "PASS host recorder output=%s\n", temp);
    // Keep the bounded artifacts for validation and diagnosis. Process exit
    // deliberately does not invoke cuptiFinalize or wait for collector threads.
    return 0;
}
