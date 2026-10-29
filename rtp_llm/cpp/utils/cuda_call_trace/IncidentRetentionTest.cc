#include "rtp_llm/cpp/utils/cuda_call_trace/IncidentRetention.h"

#include <cstdlib>
#include <iterator>
#include <map>
#include <unistd.h>

namespace detail = rtp_llm::cuda_trace::detail;
namespace {
void check(bool condition) {
    if (!condition) {
        std::abort();
    }
}
void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path);
    out << text;
    out.close();
    check(static_cast<bool>(out));
}
std::string read(const std::filesystem::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
}  // namespace

int main() {
    std::map<uint64_t, detail::ScopeEvidence> scopes{{1, {"engine.execute", 0, 0}},
                                                     {2, {"tp_sync", 1, 0}},
                                                     {3, {"engine.execute", 0, 4}},
                                                     {4, {"kernel.launch", 3, 0}}};
    auto                                      lookup = [&](uint64_t id, detail::ScopeEvidence& evidence) {
        auto found = scopes.find(id);
        if (found == scopes.end()) {
            return false;
        }
        evidence = found->second;
        return true;
    };
    check(!detail::longDemandCall(100, 5000000099ULL, 123, 0, lookup));
    check(detail::longDemandCall(100, 5000000100ULL, 123, 0, lookup));
    check(!detail::longDemandCall(100, 60000000100ULL, -1, 2, lookup));  // Idle peer.
    check(detail::longDemandCall(100, 60000000100ULL, -1, 4, lookup));   // Non-empty engine ancestor.
    check(!detail::longDemandCall(100, 99, 123, 4, lookup));
    check(!detail::longDemandCall(100, 60000000100ULL, -1, 999, lookup));  // Missing evidence.

    char temp[] = "/tmp/rtp_cuda_trace_retention_test.XXXXXX";
    check(mkdtemp(temp) != nullptr);
    const std::filesystem::path root(temp), first = root / "trace.0.jsonl", second = root / "trace.1.jsonl";
    const auto                  incident = root / "trace.incident";
    write(first, "pre-hang API/event record\n");
    write(second, "stalled API BEGIN\n");
    detail::IncidentDirectory owned{};
    detail::pinClosedFiles(incident, {first, second}, "{\"type\":\"long_call_candidate\"}", owned);
    check(owned.valid);
    check(std::filesystem::hard_link_count(first) == 2);
    // This is the recorder's rotation contract: unlink, then create a new inode.
    std::filesystem::remove(first);
    write(first, "later unrelated work\n");
    check(read(incident / first.filename()) == "pre-hang API/event record\n");
    check(read(first) == "later unrelated work\n");
    // A failed replacement must not remove the only complete previous pin.
    bool       failed            = false;
    const auto previous_manifest = read(incident / "manifest.json");
    try {
        detail::pinClosedFiles(incident, {first, root / "missing.jsonl"}, "{}", owned);
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed);
    check(read(incident / first.filename()) == "pre-hang API/event record\n");
    check(read(incident / "manifest.json") == previous_manifest);
    check(!std::filesystem::exists(incident.string() + ".pending"));
    check(!std::filesystem::exists(incident.string() + ".previous"));
    detail::pinClosedFiles(incident, {first}, "{\"type\":\"next_candidate_after_exit\"}", owned);
    check(read(incident / first.filename()) == "later unrelated work\n");
    check(!std::filesystem::exists(incident / second.filename()));
    // Refuse to remove a directory this recorder instance did not create.
    detail::IncidentDirectory unowned{};
    bool                      refused = false;
    try {
        detail::pinClosedFiles(incident, {first}, "{}", unowned);
    } catch (const std::exception&) {
        refused = true;
    }
    check(refused && read(incident / first.filename()) == "later unrelated work\n");
    // A remembered boolean is not ownership: an externally replaced directory
    // must remain untouched even when it appears at this instance's old path.
    std::filesystem::rename(incident, root / "moved_incident");
    check(std::filesystem::create_directory(incident));
    write(incident / "foreign", "keep me\n");
    refused = false;
    try {
        detail::pinClosedFiles(incident, {first}, "{}", owned);
    } catch (const std::exception&) {
        refused = true;
    }
    check(refused && read(incident / "foreign") == "keep me\n");
    check(read(root / "moved_incident" / first.filename()) == "later unrelated work\n");
    return 0;
}
