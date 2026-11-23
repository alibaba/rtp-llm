#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <chrono>
#include <thread>
#include <rapidjson/document.h>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/MetadataReplay.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/EvaluationStats.h"
namespace fs = std::filesystem;
using namespace rtp_llm::benchmark;
namespace {
template<class T>
T read(std::istream& f) {
    T value{};
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    stateRequire(bool(f), "truncated trace");
    return value;
}
std::string text(std::istream& f) {
    const auto n = read<uint32_t>(f);
    stateRequire(n < 1048576, "invalid trace label");
    std::string s(n, '\0');
    f.read(s.data(), n);
    stateRequire(bool(f), "short trace label");
    return s;
}
std::string contents(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    stateRequire(bool(f), "cannot open input file");
    return {std::istreambuf_iterator<char>(f), {}};
}
void writeText(const fs::path& p, const std::string& value) {
    std::ofstream f(p.string() + ".tmp");
    f << value << '\n';
    f.close();
    stateRequire(bool(f), "cannot write output");
    fs::rename(p.string() + ".tmp", p);
}
MetadataReplayConfig config(const std::string& value, uint32_t base) {
    rapidjson::Document d;
    d.Parse(value.c_str());
    stateRequire(!d.HasParseError() && d.IsObject(), "invalid config");
    MetadataReplayConfig c;
    c.dsv41_topology    = true;
    c.block_tokens      = base;
    c.cp_size           = 4;
    c.cp_rank           = 0;
    c.min_replay_tokens = 128;
    c.swa_block_tokens  = c.block_tokens * c.cp_size;
    c.sliding_window    = 128;
    c.linear_step       = 1;
    auto count          = [&](const char* tier, const char* group) {
        const auto& v = d["tiers"][tier][group];
        stateRequire(v.IsUint() && v.GetUint() > 0, "invalid capacity");
        return v.GetUint();
    };
    c.device_blocks   = count("device", "full");
    c.swa_blocks      = count("device", "swa");
    c.host_blocks     = count("host", "full");
    c.host_swa_blocks = count("host", "swa");
    c.disk_blocks     = count("disk", "full");
    c.disk_swa_blocks = count("disk", "swa");
    c.disk_fifo       = d.HasMember("disk_fifo") && d["disk_fifo"].GetBool();
    if (d.HasMember("last_n"))
        c.last_n = d["last_n"].GetUint();
    c.device_low  = .82;
    c.device_high = .90;
    c.host_low    = .90;
    c.host_high   = .94;
    c.disk_low    = .92;
    c.disk_high   = .97;
    return c;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--btcpfx-reader-version") {
            std::cout << "2" << std::endl;
            return 0;
        }
        stateRequire(
            argc >= 4,
            "usage: evaluation_replay CONFIG INPUT_DIR NEW_OUTPUT_DIR --requests N --identity FILE [--pause-after N]");
        std::map<std::string, std::string> options;
        for (int i = 4; i < argc; i += 2) {
            stateRequire(i + 1 < argc, "missing option value");
            const std::string key = argv[i];
            stateRequire(key == "--requests" || key == "--identity" || key == "--warmup" || key == "--pause-after",
                         "unsupported option (checkpoint resume is not supported)");
            stateRequire(options.emplace(key, argv[i + 1]).second, "duplicate option");
        }
        stateRequire(options.count("--requests") && options.count("--identity"), "requests and identity are required");
        const uint64_t limit  = std::stoull(options.at("--requests"));
        const uint64_t warmup = options.count("--warmup") ? std::stoull(options.at("--warmup")) : 0;
        stateRequire(warmup < limit, "warmup must leave a measured range");
        stateRequire(limit > 0, "empty requested range");
        const uint64_t pause = options.count("--pause-after") ? std::stoull(options.at("--pause-after")) : 0;
        stateRequire(!options.count("--pause-after") || (pause > warmup && pause < limit), "invalid pause range");
        const fs::path input = argv[2], out = argv[3];
        stateRequire(fs::create_directory(out), "output must be new directory");
        const auto    parameters = contents(argv[1]);
        const auto    identity   = contents(options.at("--identity"));
        std::ifstream first(input / "000000.btcbin", std::ios::binary);
        char          first_magic[8];
        first.read(first_magic, sizeof(first_magic));
        stateRequire(first && std::memcmp(first_magic, "BTCPFX01", 8) == 0, "invalid trace header");
        const auto base = read<uint32_t>(first);
        stateRequire(base > 0 && base <= INT32_MAX / 4, "invalid CP4 block granularity");
        writeText(out / "parameters.json", parameters);
        writeText(out / "identity.json", identity);
        MetadataReplay  engine(config(parameters, base));
        EvaluationStats stats;
        uint64_t        ordinal = 0, last_ts = 0;
        std::string                   domain;
        std::ofstream                 facts(out / "match-lengths.u64", std::ios::binary);
        std::ofstream                 arrival_facts(out / "arrival-hits.u64", std::ios::binary);
        uint64_t                      index = 0;
        while (ordinal < limit) {
            std::ostringstream name;
            name << std::setw(6) << std::setfill('0') << index << ".btcbin";
            std::ifstream f(input / name.str(), std::ios::binary);
            stateRequire(bool(f), "missing trace chunk");
            char magic[8];
            f.read(magic, 8);
            stateRequire(std::memcmp(magic, "BTCPFX01", 8) == 0, "invalid trace magic");
            stateRequire(read<uint32_t>(f) == base, "base granularity mismatch");
            const auto count           = read<uint64_t>(f);
            const auto expected_tokens = read<uint64_t>(f);
            stateRequire(count > 0, "invalid chunk count");
            const auto current_domain = text(f);
            if (domain.empty())
                domain = current_domain;
            stateRequire(domain == current_domain, "cache namespace changed");
            uint64_t rows_read    = 0;
            uint64_t chunk_tokens = 0;
            for (uint64_t i = 0; i < count && ordinal < limit; ++i) {
                MetadataRequest q;
                q.id           = ordinal + 1;
                q.timestamp_ns = read<uint64_t>(f);
                q.input_tokens = read<uint32_t>(f);
                ++rows_read;
                const auto id_length = read<uint32_t>(f);
                stateRequire(id_length <= 1048576, "invalid request id length");
                f.ignore(id_length);
                stateRequire(q.input_tokens > 0, "invalid trace length");
                const auto keys = static_cast<size_t>(q.input_tokens) / base;
                chunk_tokens += q.input_tokens;
                q.prefix_keys.resize(keys);
                f.read(reinterpret_cast<char*>(q.prefix_keys.data()), keys * sizeof(int64_t));
                stateRequire(bool(f), "truncated trace keys");
                stateRequire(q.timestamp_ns >= last_ts, "trace order changed");
                q.observe         = ordinal >= warmup;
                const auto result = engine.run(q);
                stateRequire(result.success, "native request failed");
                if (!q.observe) {
                    ++ordinal;
                    last_ts = q.timestamp_ns;
                    continue;
                }
                stateRequire(result.joint_match_tokens == uint64_t(result.reuse_tokens),
                             "native joint match and ready reuse differ");
                const uint64_t lengths[] = {result.full_match_tokens, result.joint_match_tokens};
                stateRequire(result.host_hit_tokens + result.disk_hit_tokens <= lengths[1], "tier hits exceed joint");
                const uint64_t arrival[] = {q.timestamp_ns,
                                            static_cast<uint64_t>(q.input_tokens),
                                            lengths[0],
                                            lengths[1],
                                            result.host_hit_tokens,
                                            result.disk_hit_tokens};
                arrival_facts.write(reinterpret_cast<const char*>(arrival), sizeof(arrival));
                stateRequire(bool(arrival_facts), "arrival facts write failed");
                facts.write(reinterpret_cast<const char*>(lengths), sizeof(lengths));
                stateRequire(bool(facts), "facts write failed");
                stats.add(q.timestamp_ns, q.input_tokens, lengths[0], lengths[1], result.evictions);
                ++ordinal;
                last_ts = q.timestamp_ns;
                if (ordinal == pause) {
                    facts.flush();
                    arrival_facts.flush();
                    stateRequire(bool(facts) && bool(arrival_facts), "pause facts flush failed");
                    writeText(out / "summary.json", stats.json());
                    writeText(out / "PAUSED", std::to_string(ordinal));
                    while (!fs::exists(out / "CONTINUE"))
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    fs::remove(out / "CONTINUE");
                    fs::remove(out / "PAUSED");
                }
                if (ordinal % 10000 == 0)
                    writeText(out / "progress.json", stats.json());
            }
            if (rows_read == count)
                stateRequire(chunk_tokens == expected_tokens && f.peek() == EOF, "chunk totals mismatch");
            ++index;
        }
        arrival_facts.close();
        stateRequire(bool(arrival_facts), "arrival facts flush failed");
        facts.flush();
        stateRequire(bool(facts), "facts flush failed");
        engine.writeTreeStructure(out.string());
        writeText(out / "summary.json", stats.json());
        std::cout << "completed " << ordinal << " requests, joint hit rate "
                  << double(stats.total.joint.sum) / stats.total.input_tokens << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
