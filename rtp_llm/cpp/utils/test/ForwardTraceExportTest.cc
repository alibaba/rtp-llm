#include "rtp_llm/cpp/utils/ForwardTrace.h"
#include "autil/legacy/json.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>

using autil::legacy::Any;
using autil::legacy::AnyCast;
using autil::legacy::json::JsonArray;
using autil::legacy::json::JsonMap;

namespace {
JsonMap exportFixture(const std::string& path, rtp_llm::ForwardTraceSession& session, bool include_event) {
    std::ofstream output(path);
    output << R"({"traceEvents":[)";
    if (include_event) {
        output << R"json({"name":"RTP::model_forward(id=1)","ph":"X","cat":"cpu_op","pid":1,"tid":1,"ts":0,"dur":3,"args":{}},)json";
    }
    output << R"({"name":"py_model.forward","ph":"X","cat":"cpu_op","pid":1,"tid":1,"ts":1,"dur":1}]})";
    output.close();
    rtp_llm::enrichForwardTrace(path, session);
    std::ifstream input(path);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    Any document;
    autil::legacy::json::ParseJson(text, document);
    return AnyCast<JsonMap>(document);
}
}

int main() {
    const char* tmp = std::getenv("TEST_TMPDIR");
    const std::string path = std::string(tmp ? tmp : "/tmp") + "/forward-trace-export-" + std::to_string(getpid()) + ".json";
    rtp_llm::ForwardTraceSession session(0, 4);
    auto& record = *session.append();
    record.integers = {{"logical_sequences", 2}, {"request_count", 2}};
    record.strings = {{"status", "ok"}, {"kind", "model"}, {"phase", "prefill_target"}};
    record.arrays["input_lengths"].host = {8192, 128};
    record.arrays["sequence_lengths"].host = {};
    record.arrays["prefix_lengths"].host = {0, 32640};
    session.seal();

    auto result = exportFixture(path, session, true);
    auto envelope = AnyCast<JsonMap>(result.at("rtp_forward_metadata"));
    assert(AnyCast<bool>(envelope.at("complete")));
    auto event = AnyCast<JsonMap>(AnyCast<JsonArray>(result.at("traceEvents")).at(0));
    assert(AnyCast<std::string>(event.at("name")) ==
           "RTP::model_forward(id=1,phase=prefill_target,requests=2,sequences=2,tokens=8320)");
    auto args = AnyCast<JsonMap>(event.at("args"));
    auto metadata = AnyCast<JsonMap>(args.at("rtp_forward"));
    assert(autil::legacy::json::ToString(metadata.at("q_lens"), true) == "[8192,128]");
    assert(autil::legacy::json::ToString(metadata.at("kv_lens"), true) == "[8192,32768]");

    result = exportFixture(path, session, false);
    envelope = AnyCast<JsonMap>(result.at("rtp_forward_metadata"));
    assert(!AnyCast<bool>(envelope.at("complete")));
    assert(autil::legacy::json::JsonNumberCast<int64_t>(envelope.at("unannotated_forward_events")) == 1);
    rtp_llm::ForwardTraceSession empty(0, 4);
    empty.seal();
    result = exportFixture(path, empty, false);
    envelope = AnyCast<JsonMap>(result.at("rtp_forward_metadata"));
    assert(!AnyCast<bool>(envelope.at("complete")));
    record.arrays["input_lengths"].valid = false;
    result = exportFixture(path, session, true);
    envelope = AnyCast<JsonMap>(result.at("rtp_forward_metadata"));
    assert(!AnyCast<bool>(envelope.at("complete")));
    std::remove(path.c_str());
}
