"""CPU regression for the production deferred-capture control flow.

Compile the actual lifecycle methods with GPU dependencies stubbed. Only the
allocation/capture body of initCapture is replaced; its deferral guard remains
production code. This deliberately does not claim CUDA/HIP execution coverage.
Run with a C++17 compiler available through CXX (or c++).
"""

import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def function(source, signature):
    start = source.index(signature)
    return source[start : source.index("\n}", start) + 2]


def harness():
    decode = (ROOT / "cuda_graph_decode.cc").read_text()
    runner = (ROOT / "cuda_graph_runner.cc").read_text()
    header = (ROOT / "cuda_graph_runner.h").read_text()
    init = function(runner, "void CudaGraphRunner::initCapture()")
    # Keep the real startup-defer branch and flag writes, replacing only GPU
    # capture work with a controllable success/exception/empty-result boundary.
    init = init[: init.index("    if (enable_cuda_graph_) {")]
    init += """
    if (capture_mode == 1) {
        graph_instances_[1] = 1;
        throw std::runtime_error("capture failed");
    }
    if (capture_mode == 2) return;
    graph_instances_[1] = 1;
    ++capture_runs;
}
"""
    predicates = []
    for name in ("hasDeferredDecodeInit", "hasCapturedDecodeGraph"):
        start = header.index("    bool " + name + "() const {")
        predicates.append(header[start : header.index("\n    }", start) + 6])
    stub = r"""
#include <map>
#include <stdexcept>
#include <cstdio>
#define RTP_LLM_LOG_WARNING(...) ((void)0)
#define RTP_LLM_LOG_INFO(...) ((void)0)
#define RTP_LLM_CHECK_WITH_INFO(condition, ...) \
    do { if (!(condition)) throw std::runtime_error("capture missing"); } while (0)
namespace c10 { struct InferenceMode { explicit InferenceMode(bool) {} }; }
namespace cuda_graph { void graphEmptyCache() {} void graphDeviceSynchronize() {} }
class CudaGraphRunner {
public:
    bool enable_cuda_graph_ = true, is_prefill_cuda_graph_mode_ = false;
    bool decode_init_deferred_ = false, decode_capture_deferred_ = false;
    bool decode_capture_done_ = false;
    std::map<int, int> graph_instances_;
    int shared_graph_pool_ = 0, capture_runs = 0, capture_mode = 0;
    void initCapture();
    void finishDeferredDecodeInit();
    void releaseCapturedDecodeGraph();
"""
    source = stub + "\n".join(predicates) + "\n};\n"
    source += function(decode, "void CudaGraphRunner::finishDeferredDecodeInit()")
    source += function(decode, "void CudaGraphRunner::releaseCapturedDecodeGraph()")
    source += init
    source += r"""
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        CudaGraphRunner runner;
        runner.initCapture();
        require(runner.hasDeferredDecodeInit(), "startup must defer");
        runner.finishDeferredDecodeInit();
        require(runner.hasCapturedDecodeGraph() && runner.capture_runs == 1,
                "first deferred capture missing");
        runner.finishDeferredDecodeInit();
        require(runner.capture_runs == 1, "completed capture repeated");
        for (int cycle = 0; cycle < 2; ++cycle) {
            runner.releaseCapturedDecodeGraph();
            require(!runner.hasCapturedDecodeGraph() && runner.hasDeferredDecodeInit(),
                    "release must arm recapture");
            runner.finishDeferredDecodeInit();
            require(runner.hasCapturedDecodeGraph() && runner.capture_runs == cycle + 2,
                    "recapture must not re-enter startup deferral");
        }
        for (int failure : {1, 2}) {
            CudaGraphRunner failed_runner;
            failed_runner.initCapture();
            failed_runner.capture_mode = failure;
            bool threw = false;
            try { failed_runner.finishDeferredDecodeInit(); }
            catch (const std::runtime_error&) { threw = true; }
            require(threw, "failed or empty capture must propagate");
            require(!failed_runner.decode_capture_done_
                        && !failed_runner.hasCapturedDecodeGraph()
                        && failed_runner.hasDeferredDecodeInit(),
                    "capture failure must not publish completion");
            require(failed_runner.graph_instances_.size() == (failure == 1 ? 1 : 0),
                    "failure must retain partial resources for dirty-session owner");
        }
        std::puts("PASS: first capture, two recaptures, failure/empty-capture completion gates and retained resources");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
"""
    return source


class DeferredCaptureStateTest(unittest.TestCase):
    def test_production_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source, program = root / "lifecycle.cc", root / "lifecycle"
            source.write_text(harness())
            compiler = shlex.split(os.environ.get("CXX", "c++"))
            subprocess.run(
                compiler + ["-std=c++17", str(source), "-o", str(program)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run([str(program)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
