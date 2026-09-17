"""CPU regression for the real engine-to-RPC sample mailbox."""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r"""
#include "rtp_llm/cpp/engine_base/stream/FrontendSpTpotSamples.h"
#include <cassert>
#include <chrono>
#include <thread>
using namespace rtp_llm;
int main() {
    FrontendSpTpotSamples samples;
    auto first = samples.begin();
    std::promise<void> entered;
    auto reader = std::async(std::launch::async, [&] { entered.set_value(); return samples.take(); });
    entered.get_future().wait();
    assert(reader.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    first->complete(12000.0);
    auto result = reader.get();
    assert(result.size()==1 && result[0].first==1 && result[0].second==12000.0);
    assert(samples.take().empty());
    auto cancelled = samples.begin();
    cancelled.reset(); // executor exception / no report must release the RPC reader
    auto second = samples.begin();
    auto third = samples.begin();
    second->complete(7500.0); third->complete(2000.0);
    result = samples.take(); // coalesced frames preserve both samples
    assert(result.size()==2 && result[0].first==3 && result[1].first==4);
    assert(result[0].second==7500.0 && result[1].second==2000.0);
    assert(samples.take().empty());
    auto pending = samples.begin();
    pending.reset();
    assert(samples.take().empty());
}
"""


class FrontendSpTpotSamplesTest(unittest.TestCase):
    def test_pending_cancelled_coalesced_and_once_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, exe = Path(tmp) / "test.cc", Path(tmp) / "test"
            src.write_text(SOURCE)
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    "-I",
                    str(ROOT),
                    str(src),
                    "-o",
                    str(exe),
                ],
                check=True,
            )
            subprocess.run([str(exe)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
