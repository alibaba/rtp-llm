"""Compile the real Logger on a CPU host; only alog/autil are test doubles.

This checks prefix state, not alog file descriptors or actual CRIU restore.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
STUBS = {
    "alog/Logger.h": r"""
#pragma once
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
namespace alog {
enum { LOG_LEVEL_ERROR=1, LOG_LEVEL_WARN=2, LOG_LEVEL_INFO=3,
       LOG_LEVEL_DEBUG=4, LOG_LEVEL_TRACE1=5 };
class Logger {
public:
    static Logger* getLogger(const char*) { static Logger l; return &l; }
    void setLevel(uint32_t l) { level=l; }
    uint32_t getLevel() { return level; }
    bool isLevelEnabled(int32_t) { return true; }
    void flush() {}
    void log(uint32_t, const char*, const char* message) {
        std::lock_guard<std::mutex> guard(lock);
        std::cout << message << std::endl;
    }
private:
    uint32_t level=LOG_LEVEL_INFO;
    std::mutex lock;
};
class Configurator { public: static void configureLogger(const char*) {} };
}
#define AUTIL_ROOT_LOG_CONFIG() ((void)0)
#define AUTIL_ROOT_LOG_SETLEVEL(level) ((void)0)
""",
    "alog/Appender.h": "#pragma once\nnamespace alog { class FileAppender {}; class ConsoleAppender {}; }\n",
    "autil/EnvUtil.h": r"""
#pragma once
#include <cstdlib>
namespace autil { struct EnvUtil {
static int getEnv(const char* key, int fallback) {
    const char* value=std::getenv(key); return value ? std::atoi(value) : fallback;
}
}; }
""",
    "autil/TimeUtility.h": "#pragma once\n",
    "autil/NetUtil.h": r"""
#pragma once
#include <string>
namespace autil { struct NetUtil {
inline static std::string address="192.0.2.10";
static bool GetDefaultIp(std::string& out) { out=address; return true; }
}; }
""",
}


@unittest.skipUnless(shutil.which("c++"), "C++ compiler required")
class ScrNativeLoggerTest(unittest.TestCase):
    def test_existing_and_late_loggers_use_each_restored_ip(self):
        source = r"""
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/NetUtil.h"
#include <thread>
#include <vector>
int main() {
    using rtp_llm::Logger;
    auto& engine = Logger::getEngineLogger();
    auto& query = Logger::getQueryAccessLogger();
    engine.log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "seed");
    for (const auto* ip : {"192.0.2.20", "192.0.2.30"}) {
        autil::NetUtil::address=ip;
        // The restored process retains existing Logger instances.
        Logger::refreshRuntimeIdentity(ip);
        // A logger instantiated late must use the process override, even if
        // the constructor's network dependency still supplies a stale value.
        autil::NetUtil::address="192.0.2.10";
        engine.log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "restored");
        query.log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "restored");
        Logger::getStackTraceLogger().log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "restored");
        Logger::getAccessLogger().log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "restored");
    }
    for (const auto& bad : {std::string(""), std::string("invalid"), std::string("192.0.2.1") + '\0' + "suffix"}) {
        try { Logger::refreshRuntimeIdentity(bad); return 2; }
        catch (const std::invalid_argument&) {}
    }
    engine.setRank(7);
    engine.setBaseLevel(alog::LOG_LEVEL_TRACE1);
    engine.log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "trace-check");
    std::vector<std::thread> readers;
    for (int i=0; i<4; ++i) {
        readers.emplace_back([&engine] {
            for (int j=0; j<100; ++j)
                engine.log(alog::LOG_LEVEL_INFO, "file", 1, "fn", "concurrent");
        });
    }
    for (int i=0; i<100; ++i)
        Logger::refreshRuntimeIdentity(i % 2 ? "192.0.2.20" : "192.0.2.30");
    for (auto& reader : readers) reader.join();
}
"""
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name, body in STUBS.items():
                target = temp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body)
            (temp / "main.cc").write_text(source)
            binary = temp / "logger-test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    str(temp),
                    "-I",
                    str(ROOT),
                    str(ROOT / "rtp_llm/cpp/utils/Logger.cc"),
                    str(temp / "main.cc"),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            env = {
                key: value for key, value in os.environ.items() if key != "LOG_LEVEL"
            }
            output = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True, env=env
            ).stdout.splitlines()
        self.assertIn("[192.0.2.10]", output[0])
        for line in output[1:5]:
            self.assertIn("[192.0.2.20]", line)
        for line in output[5:9]:
            self.assertIn("[192.0.2.30]", line)
        self.assertIn("[RANK 7][192.0.2.30] trace-check", output)
        concurrent = [line for line in output if line.endswith("concurrent")]
        self.assertEqual(len(concurrent), 400)
        self.assertTrue(
            all(
                line
                in {
                    "[RANK 7][192.0.2.20] concurrent",
                    "[RANK 7][192.0.2.30] concurrent",
                }
                for line in concurrent
            )
        )


if __name__ == "__main__":
    unittest.main()
