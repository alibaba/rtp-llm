"""Exercise the production prefault worker on CPU, including madvise failure."""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <utility>
#include <unistd.h>
#include <vector>
static int64_t currentTimeUs() {
 return std::chrono::duration_cast<std::chrono::microseconds>(
  std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define RTP_LLM_LOG_INFO(...) ((void)0)
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
static char* begin_ptr;
static size_t arena_bytes;
static std::mutex lock;
static std::vector<std::pair<size_t,size_t>> ranges;
static int fake_madvise(void* p,size_t n,int advice) {
 assert(advice==MADV_POPULATE_WRITE);
 auto offset=static_cast<char*>(p)-begin_ptr;
 assert(offset>=0 && size_t(offset)<arena_bytes && n<=arena_bytes-size_t(offset));
 std::lock_guard<std::mutex> guard(lock);
 ranges.emplace_back(offset,n);
 return -1; // Exercise the actual memset fallback, including the partial last chunk.
}
// Replace only the thread constructor to inject resource exhaustion after one
// real worker starts. The production catch/join path remains unchanged.
static int fail_after=-1;
static std::atomic<int> active_workers{0};
class TestThread {
 std::thread worker;
public:
 template<class F> explicit TestThread(F fn) {
  if(fail_after==0) throw std::runtime_error("injected thread creation failure");
  if(fail_after>0) --fail_after;
  worker=std::thread([fn]{++active_workers; fn(); --active_workers;});
 }
 TestThread(TestThread&&)=default;
 bool joinable() const {return worker.joinable();}
 void join() {worker.join();}
};
#define madvise fake_madvise
@@CLASS@@
#undef madvise
int main() {
 cpu_set_t original, after;
 assert(sched_getaffinity(0,sizeof(original),&original)==0);
 int cpu=0; while(cpu<CPU_SETSIZE && !CPU_ISSET(cpu,&original)) ++cpu;
 assert(cpu<CPU_SETSIZE);
 const std::string valid=std::to_string(cpu)+";"+std::to_string(cpu);
 for(const char* groups : {"",valid.c_str()}) {
  setenv("RTP_LLM_HOST_BLOCK_POOL_PREFAULT_CPU_GROUPS",groups,1);
  for(size_t n : {size_t(1),size_t(4097),size_t(2097152),size_t(4194305)}) {
   std::vector<char> memory(n+2,char(0x5a));
   begin_ptr=memory.data()+1; arena_bytes=n; ranges.clear();
   { HostArenaPrefaulter worker(begin_ptr,n,4); worker.join(); worker.join(); }
   assert(memory.front()==char(0x5a) && memory.back()==char(0x5a));
   assert(std::all_of(memory.begin()+1,memory.end()-1,[](char c){return c==0;}));
   std::sort(ranges.begin(),ranges.end());
   size_t cursor=0; for(auto range:ranges) { assert(range.first==cursor); cursor+=range.second; }
   assert(cursor==n);
   assert(sched_getaffinity(0,sizeof(after),&after)==0 && CPU_EQUAL(&original,&after));
  }
 }
 for(const char* groups : {";","0;","0,","-1","1-0","1024","0;;1","x","0;0;0;0;0"}) {
  setenv("RTP_LLM_HOST_BLOCK_POOL_PREFAULT_CPU_GROUPS",groups,1);
  bool caught=false;
  try {HostArenaPrefaulter worker(nullptr,4096,4);} catch(const std::invalid_argument&) {caught=true;}
  assert(caught);
 }
 unsetenv("RTP_LLM_HOST_BLOCK_POOL_PREFAULT_CPU_GROUPS");
 {
  std::vector<char> memory(4194305,char(0x5a));
  begin_ptr=memory.data(); arena_bytes=memory.size(); ranges.clear();
  fail_after=1;
  bool caught=false;
  try {HostArenaPrefaulter worker(begin_ptr,arena_bytes,4);}
  catch(const std::runtime_error&) {caught=true;}
  assert(caught && active_workers==0);
  assert(std::all_of(memory.begin(),memory.end(),[](char c){return c==0;}));
  fail_after=-1;
 }
 {HostArenaPrefaulter zero(nullptr,0,4); zero.join();}
}
"""


class HostArenaPrefaultTests(unittest.TestCase):
    def test_bounds_fallback_affinity_and_invalid_configuration(self):
        source = (ROOT / "rtp_llm/cpp/cache/BlockPool.cc").read_text()
        start = source.index("class HostArenaPrefaulter {")
        end = source.index("\n};", start) + len("\n};")
        with tempfile.TemporaryDirectory() as tmp:
            src, exe = Path(tmp) / "test.cc", Path(tmp) / "test"
            src.write_text(
                HARNESS.replace(
                    "@@CLASS@@", source[start:end].replace("std::thread", "TestThread")
                )
            )
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pthread",
                    str(src),
                    "-o",
                    str(exe),
                ],
                check=True,
            )
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
