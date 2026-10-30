"""Compile the actual host constructors with injected CUDA/PCCL failures."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def block(text, marker):
    start = text.index(marker)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def main():
    root = Path(__file__).parent / 'csrc/transport'
    sparse = (root / 'ppu_sparse_gin.cu').read_text()
    signal = (root / 'ppu_sparse_signal_gin.cu').read_text()
    source = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
static int operation, fail_at, windows, devices, allocations, objects;
static size_t allocation_bytes;
void* operator new(size_t n) { ++objects; return std::malloc(n); }
void operator delete(void* p) noexcept { if (p) { --objects; std::free(p); } }
void operator delete(void* p, size_t) noexcept { operator delete(p); }
using pcclComm_t = void*;
using pcclWindow_t = int;
struct pcclDevComm {};
struct pcclDevCommRequirements {
    bool ginForceEnable;
    int ginConnectionType, ginContextCount, ginSignalCount, ginCounterCount;
};
#define PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {}
#define PCCL_WIN_COLL_SYMMETRIC 1
#define PCCL_GIN_CONNECTION_FULL 1
bool failing() { return ++operation == fail_at; }
int pcclCommWindowRegister(pcclComm_t, void*, size_t, pcclWindow_t* w, int) {
    if (failing()) return 1;
    *w = ++windows; return 0;
}
int pcclCommWindowDeregister(pcclComm_t, pcclWindow_t) { --windows; return 0; }
int pcclDevCommCreate(pcclComm_t, pcclDevCommRequirements*, pcclDevComm*) {
    if (failing()) return 1;
    ++devices; return 0;
}
int pcclDevCommDestroy(pcclComm_t, pcclDevComm*) { --devices; return 0; }
int cudaMalloc(void** p, size_t n) {
    if (failing()) return 1;
    *p = std::malloc(n); ++allocations; allocation_bytes = n; return 0;
}
int cudaMemset(void*, int, size_t n) { assert(n == allocation_bytes); return failing() ? 1 : 0; }
int cudaFree(void* p) { if (p) { std::free(p); --allocations; } return 0; }
'''
    source += block(sparse, 'struct Context {') + ';\n'
    source += block(sparse, 'static int create_context(') + '\n'
    source += block(sparse, 'extern "C" int sparse_gin_create(') + '\n'
    source += block(signal, 'extern "C" int signal_gin_create(') + '\n'
    source += r'''
int main() {
    auto creators = {sparse_gin_create, signal_gin_create};
    int kind = 0;
    for (auto create : creators) {
        for (int failure = 0; failure <= 5; ++failure) {
            operation = 0; fail_at = failure;
            const int before = objects;
            void* result = reinterpret_cast<void*>(1);
            int rc = create(nullptr, nullptr, nullptr, 64, 16, 0, 4, &result);
            if (failure) {
                assert(rc != 0 && result == nullptr);
            } else {
                assert(rc == 0 && result != nullptr);
                assert(allocation_bytes == (kind ? 4 : 1) * sizeof(uint64_t));
                auto* c = static_cast<Context*>(result);
                cudaFree(c->sequence);
                pcclDevCommDestroy(c->comm, &c->dev);
                pcclCommWindowDeregister(c->comm, c->rxwin);
                pcclCommWindowDeregister(c->comm, c->txwin);
                delete c;
            }
            assert(windows == 0 && devices == 0 && allocations == 0 && objects == before);
        }
        ++kind;
    }
}
'''
    source = '#include <initializer_list>\n' + source
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / 'test.cpp').write_text(source)
        subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++17', str(path / 'test.cpp'), '-o', str(path / 'test')], check=True)
        subprocess.run([str(path / 'test')], check=True)
    print('PASS: sparse/signal success and ten injected initialization failures')


if __name__ == '__main__':
    main()
