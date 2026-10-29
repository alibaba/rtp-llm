// Stub entry point for the RDMA perf binaries on builds without //:using_rdma:
// the accl-barex backend lives in the internal deps overlay only.
#include <cstdio>

int main() {
    std::fprintf(stderr,
                 "error: rdma perf tools need --define=use_rdma=true and the "
                 "internal deps overlay; see cache_store/rdma/BUILD\n");
    return 1;
}
