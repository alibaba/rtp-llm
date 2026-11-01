// SPDX-License-Identifier: Apache-2.0
// CUDA 13+ standalone HOST_NUMA/FABRIC environment probe. No RTP dependency.
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if CUDA_VERSION < 13000
#error "Build this probe with CUDA Toolkit 13 or newer"
#endif

namespace {
volatile sig_atomic_t stopped = 0;
bool cleanup_failed = false;
void stop_handler(int) { stopped = 1; }
struct ReportedError : std::runtime_error { ReportedError(): std::runtime_error("already reported") {} };

std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else o << c;
    }
    o << '"'; return o.str();
}
void event(const char* stage, const std::string& fields = "") {
    std::cout << "{\"status\":\"PASS\",\"stage\":" << quote(stage) << fields << "}" << std::endl;
}
std::string driver_details(CUresult r) {
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(r, &name); cuGetErrorString(r, &desc);
    return ",\"cuda_code\":" + std::to_string(int(r)) + ",\"cuda_name\":" + quote(name ? name : "UNKNOWN") +
           ",\"message\":" + quote(desc ? desc : "unknown CUDA error");
}
void check(CUresult r, const char* stage) {
    if (r == CUDA_SUCCESS) return;
    const char* status = r == CUDA_ERROR_NOT_SUPPORTED ? "UNSUPPORTED" :
        (r == CUDA_ERROR_NOT_PERMITTED ? "PERMISSION_DENIED" : "FAIL");
    std::cout << "{\"status\":" << quote(status) << ",\"stage\":" << quote(stage) << driver_details(r) << "}" << std::endl;
    throw ReportedError();
}
void rtcheck(cudaError_t r, const char* stage) {
    if (r == cudaSuccess) return;
    std::cout << "{\"status\":\"FAIL\",\"stage\":" << quote(stage)
              << ",\"cuda_code\":" << int(r) << ",\"cuda_name\":" << quote(cudaGetErrorName(r))
              << ",\"message\":" << quote(cudaGetErrorString(r)) << "}" << std::endl;
    throw ReportedError();
}
void fail(const char* stage, const std::string& message) {
    std::cout << "{\"status\":\"FAIL\",\"stage\":" << quote(stage) << ",\"message\":" << quote(message) << "}" << std::endl;
    throw ReportedError();
}
void oscheck(bool ok, const char* stage) {
    if (!ok) fail(stage, std::string(std::strerror(errno)) + " (errno=" + std::to_string(errno) + ")");
}
void clean(CUresult r, const char* stage) {
    if (r != CUDA_SUCCESS) {
        cleanup_failed = true;
        std::cout << "{\"status\":\"CLEANUP_FAILED\",\"stage\":" << quote(stage) << driver_details(r) << "}" << std::endl;
    }
}
struct Fd {
    int value = -1;
    explicit Fd(int v): value(v) {}
    ~Fd() { if (value >= 0) close(value); }
    Fd(const Fd&) = delete;
};
struct Context {
    CUdevice device{}; bool retained = false, defer_cleanup = false;
    explicit Context(int ordinal) {
        check(cuDeviceGet(&device, ordinal), "device_get");
        CUcontext context{};
        check(cuDevicePrimaryCtxRetain(&context, device), "context_retain"); retained = true;
        CUresult r = cuCtxSetCurrent(context);
        if (r != CUDA_SUCCESS) { clean(cuDevicePrimaryCtxRelease(device), "context_release"); retained = false; check(r, "context_set"); }
    }
    ~Context() { if (retained && !defer_cleanup) clean(cuDevicePrimaryCtxRelease(device), "context_release"); }
};
struct Mapping {
    CUmemGenericAllocationHandle handle{}; CUdeviceptr address{}; size_t bytes = 0;
    bool allocated = false, reserved = false, mapped = false, defer_cleanup = false;
    ~Mapping() {
        if (defer_cleanup) return;
        if (mapped) clean(cuMemUnmap(address, bytes), "unmap");
        if (allocated) clean(cuMemRelease(handle), "handle_release");
        if (reserved) clean(cuMemAddressFree(address, bytes), "address_free");
    }
    void map() {
        check(cuMemAddressReserve(&address, bytes, 0, 0, 0), "address_reserve"); reserved = true;
        check(cuMemMap(address, bytes, 0, handle, 0), "map"); mapped = true;
    }
};
struct ResultBuffer {
    unsigned long long* ptr = nullptr;
    bool defer_cleanup = false;
    ~ResultBuffer() {
        if (ptr && !defer_cleanup) {
            cudaError_t r = cudaFree(ptr);
            if (r != cudaSuccess) { cleanup_failed = true;
                std::cout << "{\"status\":\"CLEANUP_FAILED\",\"stage\":\"result_free\",\"cuda_code\":" << int(r) << ",\"cuda_name\":" << quote(cudaGetErrorName(r)) << "}" << std::endl;
            }
        }
    }
};
// Declared after ResultBuffer so this runs before any result/mapping/context teardown.
// A failed synchronization does not establish quiescence. This standalone process
// exits immediately after failure, allowing driver process teardown to reclaim it.
struct DrainGuard {
    Context& context; Mapping& mapping; ResultBuffer& result; bool pending = false;
    ~DrainGuard() {
        if (!pending) return;
        CUresult r = cuCtxSynchronize();
        if (r == CUDA_SUCCESS) { event("failure_path_drain"); return; }
        cleanup_failed = true;
        context.defer_cleanup = mapping.defer_cleanup = result.defer_cleanup = true;
        std::cout << "{\"status\":\"CLEANUP_DEFERRED\",\"stage\":\"failure_path_drain\",\"action\":\"process_exit_without_explicit_unmap_or_free\""
                  << driver_details(r) << "}" << std::endl;
    }
};
struct PublishedFile {
    std::string path; dev_t dev{}; ino_t ino{}; bool published = false;
    ~PublishedFile() {
        if (!published) return;
        struct stat s{};
        if (lstat(path.c_str(), &s) == 0 && s.st_dev == dev && s.st_ino == ino && unlink(path.c_str()) != 0) {
            cleanup_failed = true;
            std::cout << "{\"status\":\"CLEANUP_FAILED\",\"stage\":\"unlink_handle_file\"}" << std::endl;
        }
    }
};
uint64_t number(const std::string& s, const char* name) {
    if (s.empty() || s[0] == '-') fail("arguments", std::string("invalid ") + name);
    size_t end = 0; uint64_t n = 0;
    try { n = std::stoull(s, &end); } catch (...) { fail("arguments", std::string("invalid ") + name); }
    if (end != s.size()) fail("arguments", std::string("invalid ") + name);
    return n;
}
struct Options {
    std::string mode, file; int device = 0, numa = -1;
    uint64_t bytes = 4 * 1024 * 1024, seconds = 0, rounds = 3;
};
Options parse(int argc, char** argv) {
    if (argc < 2) fail("arguments", "usage: egm_fabric_probe info|owner|consumer [--device N] [--numa N] [--file PATH] [--bytes N] [--seconds N] [--rounds N]");
    Options o; o.mode = argv[1];
    if (o.mode != "info" && o.mode != "owner" && o.mode != "consumer") fail("arguments", "unknown mode");
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) fail("arguments", "option is missing value");
        std::string k = argv[i], v = argv[i+1];
        if (k == "--file") o.file = v;
        else if (k == "--bytes") o.bytes = number(v, "bytes");
        else if (k == "--seconds") o.seconds = number(v, "seconds");
        else if (k == "--rounds") o.rounds = number(v, "rounds");
        else if (k == "--device" || k == "--numa") {
            uint64_t n = number(v, k.c_str()); if (n > INT_MAX) fail("arguments", "ordinal out of range");
            if (k == "--device") o.device = int(n); else o.numa = int(n);
        } else fail("arguments", "unknown option: " + k);
    }
    if (o.mode != "info" && o.file.empty()) fail("arguments", "--file is required");
    if (o.bytes == 0 || o.bytes % 4 || o.bytes > (1ULL << 30)) fail("arguments", "--bytes must be 4-byte aligned and between 4 and 1GiB");
    if (o.rounds < 1 || o.rounds > 1000000) fail("arguments", "--rounds must be 1..1000000");
    return o;
}
std::string attr(CUdevice d, CUdevice_attribute key) {
    int value = 0; CUresult r = cuDeviceGetAttribute(&value, key, d);
    if (r == CUDA_SUCCESS) return "{\"value\":" + std::to_string(value) + ",\"cuda_code\":0}";
    return "{\"value\":null" + driver_details(r) + "}";
}
std::string hex_bytes(const unsigned char* p, size_t n) {
    std::ostringstream s; s << std::hex << std::setfill('0');
    for (size_t i=0; i<n; ++i) s << std::setw(2) << unsigned(p[i]);
    return s.str();
}
int numa_for(CUdevice d, int requested) {
    if (requested >= 0) return requested;
    int n = -1; check(cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, d), "host_numa_query");
    if (n < 0) fail("host_numa_query", "device reports no HOST_NUMA_ID; select explicit --numa only after platform verification");
    return n;
}
CUmemAllocationProp properties(int numa) {
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; p.location.id = numa; return p;
}
void info() {
    int version=0, count=0; check(cuDriverGetVersion(&version), "driver_version");
    check(cuDeviceGetCount(&count), "device_count");
    event("versions", ",\"driver_version\":" + std::to_string(version) + ",\"compile_cuda_version\":" + std::to_string(CUDA_VERSION) + ",\"device_count\":" + std::to_string(count));
    for (int i=0; i<count; ++i) {
        CUdevice d{}; char name[256]{}; CUuuid uuid{};
        check(cuDeviceGet(&d, i), "device_get"); check(cuDeviceGetName(name, sizeof(name), d), "device_name");
        check(cuDeviceGetUuid(&uuid, d), "device_uuid");
        std::string fields = ",\"ordinal\":" + std::to_string(i) + ",\"name\":" + quote(name) + ",\"uuid_hex\":" + quote(hex_bytes(reinterpret_cast<unsigned char*>(uuid.bytes), sizeof(uuid.bytes)));
        fields += ",\"vmm\":" + attr(d, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);
        fields += ",\"fabric_handle\":" + attr(d, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED);
        fields += ",\"host_numa_id\":" + attr(d, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID);
        // Numeric IDs are stable CUDA Driver ABI (confirmed in the CUDA 13.2 header).
        // Older drivers report a query error in this field instead of failing info.
        fields += ",\"host_numa_vmm\":" + attr(d, static_cast<CUdevice_attribute>(141));
        fields += ",\"host_numa_multinode_ipc\":" + attr(d, static_cast<CUdevice_attribute>(143));
        fields += ",\"unified_addressing\":" + attr(d, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        fields += ",\"pageable_memory_access\":" + attr(d, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
        fields += ",\"host_native_atomic\":" + attr(d, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED);
        fields += ",\"compute_major\":" + attr(d, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        fields += ",\"compute_minor\":" + attr(d, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        event("device_capabilities", fields);
        int n=-1; CUresult r=cuDeviceGetAttribute(&n, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, d);
        if (r==CUDA_SUCCESS && n>=0) {
            auto p=properties(n); size_t g=0;
            r=cuMemGetAllocationGranularity(&g,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            std::cout << "{\"status\":" << quote(r==CUDA_SUCCESS ? "PASS" : "UNAVAILABLE") << ",\"stage\":\"host_numa_fabric_granularity\",\"ordinal\":" << i << ",\"numa\":" << n << ",\"bytes\":" << g << driver_details(r) << "}" << std::endl;
        }
    }
}

__host__ __device__ uint32_t pattern(uint64_t i, uint32_t seed) {
    uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
__global__ void verify_words(const uint32_t* source, uint64_t words, uint32_t seed, unsigned long long* result) {
    unsigned long long mismatches=0, sum=0;
    for (uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<words; i+=uint64_t(gridDim.x)*blockDim.x) {
        uint32_t value=source[i];
        if (value!=pattern(i,seed)) ++mismatches;
        sum+=value;
    }
    if(mismatches)atomicAdd(result,mismatches);
    atomicAdd(result+1,sum);
}
struct Manifest {
    uint64_t bytes=0, allocation_bytes=0; uint32_t seed=0;
    unsigned char id[16]{}; CUmemFabricHandle fabric{};
};
void append(std::vector<unsigned char>& v, uint64_t x, size_t n) { for(size_t i=0;i<n;++i) v.push_back((x>>(i*8))&255); }
uint64_t consume(const std::vector<unsigned char>& v,size_t& pos,size_t n) {
    if(n>8 || pos+n>v.size()) fail("manifest_parse","truncated manifest");
    uint64_t x=0; for(size_t i=0;i<n;++i) x|=uint64_t(v[pos++])<<(i*8); return x;
}
void read_all(int fd, void* p, size_t bytes) {
    auto* out=static_cast<unsigned char*>(p); size_t offset=0;
    while(offset<bytes) { ssize_t n=read(fd,out+offset,bytes-offset); if(n<0&&errno==EINTR)continue;
        oscheck(n>=0,"file_read"); if(n==0)fail("file_read","unexpected EOF"); offset+=size_t(n); }
}
void write_all(int fd,const void* p,size_t bytes) {
    const auto* in=static_cast<const unsigned char*>(p); size_t offset=0;
    while(offset<bytes) { ssize_t n=write(fd,in+offset,bytes-offset); if(n<0&&errno==EINTR)continue;
        oscheck(n>0,"file_write"); offset+=size_t(n); }
}
void publish(const Manifest& m, PublishedFile& out) {
    std::vector<unsigned char> v={'E','G','M','P','R','B','0','1'};
    append(v,1,4); append(v,sizeof(CUmemFabricHandle),4); append(v,m.bytes,8); append(v,m.allocation_bytes,8); append(v,m.seed,4);
    v.insert(v.end(),m.id,m.id+sizeof(m.id));
    const auto* h=reinterpret_cast<const unsigned char*>(&m.fabric); v.insert(v.end(),h,h+sizeof(m.fabric));
    std::string temporary=out.path+".tmp."+std::to_string(getpid());
    Fd fd(open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600)); oscheck(fd.value>=0,"file_create");
    try {
        oscheck(fchmod(fd.value,0600)==0,"file_chmod"); write_all(fd.value,v.data(),v.size());
        oscheck(fsync(fd.value)==0,"file_fsync"); struct stat s{}; oscheck(fstat(fd.value,&s)==0,"file_stat");
        // link publishes atomically and refuses to replace an existing credential.
        oscheck(link(temporary.c_str(),out.path.c_str())==0,"file_publish_no_replace");
        out.dev=s.st_dev; out.ino=s.st_ino; out.published=true;
        oscheck(unlink(temporary.c_str())==0,"temporary_unlink");
    } catch(...) { unlink(temporary.c_str()); throw; }
}
Manifest read_manifest(const std::string& path) {
    Fd fd(open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)); oscheck(fd.value>=0,"file_open");
    struct stat s{}; oscheck(fstat(fd.value,&s)==0,"file_stat");
    size_t expected=8+4+4+8+8+4+16+sizeof(CUmemFabricHandle);
    if(!S_ISREG(s.st_mode)||s.st_uid!=geteuid()||(s.st_mode&0777)!=0600||s.st_size!=off_t(expected))
        fail("file_validate","manifest must be a regular 0600 file owned by current uid with exact protocol size");
    std::vector<unsigned char> v(expected); read_all(fd.value,v.data(),v.size());
    if(std::memcmp(v.data(),"EGMPRB01",8)!=0)fail("manifest_parse","invalid magic");
    size_t p=8;
    if(consume(v,p,4)!=1||consume(v,p,4)!=sizeof(CUmemFabricHandle))fail("manifest_parse","protocol/handle size mismatch");
    Manifest m; m.bytes=consume(v,p,8);m.allocation_bytes=consume(v,p,8);m.seed=uint32_t(consume(v,p,4));
    std::memcpy(m.id,v.data()+p,sizeof(m.id));p+=sizeof(m.id);std::memcpy(&m.fabric,v.data()+p,sizeof(m.fabric));
    if(!m.bytes||m.bytes%4||m.bytes>m.allocation_bytes||m.allocation_bytes>(1ULL<<30))fail("manifest_parse","invalid or excessive allocation size");
    return m;
}
void owner(const Options& o) {
    Context context(o.device); Mapping m; PublishedFile file; file.path=o.file;
    int numa=numa_for(context.device,o.numa); auto prop=properties(numa);size_t granularity=0;
    check(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"allocation_granularity");
    if(!granularity || o.bytes>SIZE_MAX-granularity+1)fail("allocation_granularity","invalid granularity/size");
    m.bytes=((o.bytes+granularity-1)/granularity)*granularity;
    if(m.bytes>(1ULL<<30))fail("allocation_granularity","padded allocation exceeds probe's 1GiB safety limit");
    check(cuMemCreate(&m.handle,m.bytes,&prop,0),"owner_create_host_numa_fabric"); m.allocated=true; m.map();
    CUmemAccessDesc access{};access.location.type=CU_MEM_LOCATION_TYPE_HOST_NUMA;access.location.id=numa;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(cuMemSetAccess(m.address,m.bytes,&access,1),"owner_cpu_set_access");
    Manifest manifest;manifest.bytes=o.bytes;manifest.allocation_bytes=m.bytes;
    { Fd fd(open("/dev/urandom",O_RDONLY|O_CLOEXEC));oscheck(fd.value>=0,"random_open");read_all(fd.value,manifest.id,sizeof(manifest.id)); }
    std::memcpy(&manifest.seed,manifest.id,sizeof(manifest.seed));
    auto* host=reinterpret_cast<uint32_t*>(m.address);
    for(uint64_t i=0;i<o.bytes/4;++i)host[i]=pattern(i,manifest.seed);
    // CPU stores complete before publication. GPU acceptance is determined by the consumer kernel.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    check(cuCtxSynchronize(),"owner_publish_sync");
    check(cuMemExportToShareableHandle(&manifest.fabric,m.handle,CU_MEM_HANDLE_TYPE_FABRIC,0),"owner_export_fabric");
    publish(manifest,file);
    event("owner_ready",",\"allocation_id\":"+quote(hex_bytes(manifest.id,sizeof(manifest.id)))+",\"pid\":"+std::to_string(getpid())+",\"device\":"+std::to_string(o.device)+",\"numa\":"+std::to_string(numa)+",\"logical_bytes\":"+std::to_string(o.bytes)+",\"allocation_bytes\":"+std::to_string(m.bytes)+",\"granularity_bytes\":"+std::to_string(granularity)+",\"allocation_count\":1,\"handle_file_mode\":\"0600\"");
    for(uint64_t elapsed=0;!stopped && (!o.seconds||elapsed<o.seconds);++elapsed)sleep(1);
    event("owner_stopping"); // File withdrawn before mapping/handle/context destructors run.
}
void consumer(const Options& o) {
    Manifest manifest=read_manifest(o.file); Context context(o.device); Mapping m;
    m.bytes=manifest.allocation_bytes;
    check(cuMemImportFromShareableHandle(&m.handle,&manifest.fabric,CU_MEM_HANDLE_TYPE_FABRIC),"consumer_import_fabric"); m.allocated=true;
    CUmemAllocationProp actual{};check(cuMemGetAllocationPropertiesFromHandle(&actual,m.handle),"consumer_allocation_properties");
    if(actual.location.type!=CU_MEM_LOCATION_TYPE_HOST_NUMA)fail("consumer_allocation_properties","imported allocation is not HOST_NUMA");
    m.map(); CUmemAccessDesc access{};access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;access.location.id=o.device;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READ;
    check(cuMemSetAccess(m.address,m.bytes,&access,1),"consumer_gpu_set_read_access");
    unsigned long long flags=0;check(cuMemGetAccess(&flags,&access.location,m.address),"consumer_gpu_get_access");
    if(flags!=CU_MEM_ACCESS_FLAGS_PROT_READ)fail("consumer_gpu_get_access","read-only mapping was not confirmed");
    event("consumer_mapped",",\"allocation_id\":"+quote(hex_bytes(manifest.id,sizeof(manifest.id)))+",\"device\":"+std::to_string(o.device)+",\"logical_bytes\":"+std::to_string(manifest.bytes)+",\"allocation_bytes\":"+std::to_string(m.bytes)+",\"import_count\":1,\"owner_allocation_created\":false,\"access\":\"READ\"");
    ResultBuffer result;rtcheck(cudaMalloc(reinterpret_cast<void**>(&result.ptr),2*sizeof(unsigned long long)),"result_allocate");
    DrainGuard drain{context,m,result};
    uint64_t words=manifest.bytes/4;unsigned long long expected_sum=0;
    for(uint64_t i=0;i<words;++i)expected_sum+=pattern(i,manifest.seed);
    for(uint64_t round=0;round<o.rounds && !stopped;++round) {
        drain.pending=true;
        rtcheck(cudaMemset(result.ptr,0,2*sizeof(unsigned long long)),"result_zero");
        verify_words<<<256,256>>>(reinterpret_cast<const uint32_t*>(m.address),words,manifest.seed,result.ptr);
        rtcheck(cudaGetLastError(),"consumer_kernel_launch");
        rtcheck(cudaDeviceSynchronize(),"consumer_kernel_synchronize");
        drain.pending=false;
        unsigned long long got[2]{};rtcheck(cudaMemcpy(got,result.ptr,sizeof(got),cudaMemcpyDeviceToHost),"result_readback");
        if(got[0]!=0 || got[1]!=expected_sum)fail("consumer_verify","mismatches="+std::to_string(got[0])+", checksum_match="+std::to_string(got[1]==expected_sum));
        event("consumer_verify",",\"allocation_id\":"+quote(hex_bytes(manifest.id,sizeof(manifest.id)))+",\"device\":"+std::to_string(o.device)+",\"round\":"+std::to_string(round)+",\"words_checked\":"+std::to_string(words)+",\"mismatches\":0,\"checksum_match\":true,\"private_result_bytes\":16");
    }
    check(cuCtxSynchronize(),"consumer_final_drain");
}
} // namespace
int main(int argc,char** argv) {
    std::signal(SIGINT,stop_handler);std::signal(SIGTERM,stop_handler);umask(0077);
    try {
        Options o=parse(argc,argv);check(cuInit(0),"driver_init");
        if(o.mode=="info")info();else if(o.mode=="owner")owner(o);else consumer(o);
        if(cleanup_failed)return 3;
        event("complete",",\"mode\":"+quote(o.mode)+",\"interrupted\":"+(stopped?"true":"false"));return stopped?130:0;
    } catch(const ReportedError&) { return 1; }
    catch(const std::exception& e) {
        std::cout << "{\"status\":\"FAIL\",\"stage\":\"unexpected_exception\",\"message\":" << quote(e.what()) << "}" << std::endl;
        return 1;
    }
}
