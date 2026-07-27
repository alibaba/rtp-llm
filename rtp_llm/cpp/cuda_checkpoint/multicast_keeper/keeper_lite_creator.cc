#include "rtp_llm/cpp/cuda_checkpoint/multicast_keeper/keeper_protocol.h"

#include <cuda.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <limits>
#include <set>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

void usage(FILE* stream, const char* program) {
    fprintf(stream,
            "Usage: %s --gpus LIST --size BYTES --num-devices N "
            "--handle-types MASK --flags 0 --deposit-fd FD [--import-fabric-fd FD] [--dry-run]\n\n"
            "LIST is a comma-separated list of CUDA ordinals. BYTES accepts raw bytes\n"
            "or KiB/MiB/GiB suffixes. The holder normally invokes this process.\n\n"
            "Default (create) mode calls cuMulticastCreate for the whole team\n"
            "(--num-devices = configured global team size) and adds only LIST.\n"
            "With --import-fabric-fd, the process instead imports a 64-byte FABRIC\n"
            "handle read from FD, AddDevice's the LIST (local) devices, and re-exports\n"
            "a node-local POSIX fd — the peer-node path for cross-machine MNNVL teams.\n",
            program);
}

// Read exactly RTP_MC_FABRIC_HANDLE_BYTES from fd into out. Returns true on a
// full read. Used by the peer-node importer to receive the fabric handle.
bool readFabricHandle(int fd, unsigned char* out) {
    size_t got = 0;
    while (got < RTP_MC_FABRIC_HANDLE_BYTES) {
        ssize_t n = read(fd, out + got, RTP_MC_FABRIC_HANDLE_BYTES - got);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        got += (size_t)n;
    }
    return got == RTP_MC_FABRIC_HANDLE_BYTES;
}

bool parseDecimal(const char* text, uint64_t* value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    char* end                 = nullptr;
    errno                     = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t)parsed;
    return true;
}

bool parseUnsigned(const char* text, uint64_t* value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    char* end                 = nullptr;
    errno                     = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text) {
        return false;
    }
    uint64_t multiplier = 1;
    if (*end != '\0') {
        std::string suffix(end);
        if (suffix == "K" || suffix == "KiB" || suffix == "kib") {
            multiplier = 1024ull;
        } else if (suffix == "M" || suffix == "MiB" || suffix == "mib") {
            multiplier = 1024ull * 1024ull;
        } else if (suffix == "G" || suffix == "GiB" || suffix == "gib") {
            multiplier = 1024ull * 1024ull * 1024ull;
        } else {
            return false;
        }
    }
    if (parsed == 0 || parsed > std::numeric_limits<uint64_t>::max() / multiplier) {
        return false;
    }
    *value = (uint64_t)parsed * multiplier;
    return true;
}

bool parseGpuList(const char* text, std::vector<int>* gpus) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    std::set<int> seen;
    const char*   cursor = text;
    while (*cursor != '\0') {
        char* end = nullptr;
        errno     = 0;
        long gpu  = strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || gpu < 0 || gpu > std::numeric_limits<int>::max()
            || !seen.insert((int)gpu).second) {
            return false;
        }
        gpus->push_back((int)gpu);
        if (*end == '\0') {
            break;
        }
        if (*end != ',' || end[1] == '\0') {
            return false;
        }
        cursor = end + 1;
    }
    return !gpus->empty();
}

bool sendResult(int socket_fd, const rtp_mc_creator_result& result, int multicast_fd) {
    struct iovec iov = {
        .iov_base = const_cast<rtp_mc_creator_result*>(&result),
        .iov_len  = sizeof(result),
    };
    char control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof(control));
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov    = &iov;
    header.msg_iovlen = 1;
    if (multicast_fd >= 0) {
        header.msg_control    = control;
        header.msg_controllen = sizeof(control);
        struct cmsghdr* cmsg  = CMSG_FIRSTHDR(&header);
        cmsg->cmsg_level      = SOL_SOCKET;
        cmsg->cmsg_type       = SCM_RIGHTS;
        cmsg->cmsg_len        = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &multicast_fd, sizeof(multicast_fd));
    }
    return sendmsg(socket_fd, &header, MSG_NOSIGNAL) == (ssize_t)sizeof(result);
}

void printCudaError(const char* expression, CUresult result) {
    const char* name        = "unknown";
    const char* description = "unknown";
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    fprintf(stderr, "keeper creator: %s failed: %s (%s)\n", expression, name, description);
}

#define CUDA_CHECK(expression)                                                                                         \
    do {                                                                                                               \
        CUresult result = (expression);                                                                                \
        if (result != CUDA_SUCCESS) {                                                                                  \
            printCudaError(#expression, result);                                                                       \
            goto cleanup;                                                                                              \
        }                                                                                                              \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    const char*                gpu_text          = nullptr;
    const char*                size_text         = nullptr;
    const char*                num_devices_text  = nullptr;
    const char*                handle_types_text = nullptr;
    const char*                flags_text        = nullptr;
    int                        deposit_fd        = -1;
    int                        import_fabric_fd  = -1;
    bool                       dry_run           = false;
    static const struct option options[]         = {
        {"gpus", required_argument, nullptr, 'g'},
        {"size", required_argument, nullptr, 's'},
        {"num-devices", required_argument, nullptr, 'N'},
        {"handle-types", required_argument, nullptr, 'T'},
        {"flags", required_argument, nullptr, 'f'},
        {"deposit-fd", required_argument, nullptr, 'd'},
        {"import-fabric-fd", required_argument, nullptr, 'i'},
        {"dry-run", no_argument, nullptr, 'n'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "g:s:N:T:f:d:i:nh", options, nullptr)) != -1) {
        switch (option) {
            case 'g':
                gpu_text = optarg;
                break;
            case 's':
                size_text = optarg;
                break;
            case 'N':
                num_devices_text = optarg;
                break;
            case 'T':
                handle_types_text = optarg;
                break;
            case 'f':
                flags_text = optarg;
                break;
            case 'd': {
                char* end    = nullptr;
                long  parsed = strtol(optarg, &end, 10);
                if (end == optarg || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                    fprintf(stderr, "invalid --deposit-fd: %s\n", optarg);
                    return 2;
                }
                deposit_fd = (int)parsed;
                break;
            }
            case 'i': {
                char* end    = nullptr;
                long  parsed = strtol(optarg, &end, 10);
                if (end == optarg || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                    fprintf(stderr, "invalid --import-fabric-fd: %s\n", optarg);
                    return 2;
                }
                import_fabric_fd = (int)parsed;
                break;
            }
            case 'n':
                dry_run = true;
                break;
            case 'h':
                usage(stdout, argv[0]);
                return 0;
            default:
                usage(stderr, argv[0]);
                return 2;
        }
    }
    std::vector<int> gpu_ordinals;
    uint64_t         requested_size         = 0;
    uint64_t         requested_num_devices  = 0;
    uint64_t         requested_handle_types = 0;
    uint64_t         requested_flags        = 0;
    if (optind != argc || !parseGpuList(gpu_text, &gpu_ordinals) || !parseUnsigned(size_text, &requested_size)
        || !parseDecimal(num_devices_text, &requested_num_devices)
        || !parseDecimal(handle_types_text, &requested_handle_types) || !parseDecimal(flags_text, &requested_flags)
        || requested_num_devices > std::numeric_limits<unsigned int>::max() || requested_handle_types == 0
        || requested_handle_types > std::numeric_limits<uint32_t>::max()
        || (requested_handle_types & ~RTP_MC_SUPPORTED_HANDLE_TYPES) != 0 || requested_flags != 0
        || (!dry_run && deposit_fd < 0)) {
        usage(stderr, argv[0]);
        return 2;
    }
    // The holder enforces FABRIC numDevices against its explicit global team
    // contract; the creator independently rejects totals smaller than LIST. The
    // single-node POSIX path keeps the strict local == total invariant.
    const bool want_fabric = (import_fabric_fd >= 0) || (requested_handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0;
    if (want_fabric ? (requested_num_devices < gpu_ordinals.size()) : (requested_num_devices != gpu_ordinals.size())) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (dry_run) {
        printf("CREATOR_CONFIG gpus=%s num_devices=%zu requested_size=%llu "
               "handle_types=0x%llx flags=%llu deposit_fd=%d no_cuda=1\n",
               gpu_text,
               gpu_ordinals.size(),
               (unsigned long long)requested_size,
               (unsigned long long)requested_handle_types,
               (unsigned long long)requested_flags,
               deposit_fd);
        return 0;
    }

    rtp_mc_creator_result response = {
        .magic          = RTP_MC_CREATOR_MAGIC,
        .requested_size = requested_size,
        .served_size    = 0,
        .status         = 1,
        .flags          = 0,
        .fabric_handle  = {0},
    };
    std::vector<CUdevice>        devices;
    std::vector<CUcontext>       contexts;
    CUmemGenericAllocationHandle multicast_handle = 0;
    int                          multicast_fd     = -1;
    int                          exit_code        = 1;
    unsigned char                fabric_handle[RTP_MC_FABRIC_HANDLE_BYTES];
    memset(fabric_handle, 0, sizeof(fabric_handle));

    if (import_fabric_fd >= 0 && !readFabricHandle(import_fabric_fd, fabric_handle)) {
        fprintf(stderr, "keeper creator: failed to read 64-byte fabric handle from --import-fabric-fd\n");
        goto cleanup;
    }

    CUDA_CHECK(cuInit(0));
    int available_devices = 0;
    CUDA_CHECK(cuDeviceGetCount(&available_devices));
    for (int ordinal : gpu_ordinals) {
        if (ordinal >= available_devices) {
            fprintf(stderr, "keeper creator: GPU ordinal %d is outside device count %d\n", ordinal, available_devices);
            goto cleanup;
        }
        CUdevice  device;
        CUcontext context;
        CUDA_CHECK(cuDeviceGet(&device, ordinal));
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
        devices.push_back(device);
        contexts.push_back(context);
    }
    CUDA_CHECK(cuCtxSetCurrent(contexts.front()));

    if (import_fabric_fd >= 0) {
        // Peer-node importer: import the fabric team the creator node produced,
        // add this node's LOCAL devices, then re-export a node-local POSIX fd so
        // local ranks import exactly as they do on the creator node. No
        // cuMulticastCreate — the object already exists in the fabric.
        CUmemFabricHandle imported;
        memset(&imported, 0, sizeof(imported));
        memcpy(&imported, fabric_handle, sizeof(imported));
        CUDA_CHECK(cuMemImportFromShareableHandle(&multicast_handle, &imported, CU_MEM_HANDLE_TYPE_FABRIC));
        for (CUdevice device : devices) {
            CUDA_CHECK(cuMulticastAddDevice(multicast_handle, device));
        }
        CUDA_CHECK(
            cuMemExportToShareableHandle(&multicast_fd, multicast_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        response.served_size = requested_size;
        response.status      = 0;
        if (!sendResult(deposit_fd, response, multicast_fd)) {
            perror("keeper creator sendmsg");
            response.status = 1;
            goto cleanup;
        }
        printf("IMPORTER_DEPOSITED pid=%d gpus=%s num_devices=%llu\n",
               (int)getpid(),
               gpu_text,
               (unsigned long long)requested_num_devices);
        fflush(stdout);
        exit_code = 0;
        goto cleanup;
    }

    CUmulticastObjectProp properties;
    memset(&properties, 0, sizeof(properties));
    // numDevices is the configured whole team for FABRIC and the local list size
    // for POSIX. This node adds only its local devices; peers add theirs via the
    // importer path.
    properties.numDevices  = (unsigned int)requested_num_devices;
    properties.size        = (size_t)requested_size;
    properties.handleTypes = want_fabric ? (CU_MEM_HANDLE_TYPE_FABRIC | CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) :
                                           CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity     = 0;
    CUDA_CHECK(cuMulticastGetGranularity(&granularity, &properties, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (granularity != 0 && properties.size % granularity != 0) {
        if (properties.size > std::numeric_limits<size_t>::max() - granularity) {
            fprintf(stderr, "keeper creator: rounded multicast size overflows size_t\n");
            goto cleanup;
        }
        properties.size = ((properties.size + granularity - 1) / granularity) * granularity;
    }
    CUDA_CHECK(cuMulticastCreate(&multicast_handle, &properties));
    for (CUdevice device : devices) {
        CUDA_CHECK(cuMulticastAddDevice(multicast_handle, device));
    }
    CUDA_CHECK(
        cuMemExportToShareableHandle(&multicast_fd, multicast_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    if (want_fabric) {
        // Export the 64-byte fabric handle inline for cross-node distribution.
        CUmemFabricHandle exported;
        memset(&exported, 0, sizeof(exported));
        CUDA_CHECK(cuMemExportToShareableHandle(&exported, multicast_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        memcpy(response.fabric_handle, &exported, sizeof(exported));
        response.flags |= RTP_MC_CREATOR_FLAG_FABRIC_VALID;
    }

    response.served_size = (uint64_t)properties.size;
    response.status      = 0;
    if (!sendResult(deposit_fd, response, multicast_fd)) {
        perror("keeper creator sendmsg");
        response.status = 1;
        goto cleanup;
    }
    printf("CREATOR_DEPOSITED pid=%d gpus=%s requested=%llu served=%llu granularity=%zu\n",
           (int)getpid(),
           gpu_text,
           (unsigned long long)requested_size,
           (unsigned long long)response.served_size,
           granularity);
    fflush(stdout);
    exit_code = 0;

cleanup:
    if (multicast_fd >= 0) {
        close(multicast_fd);
    }
    // On success, match the validated upstream lite creator: exit directly and
    // let process teardown release CUDA handles and primary contexts together.
    // Releasing the multicast handle and each primary context piecemeal can
    // dismantle the cross-device ready state before every rank imports the FD.
    if (exit_code == 0) {
        close(deposit_fd);
        return 0;
    }
    if (multicast_handle != 0) {
        (void)cuMemRelease(multicast_handle);
    }
    (void)cuCtxSetCurrent(nullptr);
    for (size_t i = 0; i < contexts.size(); ++i) {
        (void)cuDevicePrimaryCtxRelease(devices[i]);
    }
    close(deposit_fd);
    return exit_code;
}
