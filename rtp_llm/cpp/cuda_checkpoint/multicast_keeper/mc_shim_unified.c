// Derived from nekyia examples/megamoe_ckpt_keeper/mc_shim_unified.c at
// commit 5e417f2cba5f4ecf73ba7ab5bb3241473cc4bc6d. See UPSTREAM.md.
#define _GNU_SOURCE

#include "rtp_llm/cpp/cuda_checkpoint/multicast_keeper/keeper_protocol.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

typedef int                CUresult;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int                CUdevice;
typedef void*              CUcontext;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR 0x1
#define CU_MEM_HANDLE_TYPE_FABRIC 0x8

typedef struct CUmulticastObjectProp {
    unsigned int       numDevices;
    size_t             size;
    unsigned long long handleTypes;
    unsigned long long flags;
} CUmulticastObjectProp;

typedef int (*getentry_fn)(const char*, void**, unsigned int, unsigned long long, int*);
typedef CUresult (*getproc_v2_fn)(const char*, void**, int, unsigned long long, int*);
typedef CUresult (*import_fn)(CUmemGenericAllocationHandle*, void*, int);
typedef CUresult (*export_fn)(void*, CUmemGenericAllocationHandle, int, unsigned long long);
typedef CUresult (*mccreate_fn)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
typedef CUresult (*mcadd_fn)(CUmemGenericAllocationHandle, CUdevice);
typedef CUresult (*memrelease_fn)(CUmemGenericAllocationHandle);
typedef CUresult (*init_fn)(unsigned int);
typedef CUresult (*device_get_count_fn)(int*);
typedef CUresult (*device_get_fn)(CUdevice*, int);
typedef CUresult (*ctx_get_device_fn)(CUdevice*);
typedef CUresult (*primary_ctx_retain_fn)(CUcontext*, CUdevice);
typedef void* (*dlsym_fn)(void*, const char*);

static dlsym_fn           real_dlsym      = NULL;
static getentry_fn        real_getentry   = NULL;
static getproc_v2_fn      real_getproc    = NULL;
static import_fn          real_import     = NULL;
static export_fn          real_export     = NULL;
static mccreate_fn        real_mccreate   = NULL;
static mcadd_fn           real_mcadd      = NULL;
static memrelease_fn      real_memrelease = NULL;
static void*              g_libcuda       = NULL;
static int                g_cuda_version  = 0;
static unsigned long long g_cuda_flags    = 0;

typedef struct keeper_handle {
    CUmemGenericAllocationHandle handle;
    rtp_mc_token                 token;
    int                          occupied;
    int                          created_locally;
    int                          released;
    int                          pending;
} keeper_handle;

typedef struct raw_fabric_import {
    CUmemGenericAllocationHandle handle;
    unsigned char                fabric[RTP_MC_FABRIC_HANDLE_BYTES];
    int                          keeper_registered;
} raw_fabric_import;

static pthread_mutex_t    g_lock = PTHREAD_MUTEX_INITIALIZER;
static keeper_handle      g_handles[1024];
static raw_fabric_import* g_raw_fabric_imports = NULL;
// Raw FABRIC IMPORT_ADD registers this process incarnation with the holder.
// Keep that registration independently of transient CUDA handles so repeated
// cuMemRelease/rebuild cycles release it exactly once at process teardown.
static rtp_mc_token   g_peer_refs[1024];
static size_t         g_peer_ref_count             = 0;
static size_t         g_handle_count               = 0;
static size_t         g_raw_fabric_import_count    = 0;
static size_t         g_raw_fabric_import_capacity = 0;
static pthread_once_t g_peer_context_once          = PTHREAD_ONCE_INIT;
static CUresult       g_peer_context_result        = CUDA_SUCCESS;
static int            g_peer_device_count          = -1;

// Owner attribution sent with every CREATE/FETCH/RELEASE. owner_id is a logical,
// restart-stable owner key; owner_generation is a per-process nonce that changes
// on relaunch. Computed once so the whole checkpoint/restore cycle shares one
// generation while a genuine restart presents a new one (see keeper_protocol.h).
static pthread_once_t g_owner_once       = PTHREAD_ONCE_INIT;
static uint64_t       g_owner_id         = 0;
static uint64_t       g_owner_generation = 0;

static void keeper_send_release(const rtp_mc_token* token);

static int env_is_one(const char* name) {
    const char* value = getenv(name);
    return value != NULL && strcmp(value, "1") == 0;
}

static int keeper_enabled(void) {
    return env_is_one("RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER");
}

static int verbose_logging(void) {
    return env_is_one("RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER_DEBUG");
}

static void log_message(int always, const char* format, ...) {
    if (!always && !verbose_logging()) {
        return;
    }
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[rtp-mc-shim pid=%d] ", (int)getpid());
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(args);
}

static dlsym_fn get_real_dlsym(void) {
    if (real_dlsym == NULL) {
        real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    return real_dlsym;
}

static void* driver_symbol(const char* name) {
    dlsym_fn lookup = get_real_dlsym();
    if (lookup == NULL) {
        return NULL;
    }
    pthread_mutex_lock(&g_lock);
    if (g_libcuda == NULL) {
        g_libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    }
    void* symbol = g_libcuda == NULL ? NULL : lookup(g_libcuda, name);
    pthread_mutex_unlock(&g_lock);
    return symbol;
}

// A ready multicast object can only be imported by every participant when each
// process has retained the primary contexts for its visible peer GPUs. The
// references intentionally live for the rank's lifetime so cuda-checkpoint can
// checkpoint/restore them together with the rank's active context.
static void retain_peer_contexts_once(void) {
    init_fn               init               = (init_fn)driver_symbol("cuInit");
    device_get_count_fn   get_count          = (device_get_count_fn)driver_symbol("cuDeviceGetCount");
    device_get_fn         get_device         = (device_get_fn)driver_symbol("cuDeviceGet");
    ctx_get_device_fn     get_current_device = (ctx_get_device_fn)driver_symbol("cuCtxGetDevice");
    primary_ctx_retain_fn retain             = (primary_ctx_retain_fn)driver_symbol("cuDevicePrimaryCtxRetain");
    if (init == NULL || get_count == NULL || get_device == NULL || retain == NULL) {
        g_peer_context_result = CUDA_ERROR_INVALID_VALUE;
        return;
    }
    CUresult result = init(0);
    if (result != CUDA_SUCCESS) {
        g_peer_context_result = result;
        return;
    }
    int device_count = 0;
    result           = get_count(&device_count);
    if (result != CUDA_SUCCESS) {
        g_peer_context_result = result;
        return;
    }
    g_peer_device_count     = device_count;
    CUdevice current_device = -1;
    if (get_current_device != NULL) {
        (void)get_current_device(&current_device);
    }
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        CUdevice device = -1;
        result          = get_device(&device, ordinal);
        if (result != CUDA_SUCCESS) {
            g_peer_context_result = result;
            return;
        }
        if (device == current_device) {
            continue;
        }
        CUcontext context = NULL;
        result            = retain(&context, device);
        if (result != CUDA_SUCCESS) {
            g_peer_context_result = result;
            return;
        }
        log_message(0, "retained peer primary context ordinal=%d device=%d", ordinal, device);
    }
}

static CUresult ensure_peer_contexts(void) {
    pthread_once(&g_peer_context_once, retain_peer_contexts_once);
    if (g_peer_context_result != CUDA_SUCCESS) {
        log_message(1, "peer primary context initialization failed result=%d", g_peer_context_result);
    }
    return g_peer_context_result;
}

static const char* keeper_socket_path(char* buffer, size_t size) {
    const char* explicit_socket = getenv("RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET");
    if (explicit_socket != NULL && explicit_socket[0] != '\0') {
        if (snprintf(buffer, size, "%s", explicit_socket) >= (int)size) {
            return NULL;
        }
        return buffer;
    }
    const char* directory = getenv("NEKYIA_KEEPER_DIR");
    if (directory == NULL || directory[0] == '\0') {
        directory = "/tmp";
    }
    if (snprintf(buffer, size, "%s/%s", directory, RTP_MC_DEFAULT_SOCKET_NAME) >= (int)size) {
        return NULL;
    }
    return buffer;
}

static int timeout_from_env(const char* name, int default_value) {
    const char* text = getenv(name);
    if (text == NULL || text[0] == '\0') {
        return default_value;
    }
    char* end  = NULL;
    errno      = 0;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 100 || value > 600000) {
        log_message(1, "invalid %s=%s", name, text);
        return default_value;
    }
    return (int)value;
}

static int parse_env_u64(const char* name, uint64_t* out) {
    const char* text = getenv(name);
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    char* end                = NULL;
    errno                    = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int parse_gpu_list(const char* text, int* values, size_t capacity, size_t* count) {
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    size_t      parsed_count = 0;
    const char* cursor       = text;
    while (*cursor != '\0') {
        char* end  = NULL;
        errno      = 0;
        long value = strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || value < 0 || value > INT_MAX || parsed_count == capacity) {
            return 0;
        }
        for (size_t i = 0; i < parsed_count; ++i) {
            if (values[i] == (int)value) {
                return 0;
            }
        }
        values[parsed_count++] = (int)value;
        if (*end == '\0') {
            break;
        }
        if (*end != ',' || end[1] == '\0') {
            return 0;
        }
        cursor = end + 1;
    }
    *count = parsed_count;
    return parsed_count > 0;
}

// FABRIC is allowed only under the explicit launcher contract. The holder list
// names physical ordinals; a rank either exposes that exact ordered list through
// CUDA_VISIBLE_DEVICES or, when unset, sees the same dense ordinal set.
static int validate_fabric_team_contract(const rtp_mc_object_properties* properties) {
    uint64_t configured_team_size = 0;
    if (!parse_env_u64("RTP_LLM_MC_FABRIC_TEAM_SIZE", &configured_team_size) || configured_team_size > UINT32_MAX
        || properties->num_devices != (uint32_t)configured_team_size) {
        log_message(1, "FABRIC team size does not match RTP_LLM_MC_FABRIC_TEAM_SIZE");
        return 0;
    }

    int    configured[256];
    size_t configured_count = 0;
    if (!parse_gpu_list(getenv("RTP_LLM_MC_LOCAL_GPUS"), configured, 256, &configured_count)
        || configured_count != (size_t)g_peer_device_count) {
        log_message(
            1, "FABRIC local GPU count mismatch configured=%zu visible=%d", configured_count, g_peer_device_count);
        return 0;
    }

    const char* visible_text = getenv("CUDA_VISIBLE_DEVICES");
    if (visible_text != NULL && visible_text[0] != '\0') {
        int    visible[256];
        size_t visible_count = 0;
        if (!parse_gpu_list(visible_text, visible, 256, &visible_count) || visible_count != configured_count) {
            log_message(1, "CUDA_VISIBLE_DEVICES is not the configured integer GPU list");
            return 0;
        }
        for (size_t i = 0; i < configured_count; ++i) {
            if (visible[i] != configured[i]) {
                log_message(1, "CUDA_VISIBLE_DEVICES differs from RTP_LLM_MC_LOCAL_GPUS");
                return 0;
            }
        }
    } else {
        for (size_t i = 0; i < configured_count; ++i) {
            if (configured[i] != (int)i) {
                log_message(1, "non-dense local GPU list requires matching CUDA_VISIBLE_DEVICES");
                return 0;
            }
        }
    }
    return 1;
}

static uint64_t random_nonzero_u64(void) {
    uint64_t value  = 0;
    size_t   offset = 0;
    while (offset < sizeof(value)) {
        ssize_t count = getrandom((char*)&value + offset, sizeof(value) - offset, 0);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        offset += (size_t)count;
    }
    if (value == 0) {
        value = ((uint64_t)getpid() << 32) ^ (uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ull;
        if (value == 0) {
            value = 0x9e3779b97f4a7c15ull;
        }
    }
    return value;
}

static void init_owner_identity(void) {
    uint64_t owner_id = 0;
    if (parse_env_u64("RTP_LLM_MC_OWNER_ID", &owner_id)) {
        g_owner_id = owner_id;
    } else {
        // Fall back to the distributed rank slot, which is stable across a
        // backend restart. Bias by one so rank 0 is a real (nonzero) owner and
        // does not collapse into the anonymous (no-reclamation) owner_id 0.
        uint64_t rank = 0;
        if (parse_env_u64("LOCAL_RANK", &rank) || parse_env_u64("RANK", &rank)) {
            g_owner_id = rank + 1;
        } else {
            g_owner_id = 0;  // anonymous: RELEASE still works, generation reclaim disabled
        }
    }
    uint64_t generation = 0;
    if (!parse_env_u64("RTP_LLM_MC_OWNER_GENERATION", &generation) || generation == 0) {
        generation = random_nonzero_u64();
    }
    g_owner_generation = generation;
    log_message(0,
                "owner identity owner_id=%llu generation=%llu",
                (unsigned long long)g_owner_id,
                (unsigned long long)g_owner_generation);
}

static void ensure_owner_identity(void) {
    pthread_once(&g_owner_once, init_owner_identity);
}

static int64_t monotonic_milliseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int set_socket_timeout(int fd, int timeout_ms) {
    struct timeval timeout = {
        .tv_sec  = timeout_ms / 1000,
        .tv_usec = (timeout_ms % 1000) * 1000,
    };
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0
                   && setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 ?
               0 :
               -1;
}

static int connect_to_keeper(const char* path, int timeout_ms) {
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(address.sun_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", path);
    int64_t deadline = monotonic_milliseconds() + timeout_ms;
    do {
        int socket_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (socket_fd < 0) {
            return -1;
        }
        if (connect(socket_fd, (struct sockaddr*)&address, sizeof(address)) == 0) {
            int64_t now       = monotonic_milliseconds();
            int     remaining = now < 0 || now >= deadline ? 1 : (int)(deadline - now);
            if (set_socket_timeout(socket_fd, remaining) != 0) {
                int saved_errno = errno;
                close(socket_fd);
                errno = saved_errno;
                return -1;
            }
            return socket_fd;
        }
        int saved_errno = errno;
        close(socket_fd);
        if (saved_errno != ENOENT && saved_errno != ECONNREFUSED) {
            errno = saved_errno;
            return -1;
        }
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 20000000};
        nanosleep(&delay, NULL);
    } while (monotonic_milliseconds() < deadline);
    errno = ETIMEDOUT;
    return -1;
}

static int
keeper_request(const rtp_mc_request* request, const unsigned char* trailing_fabric, rtp_mc_response* response) {
    char path[sizeof(((struct sockaddr_un*)0)->sun_path)];
    if (keeper_socket_path(path, sizeof(path)) == NULL) {
        log_message(1, "keeper socket path is too long");
        return -1;
    }
    // CREATE and IMPORT_ADD both fork a short-lived CUDA child in the holder, so
    // they share the long creator deadline; every other opcode is a quick lookup.
    int forks_child     = request->opcode == RTP_MC_OP_CREATE || request->opcode == RTP_MC_OP_IMPORT_ADD;
    int default_timeout = forks_child ? 125000 : 5000;
    int timeout_ms = timeout_from_env(forks_child ? "RTP_LLM_MC_CREATE_TIMEOUT_MS" : "RTP_LLM_MC_REQUEST_TIMEOUT_MS",
                                      default_timeout);
    int socket_fd = connect_to_keeper(path, timeout_ms);
    if (socket_fd < 0) {
        log_message(1, "keeper connect failed path=%s error=%s", path, strerror(errno));
        return -1;
    }
    // Always transmit the extended request so the holder can attribute the
    // object to this owner and reclaim it after a restart. IMPORT_ADD appends a
    // 64-byte fabric handle, producing the 144-byte message form.
    ensure_owner_identity();
    rtp_mc_import_add_request message;
    memset(&message, 0, sizeof(message));
    message.ext.base             = *request;
    message.ext.owner_id         = g_owner_id;
    message.ext.owner_generation = g_owner_generation;
    size_t message_size;
    if (trailing_fabric != NULL) {
        memcpy(message.fabric_handle, trailing_fabric, RTP_MC_FABRIC_HANDLE_BYTES);
        message_size = sizeof(rtp_mc_import_add_request);
    } else {
        message_size = sizeof(rtp_mc_request_ext);
    }
    message.ext.base.struct_size = (uint32_t)message_size;
    if (send(socket_fd, &message, message_size, MSG_NOSIGNAL) != (ssize_t)message_size) {
        close(socket_fd);
        return -1;
    }
    memset(response, 0, sizeof(*response));
    struct iovec iov = {.iov_base = response, .iov_len = sizeof(*response)};
    char         control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof(control));
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov        = &iov;
    header.msg_iovlen     = 1;
    header.msg_control    = control;
    header.msg_controllen = sizeof(control);
    ssize_t received      = recvmsg(socket_fd, &header, MSG_CMSG_CLOEXEC);
    int     received_fd   = -1;
    if (received == (ssize_t)sizeof(*response) && !(header.msg_flags & MSG_CTRUNC)) {
        for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header); cmsg != NULL; cmsg = CMSG_NXTHDR(&header, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS
                && cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
                memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(received_fd));
                break;
            }
        }
    }
    close(socket_fd);
    if (received != (ssize_t)sizeof(*response) || (header.msg_flags & (MSG_TRUNC | MSG_CTRUNC))
        || response->magic != RTP_MC_PROTOCOL_MAGIC || response->version != RTP_MC_PROTOCOL_VERSION
        || response->struct_size != sizeof(*response) || response->opcode != request->opcode) {
        if (received_fd >= 0) {
            close(received_fd);
        }
        log_message(1, "keeper returned an invalid protocol response opcode=%u", request->opcode);
        return -1;
    }
    if (response->status != RTP_MC_STATUS_OK || received_fd < 0) {
        if (received_fd >= 0) {
            close(received_fd);
        }
        log_message(1,
                    "keeper request failed opcode=%u status=%d object=%llu",
                    request->opcode,
                    response->status,
                    (unsigned long long)request->object_id);
        errno = EPROTO;
        return -1;
    }
    return received_fd;
}

static int same_token_properties(const rtp_mc_token* left, const rtp_mc_token* right) {
    return left->properties.size == right->properties.size
           && left->properties.num_devices == right->properties.num_devices
           && left->properties.handle_types == right->properties.handle_types
           && left->properties.flags == right->properties.flags;
}

static int same_token_identity(const rtp_mc_token* left, const rtp_mc_token* right) {
    return left->holder_instance_hi == right->holder_instance_hi
           && left->holder_instance_lo == right->holder_instance_lo && left->object_id == right->object_id
           && same_token_properties(left, right);
}

static int remember_peer_ref(const rtp_mc_token* token) {
    pthread_mutex_lock(&g_lock);
    for (size_t i = 0; i < g_peer_ref_count; ++i) {
        if (same_token_identity(&g_peer_refs[i], token)) {
            pthread_mutex_unlock(&g_lock);
            return 0;
        }
    }
    if (g_peer_ref_count == sizeof(g_peer_refs) / sizeof(g_peer_refs[0])) {
        pthread_mutex_unlock(&g_lock);
        log_message(1, "keeper peer reference table is full");
        return -1;
    }
    g_peer_refs[g_peer_ref_count++] = *token;
    pthread_mutex_unlock(&g_lock);
    return 1;
}

static int valid_token(const rtp_mc_token* token) {
    return token != NULL && memcmp(token->magic, RTP_MC_TOKEN_MAGIC, sizeof(token->magic)) == 0
           && token->version == RTP_MC_PROTOCOL_VERSION && token->token_size == sizeof(*token) && token->reserved == 0
           && (token->holder_instance_hi != 0 || token->holder_instance_lo != 0) && token->object_id != 0
           && token->properties.size != 0 && token->properties.num_devices != 0 && token->properties.handle_types != 0
           && (token->properties.handle_types & ~RTP_MC_SUPPORTED_HANDLE_TYPES) == 0 && token->properties.flags == 0;
}

static int token_from_properties(const CUmulticastObjectProp* properties, rtp_mc_token* token) {
    if (properties == NULL || properties->size == 0 || properties->numDevices == 0 || properties->handleTypes == 0
        || properties->handleTypes > UINT32_MAX
        || (properties->handleTypes & ~((unsigned long long)RTP_MC_SUPPORTED_HANDLE_TYPES)) != 0
        || properties->flags != 0) {
        return -1;
    }
    memset(token, 0, sizeof(*token));
    memcpy(token->magic, RTP_MC_TOKEN_MAGIC, sizeof(token->magic));
    token->version                 = RTP_MC_PROTOCOL_VERSION;
    token->token_size              = sizeof(*token);
    token->properties.size         = properties->size;
    token->properties.num_devices  = properties->numDevices;
    token->properties.handle_types = (uint32_t)properties->handleTypes;
    token->properties.flags        = properties->flags;
    return 0;
}

static rtp_mc_request request_from_token(uint16_t opcode, const rtp_mc_token* token) {
    rtp_mc_request request;
    memset(&request, 0, sizeof(request));
    request.magic       = RTP_MC_PROTOCOL_MAGIC;
    request.version     = RTP_MC_PROTOCOL_VERSION;
    request.opcode      = opcode;
    request.struct_size = sizeof(request);
    request.properties  = token->properties;
    if (opcode == RTP_MC_OP_FETCH || opcode == RTP_MC_OP_RELEASE || opcode == RTP_MC_OP_FETCH_FABRIC) {
        request.holder_instance_hi = token->holder_instance_hi;
        request.holder_instance_lo = token->holder_instance_lo;
        request.object_id          = token->object_id;
    }
    return request;
}

static int find_keeper_handle(CUmemGenericAllocationHandle handle) {
    for (size_t i = 0; i < sizeof(g_handles) / sizeof(g_handles[0]); ++i) {
        if (g_handles[i].occupied && !g_handles[i].released && g_handles[i].handle == handle) {
            return (int)i;
        }
    }
    return -1;
}

static int allocate_slot_locked(void) {
    for (size_t i = 0; i < sizeof(g_handles) / sizeof(g_handles[0]); ++i) {
        if (!g_handles[i].occupied) {
            memset(&g_handles[i], 0, sizeof(g_handles[i]));
            g_handles[i].occupied = 1;
            ++g_handle_count;
            return (int)i;
        }
    }
    return -1;
}

static void discard_slot_locked(int index) {
    if (index >= 0 && g_handles[index].occupied) {
        memset(&g_handles[index], 0, sizeof(g_handles[index]));
        --g_handle_count;
    }
}

static int find_raw_fabric_import_locked(CUmemGenericAllocationHandle handle) {
    for (size_t i = 0; i < g_raw_fabric_import_count; ++i) {
        if (g_raw_fabric_imports[i].handle == handle) {
            return (int)i;
        }
    }
    return -1;
}

static int remember_raw_fabric_import(CUmemGenericAllocationHandle handle, const unsigned char* fabric) {
    pthread_mutex_lock(&g_lock);
    if (g_raw_fabric_import_count == g_raw_fabric_import_capacity) {
        size_t next_capacity = g_raw_fabric_import_capacity == 0 ? 64 : g_raw_fabric_import_capacity * 2;
        if (next_capacity < g_raw_fabric_import_capacity || next_capacity > INT_MAX
            || next_capacity > SIZE_MAX / sizeof(*g_raw_fabric_imports)) {
            pthread_mutex_unlock(&g_lock);
            errno = ENOMEM;
            return -1;
        }
        raw_fabric_import* next = realloc(g_raw_fabric_imports, next_capacity * sizeof(*next));
        if (next == NULL) {
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        g_raw_fabric_imports         = next;
        g_raw_fabric_import_capacity = next_capacity;
    }
    raw_fabric_import* imported = &g_raw_fabric_imports[g_raw_fabric_import_count++];
    memset(imported, 0, sizeof(*imported));
    imported->handle = handle;
    memcpy(imported->fabric, fabric, RTP_MC_FABRIC_HANDLE_BYTES);
    pthread_mutex_unlock(&g_lock);
    return 0;
}

static int mark_raw_fabric_registered(CUmemGenericAllocationHandle handle) {
    pthread_mutex_lock(&g_lock);
    int index = find_raw_fabric_import_locked(handle);
    if (index >= 0) {
        g_raw_fabric_imports[index].keeper_registered = 1;
    }
    pthread_mutex_unlock(&g_lock);
    return index >= 0 ? 0 : -1;
}

static int get_raw_fabric_import(CUmemGenericAllocationHandle handle, unsigned char* fabric, int* keeper_registered) {
    pthread_mutex_lock(&g_lock);
    int index = find_raw_fabric_import_locked(handle);
    if (index >= 0) {
        memcpy(fabric, g_raw_fabric_imports[index].fabric, RTP_MC_FABRIC_HANDLE_BYTES);
        *keeper_registered = g_raw_fabric_imports[index].keeper_registered;
    }
    pthread_mutex_unlock(&g_lock);
    return index >= 0 ? 0 : -1;
}

static void discard_raw_fabric_import_locked(CUmemGenericAllocationHandle handle) {
    int index = find_raw_fabric_import_locked(handle);
    if (index >= 0) {
        g_raw_fabric_imports[index] = g_raw_fabric_imports[g_raw_fabric_import_count - 1];
        --g_raw_fabric_import_count;
    }
}

static CUmemGenericAllocationHandle keeper_import_request(const rtp_mc_request* request,
                                                          const unsigned char*  trailing_fabric,
                                                          rtp_mc_token*         resolved_token) {
    memset(resolved_token, 0, sizeof(*resolved_token));
    if (ensure_peer_contexts() != CUDA_SUCCESS) {
        return 0;
    }
    const int want_fabric = (request->properties.handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0;
    if (want_fabric ? !validate_fabric_team_contract(&request->properties) :
                      (request->properties.num_devices != (uint32_t)g_peer_device_count)) {
        log_message(1,
                    "multicast team contract rejected requested_devices=%u visible_devices=%d fabric=%d",
                    request->properties.num_devices,
                    g_peer_device_count,
                    want_fabric);
        return 0;
    }
    if (real_import == NULL) {
        real_import = (import_fn)driver_symbol("cuMemImportFromShareableHandle");
    }
    if (real_import == NULL) {
        log_message(1, "real cuMemImportFromShareableHandle is unavailable");
        return 0;
    }
    rtp_mc_response response;
    int             fd = keeper_request(request, trailing_fabric, &response);
    if (fd < 0) {
        return 0;
    }
    int is_unknown_import = request->opcode == RTP_MC_OP_IMPORT_ADD && request->properties.size == RTP_MC_UNKNOWN_SIZE;
    int identity_ok       = response.requested_size != 0 && response.served_size >= response.requested_size
                      && response.num_devices == request->properties.num_devices
                      && response.flags == request->properties.flags
                      && response.local_device_count == (uint32_t)g_peer_device_count && response.object_id != 0
                      && (response.holder_instance_hi != 0 || response.holder_instance_lo != 0);
    if (is_unknown_import) {
        identity_ok =
            identity_ok && (response.handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0
            && ((response.handle_types ^ request->properties.handle_types) & ~RTP_MC_HANDLE_TYPE_POSIX_FD) == 0;
    } else {
        identity_ok = identity_ok && response.requested_size == request->properties.size
                      && response.handle_types == request->properties.handle_types;
    }
    if (request->opcode == RTP_MC_OP_FETCH) {
        identity_ok = identity_ok && response.holder_instance_hi == request->holder_instance_hi
                      && response.holder_instance_lo == request->holder_instance_lo
                      && response.object_id == request->object_id;
    }
    if (!identity_ok) {
        close(fd);
        log_message(1, "keeper response identity/property mismatch opcode=%u", request->opcode);
        return 0;
    }
    memset(resolved_token, 0, sizeof(*resolved_token));
    memcpy(resolved_token->magic, RTP_MC_TOKEN_MAGIC, sizeof(resolved_token->magic));
    resolved_token->version                 = RTP_MC_PROTOCOL_VERSION;
    resolved_token->token_size              = sizeof(*resolved_token);
    resolved_token->holder_instance_hi      = response.holder_instance_hi;
    resolved_token->holder_instance_lo      = response.holder_instance_lo;
    resolved_token->object_id               = response.object_id;
    resolved_token->properties.size         = response.requested_size;
    resolved_token->properties.num_devices  = response.num_devices;
    resolved_token->properties.handle_types = response.handle_types;
    resolved_token->properties.flags        = response.flags;
    CUmemGenericAllocationHandle handle     = 0;
    CUresult result = real_import(&handle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    close(fd);
    if (result != CUDA_SUCCESS) {
        log_message(1, "keeper multicast import failed result=%d", result);
        return 0;
    }
    log_message(0,
                "keeper import handle=0x%llx object=%llu requested=%llu served=%llu",
                (unsigned long long)handle,
                (unsigned long long)response.object_id,
                (unsigned long long)response.requested_size,
                (unsigned long long)response.served_size);
    return handle;
}

// Retrieve the real 64-byte CUDA fabric handle for a locally-created FABRIC
// multicast object so torch can broadcast it over the c10d store to peer nodes.
// The handle arrives from the holder in a sealed memfd over SCM_RIGHTS (the
// response struct is unchanged); we copy the 64 bytes into torch's shareable
// handle buffer. Used only on the creating rank; peers import via IMPORT_ADD.
static CUresult keeper_export_fabric(const rtp_mc_token* token, void* shareable_handle) {
    rtp_mc_request  request = request_from_token(RTP_MC_OP_FETCH_FABRIC, token);
    rtp_mc_response response;
    int             fd = keeper_request(&request, NULL, &response);
    if (fd < 0) {
        log_message(1, "keeper FETCH_FABRIC failed object=%llu", (unsigned long long)token->object_id);
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsigned char fabric[RTP_MC_FABRIC_HANDLE_BYTES];
    ssize_t       n = pread(fd, fabric, sizeof(fabric), 0);
    close(fd);
    if (n != (ssize_t)sizeof(fabric)) {
        log_message(1, "keeper FETCH_FABRIC short read n=%zd object=%llu", n, (unsigned long long)token->object_id);
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy(shareable_handle, fabric, sizeof(fabric));
    log_message(0, "exported real fabric handle object=%llu", (unsigned long long)token->object_id);
    return CUDA_SUCCESS;
}

static int remember_imported_handle(CUmemGenericAllocationHandle handle, const rtp_mc_token* token) {
    pthread_mutex_lock(&g_lock);
    int index = allocate_slot_locked();
    if (index >= 0) {
        g_handles[index].handle = handle;
        g_handles[index].token  = *token;
    }
    pthread_mutex_unlock(&g_lock);
    if (index < 0) {
        log_message(1, "keeper handle table is full");
    }
    return index;
}

static int reserve_local_create(const rtp_mc_token* property_token, rtp_mc_request* request) {
    pthread_mutex_lock(&g_lock);
    int released_match = -1;
    for (size_t i = 0; i < sizeof(g_handles) / sizeof(g_handles[0]); ++i) {
        keeper_handle* entry = &g_handles[i];
        if (!entry->occupied || !entry->created_locally || !same_token_properties(&entry->token, property_token)) {
            continue;
        }
        if (entry->pending || !entry->released) {
            pthread_mutex_unlock(&g_lock);
            log_message(1,
                        "multiple active multicast objects with identical properties are unsafe; "
                        "explicit communicator identity is unavailable");
            return -1;
        }
        if (released_match >= 0) {
            pthread_mutex_unlock(&g_lock);
            log_message(1, "ambiguous multicast rebuild identity; refusing ordinal inference");
            return -1;
        }
        released_match = (int)i;
    }
    int index = released_match;
    if (index < 0) {
        index = allocate_slot_locked();
        if (index >= 0) {
            g_handles[index].token           = *property_token;
            g_handles[index].created_locally = 1;
            g_handles[index].released        = 1;
        }
    }
    if (index >= 0) {
        g_handles[index].pending = 1;
        *request =
            request_from_token(released_match >= 0 ? RTP_MC_OP_FETCH : RTP_MC_OP_CREATE, &g_handles[index].token);
    }
    pthread_mutex_unlock(&g_lock);
    return index;
}

static void finish_local_create(int index, CUmemGenericAllocationHandle handle, const rtp_mc_token* token) {
    pthread_mutex_lock(&g_lock);
    g_handles[index].handle   = handle;
    g_handles[index].token    = *token;
    g_handles[index].released = 0;
    g_handles[index].pending  = 0;
    pthread_mutex_unlock(&g_lock);
}

static void abort_local_create(int index, int was_create) {
    pthread_mutex_lock(&g_lock);
    if (was_create) {
        discard_slot_locked(index);
    } else {
        g_handles[index].pending = 0;
    }
    pthread_mutex_unlock(&g_lock);
}

static CUresult hook_mccreate(CUmemGenericAllocationHandle* handle, const CUmulticastObjectProp* properties) {
    if (!keeper_enabled()) {
        if (real_mccreate == NULL) {
            real_mccreate = (mccreate_fn)driver_symbol("cuMulticastCreate");
        }
        return real_mccreate == NULL ? CUDA_ERROR_INVALID_VALUE : real_mccreate(handle, properties);
    }
    if (handle == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    rtp_mc_token property_token;
    if (token_from_properties(properties, &property_token) != 0) {
        log_message(1, "unsupported multicast properties");
        return CUDA_ERROR_INVALID_VALUE;
    }
    rtp_mc_request request;
    int            index = reserve_local_create(&property_token, &request);
    if (index < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    int                          was_create = request.opcode == RTP_MC_OP_CREATE;
    rtp_mc_token                 resolved_token;
    CUmemGenericAllocationHandle keeper_handle = keeper_import_request(&request, NULL, &resolved_token);
    if (keeper_handle == 0) {
        if (was_create && valid_token(&resolved_token)) {
            keeper_send_release(&resolved_token);
        }
        abort_local_create(index, was_create);
        return CUDA_ERROR_INVALID_VALUE;
    }
    finish_local_create(index, keeper_handle, &resolved_token);
    *handle = keeper_handle;
    return CUDA_SUCCESS;
}

static int device_is_visible(CUdevice device) {
    device_get_count_fn get_count  = (device_get_count_fn)driver_symbol("cuDeviceGetCount");
    device_get_fn       get_device = (device_get_fn)driver_symbol("cuDeviceGet");
    int                 count      = 0;
    if (get_count == NULL || get_device == NULL || get_count(&count) != CUDA_SUCCESS) {
        return 0;
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice candidate = -1;
        if (get_device(&candidate, ordinal) != CUDA_SUCCESS) {
            return 0;
        }
        if (candidate == device) {
            return 1;
        }
    }
    return 0;
}

static CUresult hook_mcadd(CUmemGenericAllocationHandle handle, CUdevice device) {
    unsigned char raw_fabric[RTP_MC_FABRIC_HANDLE_BYTES];
    int           raw_registered = 0;
    int           raw_import     = keeper_enabled() && get_raw_fabric_import(handle, raw_fabric, &raw_registered) == 0;
    if (raw_import) {
        if (!device_is_visible(device)) {
            log_message(1, "refusing non-visible multicast device=%d", device);
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (raw_registered) {
            return CUDA_SUCCESS;
        }

        uint64_t team_size = 0;
        if (!parse_env_u64("RTP_LLM_MC_FABRIC_TEAM_SIZE", &team_size) || team_size > UINT32_MAX) {
            log_message(1, "missing or invalid FABRIC team contract during AddDevice");
            return CUDA_ERROR_INVALID_VALUE;
        }
        rtp_mc_token property_token;
        memset(&property_token, 0, sizeof(property_token));
        memcpy(property_token.magic, RTP_MC_TOKEN_MAGIC, sizeof(property_token.magic));
        property_token.version                 = RTP_MC_PROTOCOL_VERSION;
        property_token.token_size              = sizeof(property_token);
        property_token.properties.size         = RTP_MC_UNKNOWN_SIZE;
        property_token.properties.num_devices  = (uint32_t)team_size;
        property_token.properties.handle_types = RTP_MC_HANDLE_TYPE_FABRIC;

        rtp_mc_request               request = request_from_token(RTP_MC_OP_IMPORT_ADD, &property_token);
        rtp_mc_token                 resolved_token;
        CUmemGenericAllocationHandle temporary = keeper_import_request(&request, raw_fabric, &resolved_token);
        if (valid_token(&resolved_token) && remember_peer_ref(&resolved_token) < 0) {
            log_message(1, "cannot retain promoted FABRIC owner reference");
            if (temporary != 0) {
                if (real_memrelease == NULL) {
                    real_memrelease = (memrelease_fn)driver_symbol("cuMemRelease");
                }
                if (real_memrelease != NULL) {
                    (void)real_memrelease(temporary);
                }
            }
            // AddDevice may already be permanent in the holder. Its owner ref
            // is intentionally left for generation reclamation rather than
            // risking a second AddDevice on retry.
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (temporary == 0) {
            // The holder has already made AddDevice permanent on the underlying
            // object. Keep its owner reference for retry/teardown; releasing the
            // last ref here would make a retry attempt AddDevice a second time.
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (real_memrelease == NULL) {
            real_memrelease = (memrelease_fn)driver_symbol("cuMemRelease");
        }
        if (real_memrelease == NULL) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        CUresult release_result = real_memrelease(temporary);
        if (release_result != CUDA_SUCCESS || mark_raw_fabric_registered(handle) != 0) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        log_message(0,
                    "promoted raw FABRIC multicast handle=0x%llx object=%llu",
                    (unsigned long long)handle,
                    (unsigned long long)resolved_token.object_id);
        return CUDA_SUCCESS;
    }

    pthread_mutex_lock(&g_lock);
    int keeper_owned = find_keeper_handle(handle) >= 0;
    pthread_mutex_unlock(&g_lock);
    if (keeper_enabled() && keeper_owned) {
        if (!device_is_visible(device)) {
            log_message(1, "refusing non-visible multicast device=%d", device);
            return CUDA_ERROR_INVALID_VALUE;
        }
        log_message(
            0, "cuMulticastAddDevice keeper handle=0x%llx device=%d -> no-op", (unsigned long long)handle, device);
        return CUDA_SUCCESS;
    }
    if (real_mcadd == NULL) {
        real_mcadd = (mcadd_fn)driver_symbol("cuMulticastAddDevice");
    }
    return real_mcadd == NULL ? CUDA_ERROR_INVALID_VALUE : real_mcadd(handle, device);
}

static CUresult
hook_export(void* shareable_handle, CUmemGenericAllocationHandle handle, int type, unsigned long long flags) {
    rtp_mc_token token;
    memset(&token, 0, sizeof(token));
    pthread_mutex_lock(&g_lock);
    int index        = find_keeper_handle(handle);
    int keeper_owned = index >= 0;
    if (keeper_owned) {
        token = g_handles[index].token;
    }
    pthread_mutex_unlock(&g_lock);
    log_message(0,
                "cuMemExportToShareableHandle handle=0x%llx type=0x%x keeper_owned=%d object=%llu",
                (unsigned long long)handle,
                type,
                keeper_owned,
                (unsigned long long)token.object_id);
    if (keeper_enabled() && keeper_owned) {
        if (shareable_handle == NULL || flags != 0
            || (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && type != CU_MEM_HANDLE_TYPE_FABRIC)
            || (token.properties.handle_types & (uint32_t)type) == 0) {
            log_message(1, "unsupported keeper export type=0x%x flags=%llu", type, flags);
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
            int fd = memfd_create("rtp-llm-multicast-token", MFD_CLOEXEC | MFD_ALLOW_SEALING);
            if (fd < 0) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            if (ftruncate(fd, sizeof(token)) != 0 || pwrite(fd, &token, sizeof(token), 0) != (ssize_t)sizeof(token)
                || fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
                close(fd);
                return CUDA_ERROR_INVALID_VALUE;
            }
            *(int*)shareable_handle = fd;
            log_message(0, "exported POSIX keeper token fd=%d object=%llu", fd, (unsigned long long)token.object_id);
            return CUDA_SUCCESS;
        }
        // FABRIC: hand back the REAL 64-byte CUDA fabric handle (fetched from the
        // local holder), not the identity token, so torch can broadcast it over
        // the c10d store to peer nodes for cross-machine (MNNVL) import.
        return keeper_export_fabric(&token, shareable_handle);
    }
    if (real_export == NULL) {
        real_export = (export_fn)driver_symbol("cuMemExportToShareableHandle");
    }
    return real_export == NULL ? CUDA_ERROR_INVALID_VALUE : real_export(shareable_handle, handle, type, flags);
}

static CUresult import_keeper_token(CUmemGenericAllocationHandle* handle, const rtp_mc_token* token, int type) {
    if (handle == NULL || !valid_token(token) || (token->properties.handle_types & (uint32_t)type) == 0) {
        log_message(1, "invalid keeper token or handle type=0x%x", type);
        return CUDA_ERROR_INVALID_VALUE;
    }
    rtp_mc_request               request = request_from_token(RTP_MC_OP_FETCH, token);
    rtp_mc_token                 resolved_token;
    CUmemGenericAllocationHandle keeper_handle = keeper_import_request(&request, NULL, &resolved_token);
    if (keeper_handle == 0 || !same_token_properties(token, &resolved_token)
        || token->holder_instance_hi != resolved_token.holder_instance_hi
        || token->holder_instance_lo != resolved_token.holder_instance_lo
        || token->object_id != resolved_token.object_id) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (remember_imported_handle(keeper_handle, &resolved_token) < 0) {
        if (real_memrelease == NULL) {
            real_memrelease = (memrelease_fn)driver_symbol("cuMemRelease");
        }
        if (real_memrelease != NULL) {
            (void)real_memrelease(keeper_handle);
        }
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = keeper_handle;
    return CUDA_SUCCESS;
}

static CUresult hook_import(CUmemGenericAllocationHandle* handle, void* data, int type) {
    if (keeper_enabled() && type == CU_MEM_HANDLE_TYPE_FABRIC && data != NULL) {
        if (memcmp(data, RTP_MC_TOKEN_MAGIC, 8) == 0) {
            rtp_mc_token token;
            memcpy(&token, data, sizeof(token));
            return import_keeper_token(handle, &token, type);
        }
        // Ordinary allocations and multicast objects use the same opaque FABRIC
        // import ABI. Import every raw handle normally; only a later
        // cuMulticastAddDevice proves that this handle needs holder promotion.
        if (real_import == NULL) {
            real_import = (import_fn)driver_symbol("cuMemImportFromShareableHandle");
        }
        CUresult result = real_import == NULL ? CUDA_ERROR_INVALID_VALUE : real_import(handle, data, type);
        if (result == CUDA_SUCCESS && remember_raw_fabric_import(*handle, (const unsigned char*)data) != 0) {
            int saved_errno = errno;
            if (real_memrelease == NULL) {
                real_memrelease = (memrelease_fn)driver_symbol("cuMemRelease");
            }
            if (real_memrelease != NULL) {
                (void)real_memrelease(*handle);
            }
            log_message(1, "cannot track raw FABRIC import: %s", strerror(saved_errno));
            return CUDA_ERROR_INVALID_VALUE;
        }
        return result;
    }
    if (keeper_enabled() && type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int          fd = (int)(intptr_t)data;
        rtp_mc_token token;
        struct stat  info;
        memset(&token, 0, sizeof(token));
        char    prefix[8];
        ssize_t prefix_size = fd < 0 ? -1 : pread(fd, prefix, sizeof(prefix), 0);
        if (prefix_size == (ssize_t)sizeof(prefix) && memcmp(prefix, RTP_MC_TOKEN_MAGIC, sizeof(prefix)) == 0) {
            int seals          = fcntl(fd, F_GET_SEALS);
            int required_seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
            if (fstat(fd, &info) != 0 || info.st_size != (off_t)sizeof(token) || seals < 0
                || (seals & required_seals) != required_seals
                || pread(fd, &token, sizeof(token), 0) != (ssize_t)sizeof(token)) {
                log_message(1, "malformed POSIX keeper token fd=%d", fd);
                return CUDA_ERROR_INVALID_VALUE;
            }
            log_message(0, "imported POSIX keeper token fd=%d object=%llu", fd, (unsigned long long)token.object_id);
            return import_keeper_token(handle, &token, type);
        }
        CUresult result = ensure_peer_contexts();
        if (result != CUDA_SUCCESS) {
            return result;
        }
    }
    if (real_import == NULL) {
        real_import = (import_fn)driver_symbol("cuMemImportFromShareableHandle");
    }
    return real_import == NULL ? CUDA_ERROR_INVALID_VALUE : real_import(handle, data, type);
}

static CUresult hook_memrelease(CUmemGenericAllocationHandle handle) {
    if (real_memrelease == NULL) {
        real_memrelease = (memrelease_fn)driver_symbol("cuMemRelease");
    }
    if (real_memrelease == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult result = real_memrelease(handle);
    if (result == CUDA_SUCCESS) {
        pthread_mutex_lock(&g_lock);
        int index = g_handle_count == 0 ? -1 : find_keeper_handle(handle);
        if (index >= 0) {
            if (g_handles[index].created_locally) {
                g_handles[index].handle   = 0;
                g_handles[index].released = 1;
            } else {
                discard_slot_locked(index);
            }
        }
        if (g_raw_fabric_import_count != 0) {
            discard_raw_fabric_import_locked(handle);
        }
        pthread_mutex_unlock(&g_lock);
    }
    return result;
}

// Best-effort RELEASE for an object reference at process teardown. This drops
// the current process incarnation from the holder's owner set. It deliberately
// does NOT run during a CUDA checkpoint (the process is frozen, not exited) nor
// on a plain cuMemRelease (which retains identity for the post-restore rebuild),
// so an in-flight object is never freed out from under a pending rebuild.
static void keeper_send_release(const rtp_mc_token* token) {
    if (!valid_token(token)) {
        return;
    }
    char path[sizeof(((struct sockaddr_un*)0)->sun_path)];
    if (keeper_socket_path(path, sizeof(path)) == NULL) {
        return;
    }
    // Keep teardown snappy: the holder is normally still alive at shutdown, so a
    // short deadline is enough and avoids blocking exit if it already vanished.
    int timeout_ms = timeout_from_env("RTP_LLM_MC_RELEASE_TIMEOUT_MS", 1000);
    int socket_fd  = connect_to_keeper(path, timeout_ms);
    if (socket_fd < 0) {
        return;
    }
    ensure_owner_identity();
    rtp_mc_request     base = request_from_token(RTP_MC_OP_RELEASE, token);
    rtp_mc_request_ext extended;
    memset(&extended, 0, sizeof(extended));
    extended.base             = base;
    extended.base.struct_size = (uint32_t)sizeof(extended);
    extended.owner_id         = g_owner_id;
    extended.owner_generation = g_owner_generation;
    if (send(socket_fd, &extended, sizeof(extended), MSG_NOSIGNAL) == (ssize_t)sizeof(extended)) {
        rtp_mc_response response;
        (void)recv(socket_fd, &response, sizeof(response), 0);  // drain reply, best-effort
        log_message(0,
                    "keeper release sent object=%llu owner=%llu",
                    (unsigned long long)token->object_id,
                    (unsigned long long)g_owner_id);
    }
    close(socket_fd);
}

__attribute__((destructor)) static void shim_release_owned_objects(void) {
    if (!keeper_enabled()) {
        return;
    }
    // Snapshot and deduplicate locally-created and raw-FABRIC peer references
    // under the lock, then release outside it so socket I/O never holds g_lock.
    rtp_mc_token owned[sizeof(g_handles) / sizeof(g_handles[0]) + sizeof(g_peer_refs) / sizeof(g_peer_refs[0])];
    size_t       owned_count = 0;
    pthread_mutex_lock(&g_lock);
    for (size_t i = 0; i < sizeof(g_handles) / sizeof(g_handles[0]); ++i) {
        keeper_handle* entry = &g_handles[i];
        if (entry->occupied && entry->created_locally && entry->token.object_id != 0
            && (entry->token.holder_instance_hi != 0 || entry->token.holder_instance_lo != 0)) {
            owned[owned_count++] = entry->token;
        }
    }
    for (size_t i = 0; i < g_peer_ref_count; ++i) {
        int duplicate = 0;
        for (size_t j = 0; j < owned_count; ++j) {
            if (same_token_identity(&owned[j], &g_peer_refs[i])) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            owned[owned_count++] = g_peer_refs[i];
        }
    }
    pthread_mutex_unlock(&g_lock);
    for (size_t i = 0; i < owned_count; ++i) {
        keeper_send_release(&owned[i]);
    }
}

static void* swap_symbol(const char* symbol, void* resolved) {
    if (strcmp(symbol, "cuMemImportFromShareableHandle") == 0) {
        if (resolved != (void*)hook_import) {
            real_import = (import_fn)resolved;
        }
        log_message(0, "hooked %s resolved=%p", symbol, resolved);
        return (void*)hook_import;
    }
    if (strcmp(symbol, "cuMemExportToShareableHandle") == 0) {
        if (resolved != (void*)hook_export) {
            real_export = (export_fn)resolved;
        }
        log_message(0, "hooked %s resolved=%p", symbol, resolved);
        return (void*)hook_export;
    }
    if (strcmp(symbol, "cuMulticastCreate") == 0) {
        if (resolved != (void*)hook_mccreate) {
            real_mccreate = (mccreate_fn)resolved;
        }
        log_message(0, "hooked %s resolved=%p", symbol, resolved);
        return (void*)hook_mccreate;
    }
    if (strcmp(symbol, "cuMulticastAddDevice") == 0) {
        if (resolved != (void*)hook_mcadd) {
            real_mcadd = (mcadd_fn)resolved;
        }
        log_message(0, "hooked %s resolved=%p", symbol, resolved);
        return (void*)hook_mcadd;
    }
    if (strcmp(symbol, "cuMemRelease") == 0) {
        if (resolved != (void*)hook_memrelease) {
            real_memrelease = (memrelease_fn)resolved;
        }
        return (void*)hook_memrelease;
    }
    return NULL;
}

// Export direct symbols as well as replacing CUDA entry-point queries. Some
// NCCL/PyTorch versions link these APIs directly or dlsym them from libcuda.
CUresult cuMulticastCreate(CUmemGenericAllocationHandle* handle, const CUmulticastObjectProp* properties) {
    return hook_mccreate(handle, properties);
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle handle, CUdevice device) {
    return hook_mcadd(handle, device);
}

CUresult cuMemExportToShareableHandle(void*                        shareable_handle,
                                      CUmemGenericAllocationHandle handle,
                                      int                          type,
                                      unsigned long long           flags) {
    return hook_export(shareable_handle, handle, type, flags);
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* data, int type) {
    return hook_import(handle, data, type);
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return hook_memrelease(handle);
}

static CUresult hook_getproc(const char* symbol, void** function, int version, unsigned long long flags, int* status) {
    if (real_getproc == NULL) {
        log_message(1, "real cuGetProcAddress is unavailable");
        return 999;
    }
    g_cuda_version  = version;
    g_cuda_flags    = flags;
    CUresult result = real_getproc(symbol, function, version, flags, status);
    if (result == CUDA_SUCCESS && function != NULL && *function != NULL && symbol != NULL) {
        if (strcmp(symbol, "cuGetProcAddress") == 0 || strcmp(symbol, "cuGetProcAddress_v2") == 0) {
            if (*function != (void*)hook_getproc) {
                real_getproc = (getproc_v2_fn)*function;
            }
            *function = (void*)hook_getproc;
        } else {
            void* hook = swap_symbol(symbol, *function);
            if (hook != NULL) {
                *function = hook;
            }
        }
    }
    return result;
}

int cudaGetDriverEntryPointByVersion(
    const char* symbol, void** function, unsigned int version, unsigned long long flags, int* status) {
    if (real_getentry == NULL) {
        dlsym_fn lookup = get_real_dlsym();
        if (lookup != NULL) {
            real_getentry = (getentry_fn)lookup(RTLD_NEXT, "cudaGetDriverEntryPointByVersion");
        }
        if (real_getentry == NULL) {
            log_message(1, "real cudaGetDriverEntryPointByVersion is unavailable");
            return 999;
        }
    }
    int result = real_getentry(symbol, function, version, flags, status);
    if (result == 0 && function != NULL && *function != NULL && symbol != NULL) {
        void* hook = swap_symbol(symbol, *function);
        if (hook != NULL) {
            *function = hook;
        }
    }
    return result;
}

void* dlsym(void* handle, const char* name) {
    dlsym_fn lookup = get_real_dlsym();
    if (lookup == NULL) {
        return NULL;
    }
    void* resolved = lookup(handle, name);
    if ((strcmp(name, "cuGetProcAddress_v2") == 0 || strcmp(name, "cuGetProcAddress") == 0) && resolved != NULL) {
        real_getproc = (getproc_v2_fn)resolved;
        return (void*)hook_getproc;
    }
    if (resolved != NULL) {
        void* hook = swap_symbol(name, resolved);
        if (hook != NULL) {
            return hook;
        }
    }
    return resolved;
}
