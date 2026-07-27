#define _GNU_SOURCE

#include "rtp_llm/cpp/cuda_checkpoint/multicast_keeper/keeper_protocol.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct owner_ref {
    uint64_t owner_id;
    uint64_t owner_generation;
} owner_ref;

typedef struct held_entry {
    uint64_t                 object_id;
    rtp_mc_object_properties properties;
    uint64_t                 served_size;
    owner_ref*               owners;
    size_t                   owner_count;
    size_t                   owner_capacity;
    int                      fd;
    // Cross-machine (MNNVL) fabric: has_fabric != 0 when this entry corresponds
    // to a FABRIC multicast team and fabric_handle holds the 64-byte shareable
    // handle. On the creator node it is the exported handle; on a peer node it is
    // the handle we imported. It is the dedup key for RTP_MC_OP_IMPORT_ADD (all
    // ranks on a node share one node-local entry per fabric object) and the
    // payload returned by RTP_MC_OP_FETCH_FABRIC.
    int           has_fabric;
    unsigned char fabric_handle[RTP_MC_FABRIC_HANDLE_BYTES];
} held_entry;

static held_entry            g_entries[RTP_MC_MAX_ENTRIES];
static size_t                g_entry_count      = 0;
static uint64_t              g_instance_hi      = 0;
static uint64_t              g_instance_lo      = 0;
static uint64_t              g_next_object_id   = 1;
static uint32_t              g_local_gpu_count  = 0;
static uint32_t              g_fabric_team_size = 0;
static volatile sig_atomic_t g_stopping         = 0;
static volatile sig_atomic_t g_creator_pid      = -1;
static int                   g_listener         = -1;

static void log_message(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stdout, "[multicast-holder pid=%d] ", (int)getpid());
    vfprintf(stdout, format, args);
    fputc('\n', stdout);
    fflush(stdout);
    va_end(args);
}

static void usage(FILE* stream, const char* program) {
    fprintf(stream,
            "Usage:\n"
            "  %s --socket PATH --creator PATH --gpus LIST [options]\n"
            "  %s --check --socket PATH\n\n"
            "Options:\n"
            "  --ready-file PATH          Holder-ready file removed on exit\n"
            "  --socket-mode OCTAL        Unix socket permissions (default 0600)\n"
            "  --client-timeout-ms N      Request/reply I/O timeout (default 1000)\n"
            "  --creator-timeout-ms N     Creator timeout (default 120000)\n"
            "  --fabric-team-size N       Exact global FABRIC team size\n",
            program,
            program);
}

static int validate_socket_path(const char* path) {
    if (path == NULL || path[0] == '\0') {
        fprintf(stderr, "--socket must not be empty\n");
        return -1;
    }
    if (strlen(path) >= sizeof(((struct sockaddr_un*)0)->sun_path)) {
        fprintf(stderr, "socket path is too long: %s\n", path);
        return -1;
    }
    return 0;
}

static int parse_gpu_count(const char* text, uint32_t* count) {
    if (text == NULL || text[0] == '\0') {
        return -1;
    }
    int         values[RTP_MC_MAX_ENTRIES];
    size_t      value_count = 0;
    const char* cursor      = text;
    while (*cursor != '\0') {
        char* end  = NULL;
        errno      = 0;
        long value = strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || value < 0 || value > INT_MAX || value_count == RTP_MC_MAX_ENTRIES) {
            return -1;
        }
        for (size_t i = 0; i < value_count; ++i) {
            if (values[i] == (int)value) {
                return -1;
            }
        }
        values[value_count++] = (int)value;
        if (*end == '\0') {
            break;
        }
        if (*end != ',' || end[1] == '\0') {
            return -1;
        }
        cursor = end + 1;
    }
    *count = (uint32_t)value_count;
    return value_count == 0 ? -1 : 0;
}

static int parse_positive_u32(const char* text, uint32_t* value) {
    char* end            = NULL;
    errno                = 0;
    unsigned long parsed = strtoul(text == NULL ? "" : text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int random_instance(void) {
    uint64_t values[2] = {0, 0};
    size_t   offset    = 0;
    while (offset < sizeof(values)) {
        ssize_t count = getrandom((char*)values + offset, sizeof(values) - offset, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return -1;
        }
        offset += (size_t)count;
    }
    if (values[0] == 0 && values[1] == 0) {
        errno = EIO;
        return -1;
    }
    g_instance_hi = values[0];
    g_instance_lo = values[1];
    return 0;
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

static int same_properties(const rtp_mc_object_properties* left, const rtp_mc_object_properties* right) {
    return left->size == right->size && left->num_devices == right->num_devices
           && left->handle_types == right->handle_types && left->flags == right->flags;
}

static int compatible_import_properties(const rtp_mc_object_properties* request, const rtp_mc_object_properties* held) {
    if (request->size != RTP_MC_UNKNOWN_SIZE) {
        return same_properties(request, held);
    }
    // A peer raw FABRIC import has no size metadata. The opaque handle is the
    // object identity, so adopt the held size while keeping the team contract,
    // flags, and all non-POSIX handle capabilities strict.
    return request->num_devices == held->num_devices && request->flags == held->flags
           && (request->handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0
           && (held->handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0
           && ((request->handle_types ^ held->handle_types) & ~RTP_MC_HANDLE_TYPE_POSIX_FD) == 0;
}

static int send_response(int socket_fd, const rtp_mc_response* response, int passed_fd) {
    struct iovec iov = {.iov_base = (void*)response, .iov_len = sizeof(*response)};
    char         control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof(control));
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov    = &iov;
    header.msg_iovlen = 1;
    if (passed_fd >= 0) {
        header.msg_control    = control;
        header.msg_controllen = sizeof(control);
        struct cmsghdr* cmsg  = CMSG_FIRSTHDR(&header);
        cmsg->cmsg_level      = SOL_SOCKET;
        cmsg->cmsg_type       = SCM_RIGHTS;
        cmsg->cmsg_len        = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(cmsg), &passed_fd, sizeof(passed_fd));
    }
    return sendmsg(socket_fd, &header, MSG_NOSIGNAL) == (ssize_t)sizeof(*response) ? 0 : -1;
}

static int recv_creator_result(int socket_fd, rtp_mc_creator_result* result, int* received_fd) {
    struct iovec iov = {.iov_base = result, .iov_len = sizeof(*result)};
    char         control[CMSG_SPACE(sizeof(int))];
    memset(control, 0, sizeof(control));
    struct msghdr header;
    memset(&header, 0, sizeof(header));
    header.msg_iov        = &iov;
    header.msg_iovlen     = 1;
    header.msg_control    = control;
    header.msg_controllen = sizeof(control);
    *received_fd          = -1;
    ssize_t received      = recvmsg(socket_fd, &header, MSG_CMSG_CLOEXEC);
    if (received != (ssize_t)sizeof(*result) || (header.msg_flags & (MSG_TRUNC | MSG_CTRUNC))) {
        return -1;
    }
    for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&header); cmsg != NULL; cmsg = CMSG_NXTHDR(&header, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS
            && cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
            memcpy(received_fd, CMSG_DATA(cmsg), sizeof(*received_fd));
            break;
        }
    }
    return 0;
}

static int wait_for_fd(int fd, int timeout_ms) {
    int remaining = timeout_ms;
    while (!g_stopping && remaining > 0) {
        int           interval   = remaining < 250 ? remaining : 250;
        struct pollfd descriptor = {.fd = fd, .events = POLLIN};
        int           result     = poll(&descriptor, 1, interval);
        if (result > 0) {
            return (descriptor.revents & POLLIN) != 0 ? 0 : -1;
        }
        if (result < 0 && errno != EINTR) {
            return -1;
        }
        remaining -= interval;
    }
    errno = g_stopping ? EINTR : ETIMEDOUT;
    return -1;
}

static int wait_for_child(pid_t pid, int timeout_ms, int* status) {
    int remaining = timeout_ms;
    while (remaining >= 0) {
        pid_t result = waitpid(pid, status, WNOHANG);
        if (result == pid) {
            return 0;
        }
        if (result < 0 && errno != EINTR) {
            return -1;
        }
        if (remaining == 0) {
            break;
        }
        int             interval = remaining < 20 ? remaining : 20;
        struct timespec delay    = {.tv_sec = 0, .tv_nsec = interval * 1000000L};
        nanosleep(&delay, NULL);
        remaining -= interval;
    }
    errno = ETIMEDOUT;
    return -1;
}

static int terminate_creator(pid_t pid, int* status) {
    (void)kill(pid, SIGTERM);
    if (wait_for_child(pid, 250, status) == 0) {
        return 0;
    }
    (void)kill(pid, SIGKILL);
    return wait_for_child(pid, 1000, status);
}

static held_entry* find_entry(uint64_t object_id) {
    for (size_t i = 0; i < g_entry_count; ++i) {
        if (g_entries[i].object_id == object_id) {
            return &g_entries[i];
        }
    }
    return NULL;
}

static int find_owner_ref(const held_entry* entry, uint64_t owner_id, uint64_t owner_generation) {
    for (size_t i = 0; i < entry->owner_count; ++i) {
        if (entry->owners[i].owner_id == owner_id && entry->owners[i].owner_generation == owner_generation) {
            return (int)i;
        }
    }
    return -1;
}

// Registering the same process incarnation again is a rebuild, not another
// lifetime reference. This keeps repeated IMPORT_ADD calls idempotent.
static int add_owner_ref(held_entry* entry, uint64_t owner_id, uint64_t owner_generation) {
    if (find_owner_ref(entry, owner_id, owner_generation) >= 0) {
        return 0;
    }
    if (entry->owner_count == entry->owner_capacity) {
        size_t next_capacity = entry->owner_capacity == 0 ? 4 : entry->owner_capacity * 2;
        if (next_capacity > SIZE_MAX / sizeof(*entry->owners)) {
            errno = ENOMEM;
            return -1;
        }
        owner_ref* next = realloc(entry->owners, next_capacity * sizeof(*entry->owners));
        if (next == NULL) {
            return -1;
        }
        entry->owners         = next;
        entry->owner_capacity = next_capacity;
    }
    entry->owners[entry->owner_count++] = (owner_ref){owner_id, owner_generation};
    return 1;
}

static void remove_owner_ref_at(held_entry* entry, size_t index) {
    entry->owners[index] = entry->owners[entry->owner_count - 1];
    --entry->owner_count;
}

// Close the held fd and drop the entry, freeing its slot. Uses swap-with-last so
// the array stays dense; object ids are monotonic and never reused, so a later
// FETCH/RELEASE for a removed id fails closed with UNKNOWN_OBJECT.
static void remove_entry_at(size_t index) {
    if (index >= g_entry_count) {
        return;
    }
    close(g_entries[index].fd);
    free(g_entries[index].owners);
    if (index != g_entry_count - 1) {
        g_entries[index] = g_entries[g_entry_count - 1];
    }
    memset(&g_entries[g_entry_count - 1], 0, sizeof(g_entries[g_entry_count - 1]));
    --g_entry_count;
}

static size_t drop_owner_ref(held_entry* entry, uint64_t owner_id, uint64_t owner_generation) {
    int owner_index = find_owner_ref(entry, owner_id, owner_generation);
    if (owner_index < 0) {
        return SIZE_MAX;
    }
    remove_owner_ref_at(entry, (size_t)owner_index);
    size_t remaining = entry->owner_count;
    if (remaining == 0) {
        remove_entry_at((size_t)(entry - g_entries));
    }
    return remaining;
}

// Remove stale references left by an earlier incarnation of this owner. Other
// owners keep the shared object alive; a reference-free object is closed.
static size_t reclaim_stale_owner(uint64_t owner_id, uint64_t owner_generation) {
    size_t reclaimed = 0;
    if (owner_id == 0) {
        return 0;
    }
    for (size_t i = 0; i < g_entry_count;) {
        held_entry* entry = &g_entries[i];
        for (size_t owner_index = 0; owner_index < entry->owner_count;) {
            owner_ref stale = entry->owners[owner_index];
            if (stale.owner_id == owner_id && stale.owner_generation != owner_generation) {
                log_message("reclaim_stale owner=%llu stale_gen=%llu new_gen=%llu object=%llu",
                            (unsigned long long)owner_id,
                            (unsigned long long)stale.owner_generation,
                            (unsigned long long)owner_generation,
                            (unsigned long long)entry->object_id);
                remove_owner_ref_at(entry, owner_index);
                ++reclaimed;
            } else {
                ++owner_index;
            }
        }
        if (entry->owner_count == 0) {
            remove_entry_at(i);
        } else {
            ++i;
        }
    }
    return reclaimed;
}

// Create or import a multicast object via a short-lived child.
//   import_fabric == NULL: CREATE mode — the child calls cuMulticastCreate and
//     deposits a node-local POSIX fd (plus, for FABRIC teams, the 64-byte fabric
//     handle inline in the result). properties->num_devices is the whole team.
//   import_fabric != NULL: IMPORT_ADD mode — the child imports that 64-byte
//     fabric handle, AddDevice's this node's LOCAL devices, and re-exports a
//     node-local POSIX fd. The child receives the handle over a pipe.
// In both cases the resulting node-local POSIX fd is what local ranks import.
static held_entry* create_entry(const char*                     creator,
                                const char*                     gpus,
                                const rtp_mc_object_properties* properties,
                                uint64_t                        owner_id,
                                uint64_t                        owner_generation,
                                int                             timeout_ms,
                                const unsigned char*            import_fabric) {
    if (g_entry_count == RTP_MC_MAX_ENTRIES) {
        errno = ENOSPC;
        return NULL;
    }
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair) != 0) {
        return NULL;
    }
    int flags = fcntl(pair[1], F_GETFD);
    if (flags < 0 || fcntl(pair[1], F_SETFD, flags & ~FD_CLOEXEC) != 0) {
        int saved_errno = errno;
        close(pair[0]);
        close(pair[1]);
        errno = saved_errno;
        return NULL;
    }
    // For IMPORT_ADD, hand the 64-byte fabric handle to the child over a pipe.
    int fabric_pipe[2] = {-1, -1};
    if (import_fabric != NULL) {
        if (pipe2(fabric_pipe, O_CLOEXEC) != 0) {
            int saved_errno = errno;
            close(pair[0]);
            close(pair[1]);
            errno = saved_errno;
            return NULL;
        }
        int rflags = fcntl(fabric_pipe[0], F_GETFD);
        if (rflags < 0 || fcntl(fabric_pipe[0], F_SETFD, rflags & ~FD_CLOEXEC) != 0) {
            int saved_errno = errno;
            close(pair[0]);
            close(pair[1]);
            close(fabric_pipe[0]);
            close(fabric_pipe[1]);
            errno = saved_errno;
            return NULL;
        }
    }
    pid_t child = fork();
    if (child < 0) {
        int saved_errno = errno;
        close(pair[0]);
        close(pair[1]);
        if (fabric_pipe[0] >= 0) {
            close(fabric_pipe[0]);
            close(fabric_pipe[1]);
        }
        errno = saved_errno;
        return NULL;
    }
    if (child == 0) {
        close(pair[0]);
        if (fabric_pipe[1] >= 0) {
            close(fabric_pipe[1]);
        }
        char requested[32];
        char deposit_fd[32];
        char num_devices[32];
        char handle_types[32];
        char property_flags[32];
        char import_fd[32];
        snprintf(requested, sizeof(requested), "%llu", (unsigned long long)properties->size);
        snprintf(deposit_fd, sizeof(deposit_fd), "%d", pair[1]);
        snprintf(num_devices, sizeof(num_devices), "%u", properties->num_devices);
        snprintf(handle_types, sizeof(handle_types), "%u", properties->handle_types);
        snprintf(property_flags, sizeof(property_flags), "%llu", (unsigned long long)properties->flags);
        if (import_fabric != NULL) {
            snprintf(import_fd, sizeof(import_fd), "%d", fabric_pipe[0]);
            execl(creator,
                  creator,
                  "--gpus",
                  gpus,
                  "--size",
                  requested,
                  "--num-devices",
                  num_devices,
                  "--handle-types",
                  handle_types,
                  "--flags",
                  property_flags,
                  "--deposit-fd",
                  deposit_fd,
                  "--import-fabric-fd",
                  import_fd,
                  (char*)NULL);
        } else {
            execl(creator,
                  creator,
                  "--gpus",
                  gpus,
                  "--size",
                  requested,
                  "--num-devices",
                  num_devices,
                  "--handle-types",
                  handle_types,
                  "--flags",
                  property_flags,
                  "--deposit-fd",
                  deposit_fd,
                  (char*)NULL);
        }
        perror("exec(creator)");
        _exit(127);
    }
    close(pair[1]);
    if (fabric_pipe[0] >= 0) {
        close(fabric_pipe[0]);
        // Deliver the 64-byte handle, then close so the child sees EOF.
        ssize_t written = write(fabric_pipe[1], import_fabric, RTP_MC_FABRIC_HANDLE_BYTES);
        close(fabric_pipe[1]);
        if (written != (ssize_t)RTP_MC_FABRIC_HANDLE_BYTES) {
            close(pair[0]);
            (void)terminate_creator(child, NULL);
            errno = EIO;
            return NULL;
        }
    }
    g_creator_pid = child;
    log_message("creator_start pid=%d requested=%llu devices=%u handles=0x%x mode=%s",
                (int)child,
                (unsigned long long)properties->size,
                properties->num_devices,
                properties->handle_types,
                import_fabric != NULL ? "import" : "create");

    rtp_mc_creator_result result;
    memset(&result, 0, sizeof(result));
    int multicast_fd = -1;
    int receive_ok = wait_for_fd(pair[0], timeout_ms) == 0 && recv_creator_result(pair[0], &result, &multicast_fd) == 0;
    close(pair[0]);

    int child_status = 0;
    int child_ok     = 0;
    if (receive_ok) {
        child_ok = wait_for_child(child, 1000, &child_status) == 0;
    }
    if (!receive_ok || !child_ok) {
        child_ok = terminate_creator(child, &child_status) == 0;
    }
    g_creator_pid = -1;
    if (!receive_ok || !child_ok || result.magic != RTP_MC_CREATOR_MAGIC || result.status != 0
        || result.requested_size != properties->size || result.served_size < properties->size || multicast_fd < 0
        || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
        if (multicast_fd >= 0) {
            close(multicast_fd);
        }
        errno = EIO;
        return NULL;
    }

    held_entry* entry = &g_entries[g_entry_count++];
    memset(entry, 0, sizeof(*entry));
    entry->object_id = g_next_object_id++;
    if (entry->object_id == 0) {
        entry->object_id = g_next_object_id++;
    }
    entry->properties  = *properties;
    entry->served_size = result.served_size;
    entry->fd          = multicast_fd;
    if (add_owner_ref(entry, owner_id, owner_generation) < 0) {
        remove_entry_at((size_t)(entry - g_entries));
        return NULL;
    }
    if (import_fabric != NULL) {
        // Peer-node import: remember the imported handle so co-located ranks
        // dedup onto this one entry instead of each re-importing.
        entry->has_fabric = 1;
        memcpy(entry->fabric_handle, import_fabric, RTP_MC_FABRIC_HANDLE_BYTES);
    } else if (result.flags & RTP_MC_CREATOR_FLAG_FABRIC_VALID) {
        // Creator node: remember the exported handle for FETCH_FABRIC and for
        // dedup of the creator node's own IMPORT_ADD (torch also broadcasts the
        // handle back to the creating node's ranks).
        entry->has_fabric = 1;
        memcpy(entry->fabric_handle, result.fabric_handle, RTP_MC_FABRIC_HANDLE_BYTES);
    }
    log_message(
        "creator_done object=%llu requested=%llu served=%llu owner=%llu gen=%llu fabric=%d refs=%zu entries=%zu",
        (unsigned long long)entry->object_id,
        (unsigned long long)entry->properties.size,
        (unsigned long long)entry->served_size,
        (unsigned long long)owner_id,
        (unsigned long long)owner_generation,
        entry->has_fabric,
        entry->owner_count,
        g_entry_count);
    return entry;
}

// Locate a held FABRIC entry by its 64-byte handle (the cross-node dedup key).
static held_entry* find_entry_by_fabric(const unsigned char* fabric_handle) {
    for (size_t i = 0; i < g_entry_count; ++i) {
        if (g_entries[i].has_fabric
            && memcmp(g_entries[i].fabric_handle, fabric_handle, RTP_MC_FABRIC_HANDLE_BYTES) == 0) {
            return &g_entries[i];
        }
    }
    return NULL;
}

// Wrap a 64-byte fabric handle in a sealed, read-only memfd so FETCH_FABRIC can
// hand it back over SCM_RIGHTS without growing the fixed-size response struct.
static int make_fabric_memfd(const unsigned char* fabric_handle) {
    int fd = memfd_create("rtp_mc_fabric", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) {
        return -1;
    }
    if (write(fd, fabric_handle, RTP_MC_FABRIC_HANDLE_BYTES) != (ssize_t)RTP_MC_FABRIC_HANDLE_BYTES) {
        close(fd);
        return -1;
    }
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static rtp_mc_response base_response(const rtp_mc_request* request, rtp_mc_status status) {
    rtp_mc_response response;
    memset(&response, 0, sizeof(response));
    response.magic              = RTP_MC_PROTOCOL_MAGIC;
    response.version            = RTP_MC_PROTOCOL_VERSION;
    response.opcode             = request == NULL ? RTP_MC_OP_PING : request->opcode;
    response.struct_size        = sizeof(response);
    response.status             = status;
    response.local_device_count = g_local_gpu_count;
    response.holder_instance_hi = g_instance_hi;
    response.holder_instance_lo = g_instance_lo;
    return response;
}

static rtp_mc_status
validate_properties(const rtp_mc_object_properties* properties, uint32_t gpu_count, uint16_t opcode) {
    if (properties->size == 0 || properties->num_devices == 0 || properties->handle_types == 0) {
        return RTP_MC_STATUS_INVALID_REQUEST;
    }
    // UNKNOWN_SIZE is metadata for a raw peer FABRIC import. It may persist as
    // the exact identity of a peer-created entry, but must never reach CREATE
    // where the creator would otherwise try to allocate UINT64_MAX bytes.
    if (opcode == RTP_MC_OP_CREATE && properties->size == RTP_MC_UNKNOWN_SIZE) {
        return RTP_MC_STATUS_INVALID_REQUEST;
    }
    if (properties->flags != 0 || (properties->handle_types & ~RTP_MC_SUPPORTED_HANDLE_TYPES) != 0) {
        return RTP_MC_STATUS_UNSUPPORTED_PROPERTIES;
    }
    // FABRIC requires an explicit full-team contract at holder startup. This
    // prevents an incomplete node set from reaching CUDA, where bind/map waits
    // indefinitely for all numDevices participants. POSIX remains single-node.
    const int want_fabric = (properties->handle_types & RTP_MC_HANDLE_TYPE_FABRIC) != 0;
    if (want_fabric ? (g_fabric_team_size == 0 || properties->num_devices != g_fabric_team_size) :
                      (properties->num_devices != gpu_count)) {
        return RTP_MC_STATUS_UNSUPPORTED_PROPERTIES;
    }
    return RTP_MC_STATUS_OK;
}

static void fill_entry_response(rtp_mc_response* response, const held_entry* entry) {
    response->object_id      = entry->object_id;
    response->requested_size = entry->properties.size;
    response->served_size    = entry->served_size;
    response->num_devices    = entry->properties.num_devices;
    response->handle_types   = entry->properties.handle_types;
    response->flags          = entry->properties.flags;
}

static void
handle_client(int client_fd, const char* creator, const char* gpus, uint32_t gpu_count, int creator_timeout_ms) {
    // Accept the base 64-byte request (PING and legacy clients), the 80-byte
    // extended request that carries owner attribution, and the 144-byte
    // IMPORT_ADD request that appends a 64-byte fabric handle. SEQPACKET
    // preserves the message boundary; the extra byte in the buffer lets an
    // oversized message be rejected instead of silently truncated.
    rtp_mc_import_add_request import_add;
    memset(&import_add, 0, sizeof(import_add));
    rtp_mc_request_ext* extended_ptr = &import_add.ext;
    unsigned char       buffer[sizeof(rtp_mc_import_add_request) + 1];
    ssize_t             received         = recv(client_fd, buffer, sizeof(buffer), 0);
    uint64_t            owner_id         = 0;
    uint64_t            owner_generation = 0;
    int                 has_fabric_arg   = 0;
    if (received == (ssize_t)sizeof(rtp_mc_request)) {
        memcpy(&extended_ptr->base, buffer, sizeof(rtp_mc_request));
    } else if (received == (ssize_t)sizeof(rtp_mc_request_ext)) {
        memcpy(extended_ptr, buffer, sizeof(rtp_mc_request_ext));
        owner_id         = extended_ptr->owner_id;
        owner_generation = extended_ptr->owner_generation;
    } else if (received == (ssize_t)sizeof(rtp_mc_import_add_request)) {
        memcpy(&import_add, buffer, sizeof(rtp_mc_import_add_request));
        owner_id         = extended_ptr->owner_id;
        owner_generation = extended_ptr->owner_generation;
        has_fabric_arg   = 1;
    } else {
        log_message("invalid_request_size received=%zd expected=%zu, %zu or %zu",
                    received,
                    sizeof(rtp_mc_request),
                    sizeof(rtp_mc_request_ext),
                    sizeof(rtp_mc_import_add_request));
        return;
    }
    rtp_mc_request request = extended_ptr->base;
    if (request.magic != RTP_MC_PROTOCOL_MAGIC || request.version != RTP_MC_PROTOCOL_VERSION
        || request.struct_size != (uint32_t)received) {
        rtp_mc_response response = base_response(&request, RTP_MC_STATUS_INVALID_REQUEST);
        (void)send_response(client_fd, &response, -1);
        return;
    }

    // IMPORT_ADD is the only opcode that carries the 64-byte fabric handle (the
    // 144-byte message form); no other opcode may, and IMPORT_ADD requires it.
    if ((request.opcode == RTP_MC_OP_IMPORT_ADD) != (has_fabric_arg != 0)) {
        rtp_mc_response response = base_response(&request, RTP_MC_STATUS_INVALID_REQUEST);
        (void)send_response(client_fd, &response, -1);
        return;
    }
    if (request.opcode == RTP_MC_OP_IMPORT_ADD && owner_generation == 0) {
        rtp_mc_response response = base_response(&request, RTP_MC_STATUS_INVALID_REQUEST);
        (void)send_response(client_fd, &response, -1);
        return;
    }

    if (request.opcode == RTP_MC_OP_PING) {
        rtp_mc_response response = base_response(&request, RTP_MC_STATUS_OK);
        response.object_id       = (uint64_t)g_entry_count;
        (void)send_response(client_fd, &response, -1);
        return;
    }

    rtp_mc_status property_status = validate_properties(&request.properties, gpu_count, request.opcode);
    if (property_status != RTP_MC_STATUS_OK) {
        rtp_mc_response response = base_response(&request, property_status);
        (void)send_response(client_fd, &response, -1);
        return;
    }

    held_entry*   entry       = NULL;
    rtp_mc_status status      = RTP_MC_STATUS_OK;
    int           owner_added = 0;
    if (request.opcode == RTP_MC_OP_CREATE) {
        if (request.holder_instance_hi != 0 || request.holder_instance_lo != 0 || request.object_id != 0) {
            status = RTP_MC_STATUS_INVALID_REQUEST;
        } else {
            // Reclaim orphans from a prior incarnation of this owner before the
            // capacity check so a restart frees, rather than exhausts, slots.
            size_t reclaimed = reclaim_stale_owner(owner_id, owner_generation);
            if (reclaimed > 0) {
                log_message("create_reclaimed owner=%llu reclaimed=%zu entries=%zu",
                            (unsigned long long)owner_id,
                            reclaimed,
                            g_entry_count);
            }
            if (g_entry_count == RTP_MC_MAX_ENTRIES) {
                status = RTP_MC_STATUS_CAPACITY_EXCEEDED;
            } else {
                entry = create_entry(
                    creator, gpus, &request.properties, owner_id, owner_generation, creator_timeout_ms, NULL);
                if (entry == NULL) {
                    status = RTP_MC_STATUS_CREATOR_FAILED;
                } else {
                    owner_added = 1;
                }
            }
        }
    } else if (request.opcode == RTP_MC_OP_IMPORT_ADD) {
        // A peer-node rank (or a co-located rank on the creator node) presents
        // the 64-byte fabric handle the creator broadcast. Dedup by the handle so
        // one node-local entry backs every local rank; the import + local
        // AddDevice + node-local re-export happens only on the first request.
        // Reclaim this caller's previous generation even on a dedup hit, then
        // idempotently register the current process incarnation as an owner.
        size_t reclaimed = reclaim_stale_owner(owner_id, owner_generation);
        if (reclaimed > 0) {
            log_message("import_reclaimed owner=%llu refs=%zu entries=%zu",
                        (unsigned long long)owner_id,
                        reclaimed,
                        g_entry_count);
        }
        entry = find_entry_by_fabric(import_add.fabric_handle);
        if (entry != NULL) {
            if (!compatible_import_properties(&request.properties, &entry->properties)) {
                status = RTP_MC_STATUS_PROPERTY_MISMATCH;
                entry  = NULL;
            } else {
                int add_result = add_owner_ref(entry, owner_id, owner_generation);
                if (add_result < 0) {
                    status = RTP_MC_STATUS_INTERNAL_ERROR;
                    entry  = NULL;
                } else {
                    owner_added = add_result;
                }
            }
        } else {
            if (g_entry_count == RTP_MC_MAX_ENTRIES) {
                status = RTP_MC_STATUS_CAPACITY_EXCEEDED;
            } else {
                entry = create_entry(creator,
                                     gpus,
                                     &request.properties,
                                     owner_id,
                                     owner_generation,
                                     creator_timeout_ms,
                                     import_add.fabric_handle);
                if (entry == NULL) {
                    status = RTP_MC_STATUS_CREATOR_FAILED;
                } else {
                    owner_added = 1;
                }
            }
        }
    } else if (request.opcode == RTP_MC_OP_FETCH_FABRIC) {
        // The creating rank retrieves the 64-byte fabric handle so it can publish
        // it to peers (via torch's store exchange). Resolve like FETCH; the
        // handle is returned in a sealed memfd rather than the response struct.
        if (request.holder_instance_hi != g_instance_hi || request.holder_instance_lo != g_instance_lo) {
            status = RTP_MC_STATUS_STALE_INSTANCE;
        } else if (request.object_id == 0 || (entry = find_entry(request.object_id)) == NULL) {
            status = RTP_MC_STATUS_UNKNOWN_OBJECT;
        } else if (!entry->has_fabric || !same_properties(&request.properties, &entry->properties)) {
            status = RTP_MC_STATUS_PROPERTY_MISMATCH;
            entry  = NULL;
        }
    } else if (request.opcode == RTP_MC_OP_FETCH) {
        if (request.holder_instance_hi != g_instance_hi || request.holder_instance_lo != g_instance_lo) {
            status = RTP_MC_STATUS_STALE_INSTANCE;
        } else if (request.object_id == 0 || (entry = find_entry(request.object_id)) == NULL) {
            status = RTP_MC_STATUS_UNKNOWN_OBJECT;
        } else if (!same_properties(&request.properties, &entry->properties)) {
            status = RTP_MC_STATUS_PROPERTY_MISMATCH;
            entry  = NULL;
        }
    } else if (request.opcode == RTP_MC_OP_RELEASE) {
        // RELEASE removes only this exact process-incarnation reference. The
        // entry remains alive until its last owner releases it.
        held_entry* target = NULL;
        if (request.holder_instance_hi != g_instance_hi || request.holder_instance_lo != g_instance_lo) {
            status = RTP_MC_STATUS_STALE_INSTANCE;
        } else if (request.object_id == 0 || (target = find_entry(request.object_id)) == NULL) {
            status = RTP_MC_STATUS_UNKNOWN_OBJECT;
        } else if (find_owner_ref(target, owner_id, owner_generation) < 0) {
            status = RTP_MC_STATUS_OWNER_MISMATCH;
        } else if (!same_properties(&request.properties, &target->properties)) {
            status = RTP_MC_STATUS_PROPERTY_MISMATCH;
        } else {
            uint64_t released_object = target->object_id;
            size_t   remaining_refs  = drop_owner_ref(target, owner_id, owner_generation);
            log_message("released object=%llu owner=%llu gen=%llu refs=%zu entries=%zu",
                        (unsigned long long)released_object,
                        (unsigned long long)owner_id,
                        (unsigned long long)owner_generation,
                        remaining_refs,
                        g_entry_count);
        }
    } else {
        status = RTP_MC_STATUS_INVALID_REQUEST;
    }

    rtp_mc_response response = base_response(&request, status);
    // reply_fd is the node-local multicast POSIX fd for CREATE/FETCH/IMPORT_ADD,
    // or a transient sealed memfd carrying the 64-byte fabric handle for
    // FETCH_FABRIC (owned here and closed after the reply).
    int reply_fd       = -1;
    int owned_reply_fd = -1;
    if (entry != NULL && status == RTP_MC_STATUS_OK) {
        fill_entry_response(&response, entry);
        if (request.opcode == RTP_MC_OP_FETCH_FABRIC) {
            owned_reply_fd = make_fabric_memfd(entry->fabric_handle);
            if (owned_reply_fd < 0) {
                response = base_response(&request, RTP_MC_STATUS_INTERNAL_ERROR);
            } else {
                reply_fd = owned_reply_fd;
            }
        } else {
            reply_fd = entry->fd;
        }
    }
    if (send_response(client_fd, &response, reply_fd) != 0) {
        if (owned_reply_fd >= 0) {
            close(owned_reply_fd);
        }
        if (owner_added && request.opcode == RTP_MC_OP_CREATE && response.object_id != 0) {
            held_entry* registered = find_entry(response.object_id);
            if (registered != NULL) {
                (void)drop_owner_ref(registered, owner_id, owner_generation);
            }
        }
        log_message("reply_failed opcode=%u status=%d error=%s", request.opcode, status, strerror(errno));
        return;
    }
    if (owned_reply_fd >= 0) {
        close(owned_reply_fd);
    }
    log_message("request opcode=%u status=%d object=%llu requested=%llu served=%llu",
                request.opcode,
                status,
                (unsigned long long)response.object_id,
                (unsigned long long)response.requested_size,
                (unsigned long long)response.served_size);
}

static int write_ready_file(
    const char* path, const char* socket_path, const char* creator, const char* gpus, uint32_t fabric_team_size) {
    if (path == NULL) {
        return 0;
    }
    char temporary[4096];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path, (int)getpid()) >= (int)sizeof(temporary)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        return -1;
    }
    char contents[4096];
    int  length = snprintf(contents,
                          sizeof(contents),
                          "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1\n"
                           "state=ready\npid=%d\nprotocol=%d\n"
                           "instance=%016llx%016llx\nsocket=%s\ncreator=%s\ngpus=%s\nfabric_team_size=%u\n",
                          (int)getpid(),
                          RTP_MC_PROTOCOL_VERSION,
                          (unsigned long long)g_instance_hi,
                          (unsigned long long)g_instance_lo,
                          socket_path,
                          creator,
                          gpus,
                          fabric_team_size);
    int  result = 0;
    if (length < 0 || length >= (int)sizeof(contents) || write(fd, contents, (size_t)length) != length
        || fsync(fd) != 0) {
        result = -1;
    }
    if (close(fd) != 0) {
        result = -1;
    }
    if (result == 0 && rename(temporary, path) != 0) {
        result = -1;
    }
    if (result != 0) {
        int saved_errno = errno;
        unlink(temporary);
        errno = saved_errno;
    }
    return result;
}

static int connect_socket(const char* path) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", path);
    if (connect(fd, (struct sockaddr*)&address, sizeof(address)) != 0) {
        int saved_errno = errno;
        close(fd);
        errno = saved_errno;
        return -1;
    }
    return fd;
}

static int create_listener(const char* path, mode_t socket_mode) {
    if (access(path, F_OK) == 0) {
        int existing = connect_socket(path);
        if (existing >= 0) {
            close(existing);
            errno = EADDRINUSE;
            return -1;
        }
        if (errno != ECONNREFUSED && errno != ENOENT) {
            // A live holder speaking a different socket protocol, or a path
            // with unexpected type/permissions, must never be unlinked.
            errno = EADDRINUSE;
            return -1;
        }
        if (unlink(path) != 0) {
            return -1;
        }
    }
    int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (listener < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", path);
    if (bind(listener, (struct sockaddr*)&address, sizeof(address)) != 0 || chmod(path, socket_mode) != 0
        || listen(listener, 64) != 0) {
        int saved_errno = errno;
        close(listener);
        unlink(path);
        errno = saved_errno;
        return -1;
    }
    return listener;
}

static void stop_handler(int signal_number) {
    (void)signal_number;
    g_stopping = 1;
    if (g_creator_pid > 0) {
        kill((pid_t)g_creator_pid, SIGTERM);
    }
    if (g_listener >= 0) {
        shutdown(g_listener, SHUT_RDWR);
        close(g_listener);
        g_listener = -1;
    }
}

static int check_holder(const char* socket_path) {
    int socket_fd = connect_socket(socket_path);
    if (socket_fd < 0) {
        perror("connect(holder)");
        return 1;
    }
    (void)set_socket_timeout(socket_fd, 1000);
    rtp_mc_request request;
    memset(&request, 0, sizeof(request));
    request.magic       = RTP_MC_PROTOCOL_MAGIC;
    request.version     = RTP_MC_PROTOCOL_VERSION;
    request.opcode      = RTP_MC_OP_PING;
    request.struct_size = sizeof(request);
    rtp_mc_response response;
    ssize_t         sent     = send(socket_fd, &request, sizeof(request), MSG_NOSIGNAL);
    ssize_t         received = recv(socket_fd, &response, sizeof(response), 0);
    close(socket_fd);
    if (sent != (ssize_t)sizeof(request) || received != (ssize_t)sizeof(response)
        || response.magic != RTP_MC_PROTOCOL_MAGIC || response.version != RTP_MC_PROTOCOL_VERSION
        || response.opcode != RTP_MC_OP_PING || response.status != RTP_MC_STATUS_OK) {
        fprintf(stderr, "holder returned an invalid readiness response\n");
        return 1;
    }
    printf("HOLDER_OK protocol=%u instance=%016llx%016llx entries=%llu socket=%s\n",
           response.version,
           (unsigned long long)response.holder_instance_hi,
           (unsigned long long)response.holder_instance_lo,
           (unsigned long long)response.object_id,
           socket_path);
    return 0;
}

int main(int argc, char** argv) {
    const char*                socket_path        = NULL;
    const char*                ready_file         = NULL;
    const char*                creator            = NULL;
    const char*                gpus               = NULL;
    uint32_t                   fabric_team_size   = 0;
    int                        have_fabric_team   = 0;
    int                        check_only         = 0;
    int                        client_timeout_ms  = 1000;
    int                        creator_timeout_ms = 120000;
    mode_t                     socket_mode        = 0600;
    static const struct option options[]          = {
        {"socket", required_argument, NULL, 's'},
        {"ready-file", required_argument, NULL, 'r'},
        {"socket-mode", required_argument, NULL, 'm'},
        {"creator", required_argument, NULL, 'C'},
        {"client-timeout-ms", required_argument, NULL, 'i'},
        {"creator-timeout-ms", required_argument, NULL, 't'},
        {"gpus", required_argument, NULL, 'g'},
        {"fabric-team-size", required_argument, NULL, 'f'},
        {"check", no_argument, NULL, 'c'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "s:r:m:C:i:t:g:f:ch", options, NULL)) != -1) {
        switch (option) {
            case 's':
                socket_path = optarg;
                break;
            case 'r':
                ready_file = optarg;
                break;
            case 'C':
                creator = optarg;
                break;
            case 'g':
                gpus = optarg;
                break;
            case 'f':
                if (parse_positive_u32(optarg, &fabric_team_size) != 0) {
                    fprintf(stderr, "invalid --fabric-team-size: %s\n", optarg);
                    return 2;
                }
                have_fabric_team = 1;
                break;
            case 'm': {
                char* end            = NULL;
                errno                = 0;
                unsigned long parsed = strtoul(optarg, &end, 8);
                if (errno != 0 || end == optarg || *end != '\0' || parsed > 0777) {
                    fprintf(stderr, "invalid --socket-mode: %s\n", optarg);
                    return 2;
                }
                socket_mode = (mode_t)parsed;
                break;
            }
            case 'i':
            case 't': {
                char* end    = NULL;
                long  parsed = strtol(optarg, &end, 10);
                if (end == optarg || *end != '\0' || parsed < 100 || parsed > 600000) {
                    fprintf(stderr, "invalid timeout: %s\n", optarg);
                    return 2;
                }
                if (option == 'i') {
                    client_timeout_ms = (int)parsed;
                } else {
                    creator_timeout_ms = (int)parsed;
                }
                break;
            }
            case 'c':
                check_only = 1;
                break;
            case 'h':
                usage(stdout, argv[0]);
                return 0;
            default:
                usage(stderr, argv[0]);
                return 2;
        }
    }
    if (optind != argc || validate_socket_path(socket_path) != 0) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (check_only) {
        if (ready_file != NULL || creator != NULL || gpus != NULL || have_fabric_team) {
            fprintf(stderr, "--check only accepts --socket\n");
            return 2;
        }
        return check_holder(socket_path);
    }
    uint32_t gpu_count = 0;
    if (creator == NULL || creator[0] == '\0' || parse_gpu_count(gpus, &gpu_count) != 0) {
        fprintf(stderr, "--creator and a valid --gpus list are required\n");
        return 2;
    }
    if (have_fabric_team && fabric_team_size < gpu_count) {
        fprintf(stderr, "--fabric-team-size must be at least the local --gpus count\n");
        return 2;
    }
    g_local_gpu_count  = gpu_count;
    g_fabric_team_size = fabric_team_size;
    if (random_instance() != 0) {
        perror("getrandom(holder instance)");
        return 1;
    }

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, NULL) != 0 || sigaction(SIGINT, &action, NULL) != 0) {
        perror("sigaction");
        return 1;
    }
    g_listener = create_listener(socket_path, socket_mode);
    if (g_listener < 0) {
        perror("listen(holder)");
        return 1;
    }
    if (write_ready_file(ready_file, socket_path, creator, gpus, fabric_team_size) != 0) {
        perror("write ready file");
        close(g_listener);
        unlink(socket_path);
        return 1;
    }
    log_message("HOLDER_READY protocol=%d instance=%016llx%016llx socket=%s gpus=%s fabric_team_size=%u no_cuda=1",
                RTP_MC_PROTOCOL_VERSION,
                (unsigned long long)g_instance_hi,
                (unsigned long long)g_instance_lo,
                socket_path,
                gpus,
                fabric_team_size);

    while (!g_stopping) {
        int client = accept4(g_listener, NULL, NULL, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR || g_stopping) {
                continue;
            }
            perror("accept(holder)");
            break;
        }
        if (set_socket_timeout(client, client_timeout_ms) != 0) {
            perror("setsockopt(client timeout)");
        } else {
            handle_client(client, creator, gpus, gpu_count, creator_timeout_ms);
        }
        close(client);
    }

    for (size_t i = 0; i < g_entry_count; ++i) {
        close(g_entries[i].fd);
        free(g_entries[i].owners);
    }
    if (g_listener >= 0) {
        close(g_listener);
    }
    unlink(socket_path);
    if (ready_file != NULL) {
        unlink(ready_file);
    }
    log_message("HOLDER_EXIT cleaned_entries=%zu", g_entry_count);
    return 0;
}
