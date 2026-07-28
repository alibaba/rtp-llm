/*
 * Native CUDA Driver API checkpoint/restore probe.
 *
 * This deliberately does not use Python, PyTorch, ctypes, RTP-LLM, or the
 * cuda-checkpoint command-line utility.  A child process creates a CUDA
 * context, writes a deterministic pattern to device memory, and waits.  The
 * parent calls the checkpoint Driver APIs directly and, after a successful
 * restore/unlock, asks the child to verify that its device memory survived.
 *
 * Build:
 *   gcc -std=c11 -O2 -Wall -Wextra \
 *     -I/usr/local/cuda/include \
 *     sleep_mode_integration/cuda_checkpoint_native_probe.c \
 *     -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lcuda \
 *     -o /tmp/cuda_checkpoint_native_probe
 *
 * Run:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe
 *
 * Run with the target controlling its own checkpoint, matching NVIDIA's R580
 * migration API sample:
 *   CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 90 \
 *     /tmp/cuda_checkpoint_native_probe --self
 */

#define _POSIX_C_SOURCE 200809L

#include <cuda.h>

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    PATTERN_WORDS = 1024,
    LOCK_TIMEOUT_MS = 10000,
};

static void print_result(const char* operation, CUresult result) {
    const char* name = NULL;
    const char* description = NULL;
    (void)cuGetErrorName(result, &name);
    (void)cuGetErrorString(result, &description);
    printf(
        "%-32s -> %d (%s: %s)\n",
        operation,
        (int)result,
        name != NULL ? name : "unknown",
        description != NULL ? description : "unknown");
}

static int require_success(const char* operation, CUresult result) {
    print_result(operation, result);
    return result == CUDA_SUCCESS;
}

static int print_state(pid_t pid, const char* label, CUprocessState* state_out) {
    CUprocessState state = CU_PROCESS_STATE_FAILED;
    CUresult result = cuCheckpointProcessGetState((int)pid, &state);
    print_result(label, result);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    static const char* const state_names[] = {
        "RUNNING",
        "LOCKED",
        "CHECKPOINTED",
        "FAILED",
    };
    const char* state_name = "UNKNOWN";
    if ((unsigned int)state <
        (sizeof(state_names) / sizeof(state_names[0]))) {
        state_name = state_names[state];
    }
    printf("%-32s    state=%d (%s)\n", "", (int)state, state_name);
    if (state_out != NULL) {
        *state_out = state;
    }
    return 1;
}

static int write_all(int fd, const void* buffer, size_t size) {
    const unsigned char* cursor = buffer;
    while (size != 0) {
        ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        cursor += written;
        size -= (size_t)written;
    }
    return 1;
}

static int read_all(int fd, void* buffer, size_t size) {
    unsigned char* cursor = buffer;
    while (size != 0) {
        ssize_t count = read(fd, cursor, size);
        if (count == 0) {
            return 0;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        cursor += count;
        size -= (size_t)count;
    }
    return 1;
}

static uint32_t pattern_at(size_t index) {
    return UINT32_C(0xc0da0000) ^ (uint32_t)index;
}

static int run_cuda_child(int ready_fd, int resume_fd) {
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr allocation = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    unsigned char signal_byte = 0;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("child cuInit", cuInit(0)) ||
        !require_success("child cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "child cuCtxCreate", cuCtxCreate(&context, NULL, 0, device)) ||
        !require_success(
            "child cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "child cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success("child cuCtxSynchronize", cuCtxSynchronize())) {
        signal_byte = 1;
        (void)write_all(ready_fd, &signal_byte, sizeof(signal_byte));
        return 2;
    }

    printf(
        "child ready: pid=%ld allocation=0x%llx bytes=%zu\n",
        (long)getpid(),
        (unsigned long long)allocation,
        sizeof(expected));
    signal_byte = 0;
    if (!write_all(ready_fd, &signal_byte, sizeof(signal_byte))) {
        perror("child write ready");
        return 3;
    }

    if (!read_all(resume_fd, &signal_byte, sizeof(signal_byte))) {
        perror("child read resume");
        return 4;
    }

    if (!require_success(
            "child cuMemcpyDtoH",
            cuMemcpyDtoH(observed, allocation, sizeof(observed))) ||
        !require_success(
            "child post-restore sync", cuCtxSynchronize())) {
        return 5;
    }

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "device data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 6;
        }
    }

    printf("child verification: PASS (%d words preserved)\n", PATTERN_WORDS);
    (void)cuMemFree(allocation);
    (void)cuCtxDestroy(context);
    return 0;
}

static void kill_and_reap(pid_t child) {
    if (kill(child, SIGKILL) != 0 && errno != ESRCH) {
        perror("kill child");
    }
    while (waitpid(child, NULL, 0) < 0 && errno == EINTR) {
    }
}

static int run_self_probe(void) {
    CUdevice device;
    CUcontext context = NULL;
    CUdeviceptr allocation = 0;
    uint32_t expected[PATTERN_WORDS];
    uint32_t observed[PATTERN_WORDS];
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult lock_result;
    CUresult checkpoint_result = CUDA_ERROR_UNKNOWN;
    CUresult restore_result = CUDA_ERROR_UNKNOWN;
    CUresult unlock_result = CUDA_ERROR_UNKNOWN;

    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        expected[index] = pattern_at(index);
    }

    if (!require_success("self cuInit", cuInit(0)) ||
        !require_success(
            "self cuDeviceGet", cuDeviceGet(&device, 0)) ||
        !require_success(
            "self primary context",
            cuDevicePrimaryCtxRetain(&context, device)) ||
        !require_success(
            "self cuCtxSetCurrent", cuCtxSetCurrent(context)) ||
        !require_success(
            "self cuMemAlloc",
            cuMemAlloc(&allocation, sizeof(expected))) ||
        !require_success(
            "self cuMemcpyHtoD",
            cuMemcpyHtoD(allocation, expected, sizeof(expected))) ||
        !require_success(
            "self cuCtxSynchronize", cuCtxSynchronize())) {
        return 2;
    }

    printf(
        "self ready: pid=%ld allocation=0x%llx bytes=%zu\n",
        (long)getpid(),
        (unsigned long long)allocation,
        sizeof(expected));

    /*
     * Once this process locks itself, do not call ordinary CUDA APIs such as
     * cuGetErrorName until after Unlock.  Print raw CUresult values while the
     * API lock is held.
     */
    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    lock_result =
        cuCheckpointProcessLock((int)getpid(), &lock_args);
    printf("self Lock raw result             -> %d\n", (int)lock_result);
    if (lock_result == CUDA_SUCCESS) {
        checkpoint_result = cuCheckpointProcessCheckpoint(
            (int)getpid(), &checkpoint_args);
        printf(
            "self Checkpoint raw result       -> %d\n",
            (int)checkpoint_result);
    }
    if (checkpoint_result == CUDA_SUCCESS) {
        restore_result = cuCheckpointProcessRestore(
            (int)getpid(), &restore_args);
        printf(
            "self Restore raw result          -> %d\n",
            (int)restore_result);
    }
    if (restore_result == CUDA_SUCCESS) {
        unlock_result =
            cuCheckpointProcessUnlock((int)getpid(), &unlock_args);
        printf(
            "self Unlock raw result           -> %d\n",
            (int)unlock_result);
    }
    if (unlock_result != CUDA_SUCCESS) {
        fprintf(
            stderr,
            "RESULT: FAIL - self sequence results=%d/%d/%d/%d\n",
            (int)lock_result,
            (int)checkpoint_result,
            (int)restore_result,
            (int)unlock_result);
        return 3;
    }

    if (!require_success(
            "self cuMemcpyDtoH",
            cuMemcpyDtoH(observed, allocation, sizeof(observed))) ||
        !require_success(
            "self post-restore sync", cuCtxSynchronize())) {
        return 4;
    }
    for (size_t index = 0; index < PATTERN_WORDS; ++index) {
        if (observed[index] != expected[index]) {
            fprintf(
                stderr,
                "self data mismatch at word %zu: expected=0x%08x "
                "observed=0x%08x\n",
                index,
                expected[index],
                observed[index]);
            return 5;
        }
    }

    printf(
        "RESULT: PASS - self checkpoint/restore preserved %d words\n",
        PATTERN_WORDS);
    (void)cuMemFree(allocation);
    (void)cuDevicePrimaryCtxRelease(device);
    return 0;
}

int main(int argc, char** argv) {
    int ready_pipe[2];
    int resume_pipe[2];
    pid_t child;
    unsigned char child_status = 1;
    int driver_version = 0;
    int restore_tid = -1;
    CUprocessState state;
    CUcheckpointLockArgs lock_args = {0};
    CUcheckpointCheckpointArgs checkpoint_args = {0};
    CUcheckpointRestoreArgs restore_args = {0};
    CUcheckpointUnlockArgs unlock_args = {0};
    CUresult result;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    if (argc == 2 && strcmp(argv[1], "--self") == 0) {
        return run_self_probe();
    }

    if (argc == 4 && strcmp(argv[1], "--target") == 0) {
        char* ready_end = NULL;
        char* resume_end = NULL;
        long ready_fd = strtol(argv[2], &ready_end, 10);
        long resume_fd = strtol(argv[3], &resume_end, 10);
        if (ready_end == argv[2] || *ready_end != '\0' ||
            resume_end == argv[3] || *resume_end != '\0' ||
            ready_fd < 0 || resume_fd < 0) {
            fprintf(stderr, "invalid target pipe descriptors\n");
            return 2;
        }
        return run_cuda_child((int)ready_fd, (int)resume_fd);
    }

    if (pipe(ready_pipe) != 0 || pipe(resume_pipe) != 0) {
        perror("pipe");
        return 2;
    }

    child = fork();
    if (child < 0) {
        perror("fork");
        return 2;
    }
    if (child == 0) {
        char ready_fd_text[32];
        char resume_fd_text[32];
        close(ready_pipe[0]);
        close(resume_pipe[1]);
        (void)snprintf(
            ready_fd_text,
            sizeof(ready_fd_text),
            "%d",
            ready_pipe[1]);
        (void)snprintf(
            resume_fd_text,
            sizeof(resume_fd_text),
            "%d",
            resume_pipe[0]);
        execl(
            "/proc/self/exe",
            "cuda_checkpoint_native_probe",
            "--target",
            ready_fd_text,
            resume_fd_text,
            (char*)NULL);
        perror("exec /proc/self/exe");
        _exit(127);
    }

    close(ready_pipe[1]);
    close(resume_pipe[0]);
    if (!read_all(ready_pipe[0], &child_status, sizeof(child_status)) ||
        child_status != 0) {
        fprintf(stderr, "child failed during CUDA initialization\n");
        kill_and_reap(child);
        return 3;
    }
    close(ready_pipe[0]);

    if (!require_success("parent cuInit", cuInit(0)) ||
        !require_success(
            "parent cuDriverGetVersion",
            cuDriverGetVersion(&driver_version))) {
        kill_and_reap(child);
        return 4;
    }
    printf("CUDA Driver API version: %d\n", driver_version);

    result = cuCheckpointProcessGetRestoreThreadId((int)child, &restore_tid);
    print_result("GetRestoreThreadId", result);
    if (result == CUDA_SUCCESS) {
        printf("%-32s    tid=%d\n", "", restore_tid);
    }
    if (!print_state(child, "GetState before lock", &state)) {
        kill_and_reap(child);
        return 5;
    }

    lock_args.timeoutMs = LOCK_TIMEOUT_MS;
    result = cuCheckpointProcessLock((int)child, &lock_args);
    print_result("Lock", result);
    if (result != CUDA_SUCCESS ||
        !print_state(child, "GetState after lock", &state)) {
        kill_and_reap(child);
        return 6;
    }

    result =
        cuCheckpointProcessCheckpoint((int)child, &checkpoint_args);
    print_result("Checkpoint(zero args)", result);
    if (result != CUDA_SUCCESS ||
        !print_state(child, "GetState after checkpoint", &state)) {
        kill_and_reap(child);
        return 7;
    }

    result = cuCheckpointProcessRestore((int)child, &restore_args);
    print_result("Restore(zero args)", result);
    (void)print_state(child, "GetState after restore", &state);
    if (result != CUDA_SUCCESS) {
        printf(
            "RESULT: FAIL - native cuCheckpointProcessRestore returned %d\n",
            (int)result);
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 8;
    }

    /*
     * Let the target enter its post-restore CUDA verification call while API
     * entry is still locked.  The call is expected to block until Unlock.
     */
    child_status = 0;
    if (!write_all(
            resume_pipe[1], &child_status, sizeof(child_status))) {
        perror("parent signal child");
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 9;
    }

    result = cuCheckpointProcessUnlock((int)child, &unlock_args);
    print_result("Unlock(zero args)", result);
    (void)print_state(child, "GetState after unlock", &state);
    if (result != CUDA_SUCCESS) {
        close(resume_pipe[1]);
        kill_and_reap(child);
        return 10;
    }
    close(resume_pipe[1]);

    int wait_status = 0;
    while (waitpid(child, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return 11;
        }
    }
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        fprintf(stderr, "child verification failed: status=0x%x\n", wait_status);
        return 12;
    }

    printf("RESULT: PASS - native checkpoint/restore preserved CUDA state\n");
    return 0;
}
