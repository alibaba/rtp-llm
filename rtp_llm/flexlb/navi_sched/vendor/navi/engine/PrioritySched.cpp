#include "navi/engine/PrioritySched.h"
#include "navi/log/NaviLogger.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/resource.h>

#define PRIORITY_SCHED_IOC_MAGIC 'P'
#define PRIORITY_SCHED_IOC_ALLOC_SCHED_INFO _IOR(PRIORITY_SCHED_IOC_MAGIC, 1, size_t)
#define PRIORITY_SCHED_IOC_MAXNR 1

#define PAGE_SIZE 4096
#define PAGE_MASK (~(PAGE_SIZE-1))
#define PAGE_ALIGN(x) ((x + PAGE_SIZE - 1) & PAGE_MASK)

#define CACHE_SIZE 64
#define CACHE_MASK (~(CACHE_SIZE-1))
#define CACHE_ALIGN(x) ((x + CACHE_SIZE - 1) & CACHE_MASK)

namespace navi {

PrioritySched::~PrioritySched() {
    stop();
}

bool PrioritySched::init(int threadNum) {
    auto fd = open("/dev/priority_sched0", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        NAVI_KERNEL_LOG(INFO, "open /dev/priority_sched0 failed, ignored");
        return false;
    }
    size_t mmapSize = PAGE_ALIGN(CACHE_ALIGN(sizeof(UserPriorityInfo)) * threadNum);
    void *mem = mmap(0, mmapSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (MAP_FAILED == mem) {
        NAVI_KERNEL_LOG(INFO, "mmap /dev/priority_sched0 failed, ignored");
        close(fd);
        return false;
    }
    _fd = fd;
    _mmap = mem;
    _mmapSize = mmapSize;
    NAVI_KERNEL_LOG(INFO, "init priority sched success, fd: %d, mem: %p, size: %lu", fd, mem, mmapSize);
    return true;
}

UserPriorityInfo *PrioritySched::alloc() {
    if (_fd == -1) {
        return nullptr;
    }
    UserPriorityInfo *priorityInfo = NULL;
    int err = ioctl(_fd, PRIORITY_SCHED_IOC_ALLOC_SCHED_INFO, &priorityInfo);
    if (err < 0) {
        return nullptr;
    }
    return priorityInfo;
}

void PrioritySched::stop() {
    if (_mmap) {
        munmap(_mmap, _mmapSize);
        _mmap = nullptr;
    }
    if (_fd != -1) {
        close(_fd);
        _fd = -1;
    }
}

}
