#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/filecontrol.h"

#include <fcntl.h>
namespace anet {
bool FileControl::setCloseOnExec(int fd) {
    int flags;
    flags = fcntl(fd, F_GETFD);
    if (flags == -1) {
        return false;
    }
    flags |= FD_CLOEXEC;
    return fcntl(fd, F_SETFD, flags) != -1;
}

bool FileControl::clearCloseOnExec(int fd) {
    int flags;
    flags = fcntl(fd, F_GETFD);
    if (flags == -1) {
        return false;
    }
    flags &= ~FD_CLOEXEC;
    return fcntl(fd, F_SETFD, flags) != -1;
}

bool FileControl::isCloseOnExec(int fd) {
    int flags;
    flags = fcntl(fd, F_GETFD);
    if (flags == -1) {
        return false;
    }
    return flags & FD_CLOEXEC;
}

} // namespace anet
