#pragma once
#include <unistd.h>

namespace navi {

struct UserPriorityInfo {
    size_t high;
    size_t low;
};

class PrioritySched {
public:
    ~PrioritySched();
public:
    bool init(int threadNum);
    UserPriorityInfo *alloc();
    void stop();
private:
    int _fd = -1;
    size_t _mmapSize = 0;
    void *_mmap = nullptr;
};

}
