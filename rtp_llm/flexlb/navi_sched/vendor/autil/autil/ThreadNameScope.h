#pragma once

#include <pthread.h>
#include <string>

namespace autil {

class ThreadNameScope {
private:
    static constexpr int MAX_LENGTH = 16;

public:
    ThreadNameScope(const std::string &name);
    ~ThreadNameScope();

private:
    std::string _name;
    pthread_t _self;
    char _currentName[MAX_LENGTH];
    bool _ok;
};

} // namespace autil
