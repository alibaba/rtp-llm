#include "autil/ThreadNameScope.h"

namespace autil {

ThreadNameScope::ThreadNameScope(const std::string &name) : _name(name), _self(pthread_self()) {
    if (_name.size() >= MAX_LENGTH) {
        _name.resize(MAX_LENGTH);
        _name[MAX_LENGTH - 1] = '\0';
    }
    _ok = (0 == pthread_getname_np(_self, _currentName, MAX_LENGTH));
    if (_ok) {
        pthread_setname_np(_self, _name.c_str());
    }
}

ThreadNameScope::~ThreadNameScope() {
    if (_ok) {
        pthread_setname_np(_self, _currentName);
    }
}

} // namespace autil
