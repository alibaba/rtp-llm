#include <bits/types/struct_tm.h>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fstream>
#include <sstream>
#include <string>

#include "autil/Lock.h"

using namespace std;

static autil::ThreadMutex ENV_MUTEX;

typedef char *(*GetenvFunc)(const char *name);
static GetenvFunc libc_real_getenv = nullptr;
static int32_t MAX_STACK_TRACE_SIZE = 15;
static int32_t f_count = 0;

static int64_t currentTime() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (tval.tv_sec * 1000000LL + tval.tv_usec);
}

static int32_t run_time = currentTime();

std::string getSubString(const std::string &str, char begin, char end) {
    size_t startPos = str.find(begin);
    size_t endPos = str.find(end);
    return str.substr(startPos + 1, endPos - startPos - 1);
}

string backtrace(const char *op, const char *name) {
    std::string value;
    const int32_t maxDepth = MAX_STACK_TRACE_SIZE;
    void *stack_addrs[maxDepth];
    size_t stack_depth = backtrace(stack_addrs, maxDepth);
    char **stack_strings = backtrace_symbols(stack_addrs, maxDepth);

    for (size_t i = 0; i < stack_depth; i++) {
        std::string stackStr = stack_strings[stack_depth - i - 1];

        size_t start = stackStr.find('(');
        std::string prefixStr = stackStr.substr(0, start);
        std::string funcSym = getSubString(stackStr, '(', '+');
        std::string moveNumStr = getSubString(stackStr, '+', ')');
        std::string addressStr = getSubString(stackStr, '[', ']');
        char *name = abi::__cxa_demangle(funcSym.c_str(), NULL, NULL, NULL);
        std::string funcName = (name == NULL) ? "" : std::string(name);
        std::string lineStr = prefixStr + "(" + funcName + "):[" + funcSym + "]" + addressStr + ":" + moveNumStr;
        if (i == 0) {
            value = lineStr;
        } else {
            value += std::string(" -> ") + lineStr + std::string("\n");
        }
        free(name);
    }
    free(stack_strings);

    stringstream oss;
    oss << run_time << "_" << f_count++;
    ofstream out(string(op) + "_" + string(name) + "_" + oss.str());
    out << "env name: " << name << endl;
    out << value << endl;
    out.close();
    return value;
}

extern "C" {
char *getenv(const char *name) {
    autil::ScopedLock lock(ENV_MUTEX);
    //    backtrace("getenv", name);
    if (!libc_real_getenv) {
        libc_real_getenv = (GetenvFunc)dlsym(RTLD_NEXT, "getenv");
    }
    return libc_real_getenv(name);
}
}

static GetenvFunc libc_real_secure_getenv = nullptr;

extern "C" {
char *secure_getenv(const char *name) {
    autil::ScopedLock lock(ENV_MUTEX);
    //    backtrace("getenv", name);
    if (!libc_real_secure_getenv) {
        libc_real_secure_getenv = (GetenvFunc)dlsym(RTLD_NEXT, "secure_getenv");
    }
    return libc_real_secure_getenv(name);
}
}

typedef int (*SetenvFunc)(const char *name, const char *value, int overwrite);
static SetenvFunc libc_real_setenv = nullptr;

extern "C" {
int setenv(const char *name, const char *value, int overwrite) {
    autil::ScopedLock lock(ENV_MUTEX);
    //    backtrace("setenv", name);
    if (!libc_real_setenv) {
        libc_real_setenv = (SetenvFunc)dlsym(RTLD_NEXT, "setenv");
    }
    return libc_real_setenv(name, value, overwrite);
}
}

typedef int (*UnsetenvFunc)(const char *name);
static UnsetenvFunc libc_real_unsetenv = nullptr;

extern "C" {
int unsetenv(const char *name) {
    autil::ScopedLock lock(ENV_MUTEX);
    // backtrace("unsetenv", name);
    if (!libc_real_unsetenv) {
        libc_real_unsetenv = (UnsetenvFunc)dlsym(RTLD_NEXT, "unsetenv");
    }
    return libc_real_unsetenv(name);
}
}
