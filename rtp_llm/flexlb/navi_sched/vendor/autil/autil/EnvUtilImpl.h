#pragma once

#include <stdlib.h>
#include <string>

#include "autil/Log.h"
#include "autil/Singleton.h"
#include "autil/StringUtil.h"

namespace autil {

class EnvUtilImpl {
public:
    template <typename T>
    T getEnv(const std::string &key, const T &defaultValue) {
        ScopedReadLock lock(_rwLock);
        const char *str = std::getenv(key.c_str());
        if (!str) {
            return defaultValue;
        }
        T ret = T();
        auto success = StringUtil::fromString(str, ret);
        if (success) {
            return ret;
        } else {
            AUTIL_LOG(WARN,
                      "failed to parse env [%s:%s] to typed, use default value [%s]",
                      key.c_str(),
                      str,
                      StringUtil::toString(defaultValue).c_str());
            return defaultValue;
        }
    }
    template <typename T>
    bool getEnvWithoutDefault(const std::string &key, T &value) {
        ScopedReadLock lock(_rwLock);
        const char *str = std::getenv(key.c_str());
        if (!str) {
            return false;
        }
        return StringUtil::fromString(str, value);
    }

    std::string getEnv(const std::string &key, const std::string &defaulValue = "");

    bool setEnv(const std::string &env, const std::string &value, bool overwrite = true);

    bool unsetEnv(const std::string &env);

    bool hasEnv(const std::string &env);

    std::string envReplace(const std::string &value);

private:
    autil::ReadWriteLock _rwLock;

private:
    AUTIL_LOG_DECLARE();
};
} // namespace autil
