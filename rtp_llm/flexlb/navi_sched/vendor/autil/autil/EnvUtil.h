#pragma once

#include <stdlib.h>
#include <string>

#include "autil/EnvUtilImpl.h"
#include "autil/Log.h"
#include "autil/Singleton.h"
#include "autil/StringUtil.h"

namespace autil {

class EnvUtil {
public:
    template <typename T>
    static T getEnv(const std::string &key, const T &defaultValue) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->getEnv(key, defaultValue);
    }
    template <typename T>
    static bool getEnvWithoutDefault(const std::string &key, T &value) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->getEnvWithoutDefault(key, value);
    }

    static std::string getEnv(const std::string &key, const std::string &defaulValue = "") {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->getEnv(key, defaulValue);
    }

    static bool setEnv(const std::string &env, const std::string &value, bool overwrite = true) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->setEnv(env, value, overwrite);
    }

    static bool unsetEnv(const std::string &env) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->unsetEnv(env);
    }

    static bool hasEnv(const std::string &env) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->hasEnv(env);
    }

    static std::string envReplace(const std::string &value) {
        EnvUtilImpl *impl = autil::Singleton<EnvUtilImpl>::getInstance();
        return impl->envReplace(value);
    }
};

class EnvGuard {
public:
    EnvGuard(const std::string &key, const std::string &newValue) : _key(key) {
        _keyExist = EnvUtil::getEnvWithoutDefault(key, _oldValue);
        EnvUtil::setEnv(key, newValue);
    }
    ~EnvGuard() {
        if (!_keyExist) {
            EnvUtil::unsetEnv(_key);
        } else {
            EnvUtil::setEnv(_key, _oldValue);
        }
    }

private:
    std::string _key;
    std::string _oldValue;
    bool _keyExist = false;
};

} // namespace autil
