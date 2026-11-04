#pragma once

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class GigMetaEnv;

/**
 * meta info from env
 * pass to client as MetaNode proto
 **/
class MetaEnv
{
public:
    MetaEnv();
    ~MetaEnv();

public:
    bool init();
    bool operator==(const MetaEnv &rhs) const;
    std::map<std::string, std::string> getEnvTags() const;
    std::map<std::string, std::string> getTargetTags() const;

    const std::string getPlatform() const {
        return _platform;
    }
    const std::string getHippoCluster() const {
        return _hippoCluster;
    }
    const std::string getHippoApp() const {
        return _hippoApp;
    }
    const std::string getC2Role() const {
        return _c2role;
    }
    const std::string getC2Group() const {
        return _c2group;
    }

    void setPlatform(const std::string &p) {
        _platform = p;
    }
    void setHippoCluster(const std::string &c) {
        _hippoCluster = c;
    }
    void setHippoApp(const std::string &a) {
        _hippoApp = a;
    }
    void setC2Role(const std::string &r) {
        _c2role = r;
    }
    void setC2Group(const std::string &g) {
        _c2group = g;
    }

    bool valid() const;
    void setFromProto(const GigMetaEnv &gigMetaEnv);
    void fillProto(GigMetaEnv &gigMetaEnv) const;
    void toString(std::string &ret) const;

public:
    static bool isUnknown(const std::string &env);
    static std::string getFromEnv(const std::string &key);

private:
    std::string _platform;
    std::string _hippoCluster;
    std::string _hippoApp;
    std::string _c2role;
    std::string _c2group;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(MetaEnv);

} // namespace multi_call

