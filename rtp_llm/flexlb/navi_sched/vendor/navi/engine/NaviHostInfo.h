#pragma once

#include "navi/common.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/engine/NaviHistory.h"

namespace multi_call {
class GigRpcServer;
}

namespace navi {

class NaviHostInfo
{
public:
    NaviHostInfo() {
    }
    ~NaviHostInfo() {
    }
    NaviHostInfo(const NaviHostInfo &) = delete;
    NaviHostInfo &operator=(const NaviHostInfo &) = delete;
public:
    bool init(const std::string &naviName, const std::string &installRoot, multi_call::GigRpcServer *gigRpcServer);
    void fillHostInfo(bool needAll, HostInfoDef *hostInfoDef) const;
    NaviHistory *getHistory() {
        return &_history;
    }
    const std::string &getNaviName() const {
        return _naviName;
    }
    void setInitVisProto(const std::shared_ptr<GraphVisDef> &visProto);
    std::shared_ptr<GraphVisDef> getInitVisProto(bool shortQueryHistory) const;
private:
    void initSpec(multi_call::GigRpcServer *gigRpcServer);
    void initCommitInfo(const std::string &installRoot);
    void initPackageInfo();
    void initOpsUrl();
    void parseCommit(const std::string &valueStr);
private:
    static void fillProcessInfo(ProcessInfoDef &processInfo);
public:
    static void addVisProtoChild(GraphVisNodeDef *parentDef, const std::string &name, GraphVisNodeDef *childDef);
    static std::string getFileContent(const std::string &path);
private:
    std::string _naviName;
    std::string _installRoot;
    HostSpecDef _specDef;
    std::string _commitId;
    std::string _packager;
    std::string _packageInfo;
    std::string _opsUrl;
    mutable autil::ReadWriteLock _visProtoLock;
    std::shared_ptr<GraphVisDef> _initVisProto;
    NaviHistory _history;
};

NAVI_TYPEDEF_PTR(NaviHostInfo);

}

