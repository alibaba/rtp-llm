#include "navi/engine/NaviHostInfo.h"
#include "navi/log/NaviLogger.h"
#include "navi/proto/GraphVis.pb.h"
#include <multi_call/rpc/GigRpcServer.h>
#include "autil/CompressionUtil.h"
#include "autil/EnvUtil.h"
#include "autil/FileUtil.h"

namespace navi {

bool NaviHostInfo::init(const std::string &naviName,
                        const std::string &installRoot,
                        multi_call::GigRpcServer *gigRpcServer)
{
    _naviName = naviName;
    initSpec(gigRpcServer);
    initCommitInfo(installRoot);
    initPackageInfo();
    initOpsUrl();
    if (!_history.init(naviName)) {
        NAVI_KERNEL_LOG(ERROR, "init navi history failed");
        return false;
    }
    return true;
}

void NaviHostInfo::fillHostInfo(bool needAll, HostInfoDef *hostInfoDef) const {
    hostInfoDef->set_git_commit_id(_commitId);
    if (!needAll) {
        hostInfoDef->mutable_spec()->set_http_port(_specDef.http_port());
        return;
    }
    hostInfoDef->set_navi_name(_naviName);
    hostInfoDef->mutable_spec()->CopyFrom(_specDef);
    hostInfoDef->set_packager(_packager);
    hostInfoDef->set_package_info(_packageInfo);
    hostInfoDef->set_ops_url(_opsUrl);
}

void NaviHostInfo::initSpec(multi_call::GigRpcServer *gigRpcServer) {
    if (!gigRpcServer) {
        return;
    }
    const auto &spec = gigRpcServer->getRpcSpec();
    _specDef.set_ip(spec.ip);
    _specDef.set_tcp_port(spec.ports[multi_call::MC_PROTOCOL_TCP]);
    _specDef.set_arpc_port(spec.ports[multi_call::MC_PROTOCOL_ARPC]);
    _specDef.set_http_port(spec.ports[multi_call::MC_PROTOCOL_HTTP]);
    _specDef.set_grpc_port(spec.ports[multi_call::MC_PROTOCOL_GRPC]);
    _specDef.set_grpc_stream_port(spec.ports[multi_call::MC_PROTOCOL_GRPC_STREAM]);
    char hostname[256];
    if (0 == gethostname(hostname, 256)) {
        _specDef.set_host_name(std::string(hostname));
    } else {
        _specDef.set_host_name("unknown");
    }
}

void NaviHostInfo::initCommitInfo(const std::string &installRoot) {
    auto commitStr = getFileContent(autil::FileUtil::joinPath(installRoot, "usr/local/etc/git_commit"));
    parseCommit(commitStr);
}

void NaviHostInfo::parseCommit(const std::string &valueStr) {
    auto splited = autil::StringUtil::split(valueStr, "\n");
    for (const auto &line : splited) {
        auto lineSplited = autil::StringUtil::split(line, " ");
        if (lineSplited.size() < 2u) {
            continue;
        }
        const auto &key = lineSplited[0];
        const auto &value = lineSplited[1];
        if (key == "GIT_COMMIT") {
            _commitId = value;
        } else if (key == "USER_NAME") {
            _packager = value;
        }
    }
}

std::string NaviHostInfo::getFileContent(const std::string &path) {
    std::string content;
    if (!autil::FileUtil::readFile(path, content)) {
        return std::string();
    }
    return content;
}

void NaviHostInfo::initPackageInfo() {
    auto content = getFileContent("../package_info.log");
    auto splited = autil::StringUtil::split(content, "\n");
    std::vector<std::string> packageVec;
    for (const auto &line : splited) {
        auto lineSplited = autil::StringUtil::split(line, " ");
        if (lineSplited.size() < 2u) {
            continue;
        }
        const auto &key = lineSplited[0];
        const auto &value = lineSplited[1];
        if (autil::StringUtil::startsWith(key, "Uri")) {
            packageVec.push_back(value);
        }
    }
    auto infoStr = autil::StringUtil::join(packageVec, "\n");
    constexpr size_t MAX_LENGTH = 64 * 1024;
    if (infoStr.size() > MAX_LENGTH) {
        infoStr = infoStr.substr(0, MAX_LENGTH);
    }
    std::string compressed;
    if (autil::CompressionUtil::compress(infoStr, autil::CompressType::Z_BEST_COMPRESS, compressed, nullptr)) {
        _packageInfo = compressed;
    } else {
        _packageInfo = infoStr;
    }
}

void NaviHostInfo::initOpsUrl() {
    _opsUrl = autil::EnvUtil::getEnv("suezops_url", "");
}

void NaviHostInfo::setInitVisProto(const std::shared_ptr<GraphVisDef> &visProto) {
    autil::ScopedWriteLock scope(_visProtoLock);
    _initVisProto = visProto;
}

std::shared_ptr<GraphVisDef> NaviHostInfo::getInitVisProto(bool shortQueryHistory) const {
    std::shared_ptr<GraphVisDef> visProto;
    {
        autil::ScopedReadLock scope(_visProtoLock);
        visProto = _initVisProto;
    }
    if (visProto) {
        auto retProto = std::make_shared<GraphVisDef>();
        retProto->CopyFrom(*visProto);
        if (retProto->nodes_size() > 0) {
            auto nodeDef = retProto->mutable_nodes(0);
            auto naviSummaryDef = nodeDef->mutable_navi_summary();
            fillProcessInfo(*naviSummaryDef->mutable_process_info());
            _history.toProto(shortQueryHistory, *nodeDef->mutable_navi_history());
            nodeDef->mutable_symbol_table()->MergeFrom(nodeDef->navi_history().symbol_table());
            nodeDef->mutable_navi_history()->clear_symbol_table();
        }
        return retProto;
    } else {
        return nullptr;
    }
}

void NaviHostInfo::fillProcessInfo(ProcessInfoDef &processInfo) {
    processInfo.set_cpu_info(NaviHostInfo::getFileContent("/proc/cpuinfo"));
    processInfo.set_proc_maps(NaviHostInfo::getFileContent("/proc/self/maps"));
}

void NaviHostInfo::addVisProtoChild(GraphVisNodeDef *parentDef, const std::string &name, GraphVisNodeDef *childDef) {
    if (!parentDef || !childDef) {
        return;
    }
    auto childSymbolTableDef = childDef->mutable_symbol_table();
    parentDef->mutable_symbol_table()->MergeFrom(*childSymbolTableDef);
    childDef->clear_symbol_table();
    (*parentDef->mutable_sub_results())[name].CopyFrom(*childDef);
}
}

