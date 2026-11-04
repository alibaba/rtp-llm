#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/GeneratorBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Request.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MiscUtil.h"
#include "google/protobuf/arena.h"

namespace multi_call {

class RequestGenerator : public GeneratorBase
{
public:
    RequestGenerator(const std::string &bizName, SourceIdTy sourceId = INVALID_SOURCE_ID,
                     VersionTy version = INVALID_VERSION_ID)
        : GeneratorBase("", bizName, "", "", sourceId, version, INVALID_VERSION_ID)
        , _arena(new google::protobuf::Arena()) {
    }
    RequestGenerator(const std::string &clusterName, const std::string &bizName,
                     SourceIdTy sourceId = INVALID_SOURCE_ID,
                     VersionTy version = INVALID_VERSION_ID)
        : GeneratorBase(clusterName, bizName, "", "", sourceId, version, INVALID_VERSION_ID)
        , _arena(new google::protobuf::Arena()) {
    }
    RequestGenerator(const std::string &bizName,
                     const std::shared_ptr<google::protobuf::Arena> &arena,
                     SourceIdTy sourceId = INVALID_SOURCE_ID,
                     VersionTy version = INVALID_VERSION_ID)
        : GeneratorBase("", bizName, "", "", sourceId, version, INVALID_VERSION_ID)
        , _arena(arena) {
    }
    RequestGenerator(const std::string &clusterName, const std::string &bizName,
                     const std::shared_ptr<google::protobuf::Arena> &arena,
                     SourceIdTy sourceId = INVALID_SOURCE_ID,
                     VersionTy version = INVALID_VERSION_ID)
        : GeneratorBase(clusterName, bizName, "", "", sourceId, version, INVALID_VERSION_ID)
        , _arena(arena) {
    }
    virtual ~RequestGenerator() {
    }

private:
    RequestGenerator(const RequestGenerator &);
    RequestGenerator &operator=(const RequestGenerator &);

public:
    virtual void generate(PartIdTy partCnt, PartRequestMap &requestMap) = 0;
    virtual bool hasError() const {
        return false;
    }

public:
    const std::shared_ptr<google::protobuf::Arena> &getProtobufArena() const {
        return _arena;
    }

private:
    std::shared_ptr<google::protobuf::Arena> _arena;
};

MULTI_CALL_TYPEDEF_PTR(RequestGenerator);
typedef std::vector<RequestGeneratorPtr> RequestGeneratorVec;

} // namespace multi_call

