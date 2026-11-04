#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/VersionInfo.h"
#include "navi/engine/Resource.h"

namespace navi {

class GigMetaR;
NAVI_TYPEDEF_PTR(GigMetaR);

class GigMetaR : public navi::Resource {
public:
    GigMetaR();
    GigMetaR(const std::vector<multi_call::BizMetaInfo> &bizMetaInfos,
             const multi_call::TopoSignatureSetPtr &gigMetaLocalSignatures);
    ~GigMetaR();
    GigMetaR(const GigMetaR &) = delete;
    GigMetaR &operator=(const GigMetaR &) = delete;

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    navi::SummaryErrorCode getSummary(const std::string &query,
            navi::SummaryBuilder &builder) const override;

public:
    int64_t version() const { return _version; }
    const std::vector<multi_call::BizMetaInfo> &getBizMetaInfos() const { return _bizMetaInfos; }
    bool isSame(const GigMetaRPtr &otherPtr) const;
    void log() const;
    const multi_call::TopoSignatureSetPtr &getGigMetaLocalSignatures() const {
        return _gigMetaLocalSignatures;
    }
private:
    bool isVersionSame(const std::map<multi_call::VersionTy, multi_call::VersionInfo> &leftVersions,
                       const std::map<multi_call::VersionTy, multi_call::VersionInfo> &rightVersions) const;

public:
    static const std::string RESOURCE_ID;

private:
    int64_t _version = -1;
    multi_call::TopoSignatureSetPtr _gigMetaLocalSignatures;
    std::vector<multi_call::BizMetaInfo> _bizMetaInfos;
};

} // namespace navi
