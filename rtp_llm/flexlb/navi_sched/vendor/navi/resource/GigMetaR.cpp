#include "navi/resource/GigMetaR.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

static std::atomic<int64_t> GIG_META_VERSION_ID_ALLOCATOR = 0;

const std::string GigMetaR::RESOURCE_ID = "navi.buildin.gig_meta.r";

GigMetaR::GigMetaR() : _version(GIG_META_VERSION_ID_ALLOCATOR.fetch_add(1)) {}

GigMetaR::GigMetaR(const std::vector<multi_call::BizMetaInfo> &bizMetaInfos,
                   const multi_call::TopoSignatureSetPtr &gigMetaLocalSignatures)
    : _version(GIG_META_VERSION_ID_ALLOCATOR.fetch_add(1))
    , _gigMetaLocalSignatures(gigMetaLocalSignatures)
    , _bizMetaInfos(bizMetaInfos) {}

GigMetaR::~GigMetaR() {
}

void GigMetaR::def(navi::ResourceDefBuilder &builder) const {
    builder.name(RESOURCE_ID, navi::RS_SNAPSHOT);
}

bool GigMetaR::config(navi::ResourceConfigContext &ctx) {
    return true;
}

navi::ErrorCode GigMetaR::init(navi::ResourceInitContext &ctx) {
    return navi::EC_NONE;
}

bool GigMetaR::isSame(const GigMetaRPtr &otherPtr) const {
    if (!otherPtr) {
        return false;
    }
    const auto &other = *otherPtr;
    if (_bizMetaInfos.size() != other._bizMetaInfos.size()) {
        return false;
    }
    for (size_t bizIndex = 0; bizIndex < _bizMetaInfos.size(); bizIndex++) {
        const auto &left = _bizMetaInfos[bizIndex];
        const auto &right = other._bizMetaInfos[bizIndex];
        if (left.bizName != right.bizName) {
            return false;
        }
        if (!isVersionSame(left.versions, right.versions)) {
            return false;
        }
    }
    if (!_gigMetaLocalSignatures && !other._gigMetaLocalSignatures) {
        return true;
    } else if (_gigMetaLocalSignatures && other._gigMetaLocalSignatures) {
        return *_gigMetaLocalSignatures == *other._gigMetaLocalSignatures;
    }
    return false;
}

bool GigMetaR::isVersionSame(const std::map<multi_call::VersionTy, multi_call::VersionInfo> &leftVersions,
                             const std::map<multi_call::VersionTy, multi_call::VersionInfo> &rightVersions) const {
    if (leftVersions.size() != rightVersions.size()) {
        return false;
    }
    for (const auto &pair : leftVersions) {
        const auto iter = rightVersions.find(pair.first);
        if (iter == rightVersions.end()) {
            return false;
        }
        const auto &leftVersion = pair.second;
        const auto &rightVersion = iter->second;
        if (leftVersion.partCount != rightVersion.partCount) {
            return false;
        }
        if (!leftVersion.isMetaSame(rightVersion.metas)) {
            return false;
        }
    }
    return true;
}

void GigMetaR::log() const {
    for (const auto &info : _bizMetaInfos) {
        for (const auto &pair : info.versions) {
            const auto versionInfo = pair.second;
            std::string metaStr;
            for (const auto &metaPair : versionInfo.metas) {
                metaStr += std::to_string(metaPair.first) + ": ";
                if (metaPair.second) {
                    metaStr += autil::StringUtil::toString(*metaPair.second) + ", ";
                } else {
                    metaStr += ", ";
                }
            }
            NAVI_KERNEL_LOG(
                INFO,
                "this: %p, ID: %ld, biz: %s, version: %ld, partCount: %lu, healthCount: %lu, other:[%lu, %lu, "
                "%lu, %lu], metas[%s]",
                this,
                _version,
                info.bizName.c_str(),
                pair.first,
                versionInfo.partCount,
                versionInfo.healthCount,
                versionInfo.subscribeCount,
                versionInfo.stopCount,
                versionInfo.heartbeatHealthCount,
                versionInfo.minHeartbeatHealthCount,
                metaStr.c_str());
        }
    }
}

navi::SummaryErrorCode GigMetaR::getSummary(const std::string &query, navi::SummaryBuilder &builder) const {
    {
        auto table = builder.newTable("BizMeta Table");
        table->setDesc("localSubTopoCount: " + (_gigMetaLocalSignatures
                                                    ? std::to_string(_gigMetaLocalSignatures->size())
                                                    : std::string("null(no local sub)")));
        for (const auto &info : _bizMetaInfos) {
            const auto &bizName = info.bizName;
            for (const auto &pair : info.versions) {
                auto version = pair.first;
                const auto &versionInfo = pair.second;
                TABLE_ADD_COLUMN_VALUE(table, "bizName", bizName);
                TABLE_ADD_COLUMN_VALUE(table, "version", version);
                TABLE_ADD_COLUMN_VALUE(table, "partCount", versionInfo.partCount);
                TABLE_ADD_COLUMN_VALUE(table, "subscribeCount", versionInfo.subscribeCount);
                TABLE_ADD_COLUMN_VALUE(table, "stopCount", versionInfo.stopCount);
                TABLE_ADD_COLUMN_VALUE(table, "healthCount", versionInfo.healthCount);
                TABLE_ADD_COLUMN_VALUE(table, "normalProviderCount", versionInfo.normalProviderCount);
                TABLE_ADD_COLUMN_VALUE(table, "heartbeatHealthCount", versionInfo.heartbeatHealthCount);
                TABLE_ADD_COLUMN_VALUE(table, "minHeartbeatHealthCount", versionInfo.minHeartbeatHealthCount);
                TABLE_ADD_COLUMN_VALUE(table, "part_id_count", versionInfo.metas.size());
            }
        }
    }
    {
        auto metaTable = builder.newTable("First PartId Meta Table");
        for (const auto &info : _bizMetaInfos) {
            const auto &bizName = info.bizName;
            for (const auto &pair : info.versions) {
                auto version = pair.first;
                for (const auto &metaPair : pair.second.metas) {
                    const auto &metaPtr = metaPair.second;
                    TABLE_ADD_COLUMN_VALUE(metaTable, "bizName", bizName);
                    TABLE_ADD_COLUMN_VALUE(metaTable, "version", version);
                    TABLE_ADD_COLUMN_VALUE(metaTable, "partId", metaPair.first);
                    TABLE_ADD_COLUMN_VALUE(
                        metaTable, "metas", metaPtr ? autil::StringUtil::toString(*metaPtr) : std::string());
                    break;
                }
            }
        }
    }
    return navi::SEC_OK;
}

REGISTER_BUILDIN_RESOURCE(GigMetaR);

}
