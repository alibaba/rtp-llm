#include "navi/engine/NaviSnapshotSummary.h"
#include "navi/engine/CreatorRegistry.h"
#include "navi/engine/ResourceCreator.h"
#include <fstream>
#include <rapidjson/prettywriter.h>

namespace navi {

void RegistryInfo::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    if (creator) {
        creator->Jsonize(json);
    }
}

void RegistrySummary::sort() {
    std::sort(infos.begin(), infos.end(),
              [](const RegistryInfo &lhs, const RegistryInfo &rhs) {
                  if (!lhs.creator || !rhs.creator) {
                      return true;
                  }
                  auto lhsResource = dynamic_cast<ResourceCreator *>(lhs.creator);
                  auto rhsResource = dynamic_cast<ResourceCreator *>(rhs.creator);
                  if (lhsResource && rhsResource) {
                      auto lhsStage = lhsResource->getStage();
                      auto rhsStage = rhsResource->getStage();
                      if (lhsStage != rhsStage) {
                          return lhsStage < rhsStage;
                      }
                  }
                  return lhs.creator->getName() < rhs.creator->getName();
              });
}

const std::string NaviSnapshotSummary::NAVI_VERSION = "navi-2.0.0";

void NaviSnapshotSummary::Jsonize(
    autil::legacy::Jsonizable::JsonWrapper &json) {
    if (json.GetMode() != TO_JSON) {
        NAVI_KERNEL_LOG(ERROR, "not supported");
        return;
    }
    json.Jsonize("stage", stage);
    if (config) {
        json.Jsonize("config", *config);
    }
    json.Jsonize("biz_summary", bizs);
    json.Jsonize("version", NAVI_VERSION);
}

void NaviSnapshotSummary::logSummary(const NaviSnapshotSummary &summary) {
    std::string summaryStr;
    if (!NaviConfig::prettyDumpToStr(summary, summaryStr)) {
        NAVI_KERNEL_LOG(ERROR,
                        "log navi snapshot summary failed, can't dump summary");
        return;
    }
    const auto &naviName = summary.naviName;
    std::string fileName;
    if (!naviName.empty()) {
        fileName = "./navi." + naviName + ".snapshot_summary";
    } else {
        fileName = "./navi.snapshot_summary";
    }
    writeFile(fileName, summaryStr);
}

void NaviSnapshotSummary::writeFile(const std::string &fileName,
                                    const std::string &content)
{
    std::ofstream out(fileName);
    if (!out.good()) {
        NAVI_KERNEL_LOG(ERROR, "write file [%s] content [%s] failed",
                        fileName.c_str(), content.c_str());
        return;
    }
    out << content;
    out.close();
}
}

