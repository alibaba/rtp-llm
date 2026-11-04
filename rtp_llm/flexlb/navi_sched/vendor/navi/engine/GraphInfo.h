#pragma once

#include "navi/config/NaviConfig.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

class GraphInfo
{
public:
    GraphInfo(const std::string &configPath, const GraphConfig &config);
    ~GraphInfo();
private:
    GraphInfo(const GraphInfo &);
    GraphInfo &operator=(const GraphInfo &);
public:
    bool init();
    const GraphDef &getGraphDef() const;
private:
    bool loadGraphDef();
private:
    std::string _configPath;
    GraphConfig _config;
    GraphDef _graphDef;
};

}
