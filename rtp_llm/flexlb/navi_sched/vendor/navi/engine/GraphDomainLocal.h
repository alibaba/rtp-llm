#pragma once

#include "navi/engine/GraphDomain.h"

namespace navi {

class GraphDomainLocal : public GraphDomain
{
public:
    GraphDomainLocal(Graph *graph);
    ~GraphDomainLocal();
private:
    GraphDomainLocal(const GraphDomainLocal &);
    GraphDomainLocal &operator=(const GraphDomainLocal &);
public:
    bool run() override;
};

}
