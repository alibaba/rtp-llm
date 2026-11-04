#include "navi/engine/GraphDomainLocal.h"

namespace navi {

GraphDomainLocal::GraphDomainLocal(Graph *graph)
    : GraphDomain(graph, GDT_LOCAL)
{
}

GraphDomainLocal::~GraphDomainLocal() {
}

bool GraphDomainLocal::run() {
    return true;
}

}
