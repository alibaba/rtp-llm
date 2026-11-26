#pragma once
#include <stdexcept>

#include "aios/navi_sched/resource/TairKvcmGrpcClientR.h"
#include "aios/network/gig/multi_call/service/GrpcConnectionPool.h"
#include "aios/network/gig/multi_call/service/SearchServiceManager.h"

namespace navi_sched { namespace test {

std::shared_ptr<navi::GigClientR> unaryGigClient(size_t threads = 4)
{
    auto gig = std::make_shared<navi::GigClientR>();
    gig->_searchService = std::make_shared<multi_call::SearchService>();
    multi_call::MultiCallConfig config;
    config.connectConf["grpc"].threadNum = threads;
    config.miscConf.disableMetricReport = true;
    if (!gig->_searchService->init(config)) throw std::runtime_error("test GIG unary init failed");
    return gig;
}

std::shared_ptr<TairKvcmGrpcClientR> gigQueryClient(int port, const std::string &host = "127.0.0.1")
{
    auto gig = unaryGigClient();
    auto client = std::shared_ptr<TairKvcmGrpcClientR>(new TairKvcmGrpcClientR(),
        [gig](TairKvcmGrpcClientR *value) { delete value; });
    client->_gigClientR = gig.get();
    client->_topology.address = host; client->_topology.port = port;
    client->_topology.discovery.type = "direct";
    if (!client->configureTransport()) throw std::runtime_error("test unary Resource configuration failed");
    return client;
}

std::shared_ptr<multi_call::GrpcConnectionPool> unaryPool(const TairKvcmGrpcClientR &client)
{
    return std::dynamic_pointer_cast<multi_call::GrpcConnectionPool>(client._gigClientR->getSearchService()
        ->getSearchServiceManager()->getConnectionManager()->getConnectionPool(multi_call::MC_PROTOCOL_GRPC));
}

} }
