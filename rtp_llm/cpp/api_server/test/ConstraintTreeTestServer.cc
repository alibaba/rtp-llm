#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "http_server/HttpServer.h"
#include "rtp_llm/cpp/api_server/ConstraintTreeService.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: constraint_tree_test_server <port>" << std::endl;
        return 2;
    }

    const int port = std::atoi(argv[1]);
    if (port <= 0 || port > 65535) {
        std::cerr << "port must be between 1 and 65535" << std::endl;
        return 2;
    }

    auto service = std::make_shared<rtp_llm::ConstraintTreeService>();
    auto server  = std::make_shared<http_server::HttpServer>(nullptr, 2, 50, std::numeric_limits<int>::max());
    if (!server->RegisterRoute(
            "POST",
            "/update_constraint_tree",
            [service](std::unique_ptr<http_server::HttpResponseWriter> writer,
                      const http_server::HttpRequest& request) { service->updateConstraintTree(writer, request); })
        || !server->RegisterRoute(
            "GET",
            "/constraint_tree_status",
            [service](std::unique_ptr<http_server::HttpResponseWriter> writer,
                      const http_server::HttpRequest& request) { service->constraintTreeStatus(writer, request); })
        || !server->Start("tcp:127.0.0.1:" + std::to_string(port))) {
        std::cerr << "failed to start constraint-tree test server" << std::endl;
        return 1;
    }

    std::cout << "READY " << port << std::endl;
    std::string ignored;
    std::getline(std::cin, ignored);
    server->Stop();
    return 0;
}
