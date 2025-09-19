#include "rtp_llm/cpp/api_server/http_server/http_client/ConnectionPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace http_server {
static anet::HTTPPacketFactory packet_factory;
static anet::HTTPStreamer      packet_streamer(&packet_factory);

ConnectionPool::ConnectionPool() {
    transport_ = std::make_shared<::anet::Transport>();
    transport_->start();
}

ConnectionPool::~ConnectionPool() {
    auto clearConnectionPool = [](ConnectionListMap& connection_pool) {
        ConnectionListMapItr iter;
        for (iter = connection_pool.begin(); iter != connection_pool.end(); ++iter) {
            auto              tmp_list = iter->second;
            ConnectionListItr tmp_iter;
            for (tmp_iter = tmp_list->begin(); tmp_iter != tmp_list->end(); ++tmp_iter) {
                (*tmp_iter).reset();
            }
        }
        connection_pool.clear();
    };
    clearConnectionPool(idle_connection_pool_);
    clearConnectionPool(busy_connection_pool_);
    if (transport_) {
        transport_->stop();
        transport_->wait();
        transport_.reset();
    }
}

void ConnectionPool::createConnectionList(const std::string& address) {
    ConnectionList* idle_list = new ConnectionList();
    ConnectionList* busy_list = new ConnectionList();
    idle_connection_pool_.insert(std::make_pair(address, idle_list));
    busy_connection_pool_.insert(std::make_pair(address, busy_list));
}

std::shared_ptr<anet::Connection> ConnectionPool::createAndInsertBusyConnection(const std::string& address) {
    auto conn            = transport_->connect(address.c_str(), &packet_streamer, false);
    auto conn_shared_ptr = std::shared_ptr<anet::Connection>(conn, [](anet::Connection* connection) {
        if (connection) {
            connection->close();
            connection->subRef();
        }
    });
    insertBusyConnection(conn_shared_ptr, address);
    return conn_shared_ptr;
}

void ConnectionPool::insertBusyConnection(const std::shared_ptr<anet::Connection>& conn, const std::string& address) {
    if (!conn) {
        return;
    }
    auto busy_iter = busy_connection_pool_.find(address);
    if (busy_iter == busy_connection_pool_.end()) {
        return;
    }
    busy_iter->second->push_back(conn);
}

std::shared_ptr<anet::Connection> ConnectionPool::getAndRemoveIdleConnection(const std::string& address) {
    auto idle_iter = idle_connection_pool_.find(address);
    // never create connection for this address, create idle and busy connection list
    if (idle_iter == idle_connection_pool_.end()) {
        createConnectionList(address);
        return nullptr;
    }
    // have no idle connection now
    auto idle_conn_list = idle_iter->second;
    if (idle_conn_list->empty()) {
        return nullptr;
    }
    // have idle connection
    auto idle_conn = idle_conn_list->front();
    idle_conn_list->pop_front();
    if (idle_conn && !idle_conn->isClosed()) {
        insertBusyConnection(idle_conn, address);
    }

    return idle_conn;
}

std::shared_ptr<anet::Connection> ConnectionPool::makeHttpConnection(const std::string& address) {
    std::lock_guard<std::mutex> lock(connection_pool_mutex_);
    // find idle connection
    auto conn = getAndRemoveIdleConnection(address);
    if (conn == nullptr || conn->isClosed()) {
        return createAndInsertBusyConnection(address);
    }
    return conn;
}

void ConnectionPool::recycleHttpConnection(const std::string&                       address,
                                           const std::shared_ptr<anet::Connection>& conn,
                                           bool                                     close) {
    ConnectionListMapItr idle_iter;
    ConnectionListMapItr busy_iter;
    {
        std::lock_guard<std::mutex> lock(connection_pool_mutex_);
        idle_iter = idle_connection_pool_.find(address);
        busy_iter = busy_connection_pool_.find(address);
        if (idle_iter == idle_connection_pool_.end() || busy_iter == busy_connection_pool_.end()) {
            RTP_LLM_LOG_WARNING("recycle http connection failed, address %s is not in connection pool",
                                address.c_str());
            return;
        }
        auto idle_list = idle_iter->second;
        auto busy_list = busy_iter->second;

        if (!conn) {
            RTP_LLM_LOG_WARNING("recycle http connection failed, connection is null");
            return;
        }
        if (close && !conn->isClosed()) {
            conn->close();
        }

        busy_list->remove(conn);
        if (!conn->isClosed()) {
            idle_list->push_back(conn);
        }
    }
}
}  // namespace http_server