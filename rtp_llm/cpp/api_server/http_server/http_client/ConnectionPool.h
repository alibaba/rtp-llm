#pragma once

#include <mutex>
#include "aios/network/anet/anet.h"
namespace http_server {
typedef std::list<std::shared_ptr<anet::Connection>>           ConnectionList;
typedef std::list<std::shared_ptr<anet::Connection>>::iterator ConnectionListItr;

typedef std::map<std::string, std::shared_ptr<std::list<std::shared_ptr<anet::Connection>>>> ConnectionListMap;
typedef std::map<std::string, std::shared_ptr<std::list<std::shared_ptr<anet::Connection>>>>::iterator
    ConnectionListMapItr;

class ConnectionPool {
public:
    ConnectionPool();
    ~ConnectionPool();

public:
    std::shared_ptr<anet::Connection> makeHttpConnection(const std::string& address);
    void                              recycleHttpConnection(const std::string&                       address,
                                                            const std::shared_ptr<anet::Connection>& conn,
                                                            bool                                     close = false);

private:
    void                              createConnectionList(const std::string& address);
    std::shared_ptr<anet::Connection> createAndInsertBusyConnection(const std::string& address);
    void insertBusyConnection(const std::shared_ptr<anet::Connection>& conn, const std::string& address);
    std::shared_ptr<anet::Connection> getAndRemoveIdleConnection(const std::string& address);

private:
    std::shared_ptr<::anet::Transport> transport_{nullptr};

    std::mutex        connection_pool_mutex_;
    ConnectionListMap idle_connection_pool_;
    ConnectionListMap busy_connection_pool_;
};
}  // namespace http_server