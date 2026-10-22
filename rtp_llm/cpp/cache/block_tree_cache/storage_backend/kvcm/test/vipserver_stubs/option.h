#pragma once
namespace middleware::vipclient {
class Option {
public:
    void set_failover_path(const char*) {}
    void set_log_path(const char*) {}
    void set_cache_path(const char*) {}
};
}  // namespace middleware::vipclient
