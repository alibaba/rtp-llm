#pragma once

#include <functional>
#include <memory>

namespace http_server {

using RequestAdmissionToken   = std::shared_ptr<void>;
using RequestAdmissionHandler = std::function<RequestAdmissionToken()>;

}  // namespace http_server
