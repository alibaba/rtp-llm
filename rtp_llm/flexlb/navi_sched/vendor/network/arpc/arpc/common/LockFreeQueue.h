#pragma once
#include "autil/LockFreeQueue.h"

namespace arpc::common {

template <typename T>
using LockFreeQueue = ::autil::LockFreeQueue<T>;

}
