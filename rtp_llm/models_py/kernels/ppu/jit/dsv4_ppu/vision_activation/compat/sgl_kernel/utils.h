#pragma once

// Retain RTP's qualified PPU host/toolchain support. The newer activation
// header additionally parses stream-style checks in its routed entry points.
#include "../../include/sgl_kernel/utils.h"

#ifndef CHECK_HOST
namespace host {

/** \brief Error class for stream-style error logging. */
struct Error {
  Error(DebugInfo location = {}) {
    m_oss << "Failed at " << location.file_name() << ":" << location.line() << ": ";
  }

  template <typename T>
  Error& operator<<(T&& arg) {
    m_oss << std::forward<T>(arg);
    return *this;
  }

  [[noreturn]]
  ~Error() noexcept(false) {
    throw PanicError(std::move(m_oss).str());
  }

 private:
  std::ostringstream m_oss;
};

/**
 * \brief 0-overhead CHECK macro for host code. This can avoid unnecessary
 * instantiation of error messages when the condition is true.
 *
 * Usage: CHECK_HOST(ptr != nullptr) << "Pointer must not be null";
 */
// The empty-true-branch if/else form keeps a trailing `else` in user code
// bound to the user's `if`, not to the macro's.
#define CHECK_HOST(COND) \
  if (COND) [[likely]] { \
  } else                 \
    host::Error()

}  // namespace host
#endif
