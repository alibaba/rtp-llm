#include "autil/result/Result.h"

#include <deque>
#include <experimental/source_location>
#include <sstream>
#include <string>
#include <typeinfo>

#include "autil/Demangle.h"

namespace autil::result {

std::string Error::get_stack_trace() const {
    std::ostringstream oss;
    oss << get_runtime_type_name() << ": " << message() << "\n";
    _locs.for_each([&](auto &loc) {
        oss << "  -> " << loc.function_name() << "() [" << loc.file_name() << ":" << loc.line() << "]\n";
    });
    return oss.str();
}

std::string Error::get_runtime_type_name() const { return demangle(typeid(*this).name()); }

} // namespace autil::result
