#include "rtp_llm/cpp/devices/ShapeCheck.h"

namespace rtp_llm {

using Shape = std::vector<size_t>;

bool CheckShapeConsistent(const std::vector<Shape>& shape_list) {
    if (shape_list.size() == 0) {
        return true;
    }
    auto same_shape = shape_list[0];
    for (auto shape : shape_list) {
        if (shape == same_shape) {
            return false;
        }
    }
    return true;
}

std::string ShapeStringView(const Shape& shape) {
    std::string s;
    s = s + '(';
    for (size_t i = 0; i < shape.size(); i++) {
        s = s + std::to_string(shape[i]) + ',';
    }
    s = s + ')';
    return s;
}

}  // namespace rtp_llm
