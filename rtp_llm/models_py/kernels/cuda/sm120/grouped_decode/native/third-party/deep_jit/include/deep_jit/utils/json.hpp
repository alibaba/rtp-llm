#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace deep_jit {

class json {
public:
    using array_t = std::vector<json>;
    using object_t = std::vector<std::pair<std::string, json>>;
    using value_t = std::variant<std::nullptr_t, bool, int64_t, uint64_t, std::string, array_t, object_t>;

    value_t value = nullptr;

    json() noexcept = default;
    json(std::nullptr_t) noexcept : value(nullptr) {}
    template <typename T>
        requires(std::is_same_v<std::remove_cv_t<T>, bool>)
    json(const T value) noexcept : value(value) {}
    template <typename T>
        requires(std::is_integral_v<T> and std::is_signed_v<T> and not std::is_same_v<std::remove_cv_t<T>, bool>)
    json(const T value) noexcept : value(static_cast<int64_t>(value)) {}
    template <typename T>
        requires(std::is_integral_v<T> and std::is_unsigned_v<T> and not std::is_same_v<std::remove_cv_t<T>, bool>)
    json(const T value) noexcept : value(static_cast<uint64_t>(value)) {}
    json(const char* value) : value(std::string(value)) {}
    json(std::string value) : value(std::move(value)) {}
    json(array_t value) : value(std::move(value)) {}
    json(object_t value) : value(std::move(value)) {}

    template <typename T>
    json(const std::optional<T>& value) : json(value ? json(*value) : json()) {}

    template <typename T>
    json(const std::vector<T>& values) : value(array_t{}) {
        auto& array = std::get<array_t>(value);
        array.reserve(values.size());
        for (const auto& item : values)
            array.emplace_back(item);
    }

    [[nodiscard]] std::string dump() const {
        std::string output;
        dump_to(output);
        return output;
    }

    static void dump_string(std::string& output, const std::string& value) {
        static constexpr char hex[] = "0123456789abcdef";
        output.push_back('"');
        for (const unsigned char c : value) {
            switch (c) {
                case '"':
                    output += "\\\"";
                    break;
                case '\\':
                    output += "\\\\";
                    break;
                case '\b':
                    output += "\\b";
                    break;
                case '\f':
                    output += "\\f";
                    break;
                case '\n':
                    output += "\\n";
                    break;
                case '\r':
                    output += "\\r";
                    break;
                case '\t':
                    output += "\\t";
                    break;
                default:
                    if (c < 0x20) {
                        output += "\\u00";
                        output.push_back(hex[c >> 4]);
                        output.push_back(hex[c & 0x0f]);
                    } else {
                        output.push_back(static_cast<char>(c));
                    }
            }
        }
        output.push_back('"');
    }

    void dump_to(std::string& output) const {
        if (std::holds_alternative<std::nullptr_t>(value)) {
            output += "null";
        } else if (std::holds_alternative<bool>(value)) {
            output += std::get<bool>(value) ? "true" : "false";
        } else if (std::holds_alternative<int64_t>(value)) {
            output += std::to_string(std::get<int64_t>(value));
        } else if (std::holds_alternative<uint64_t>(value)) {
            output += std::to_string(std::get<uint64_t>(value));
        } else if (std::holds_alternative<std::string>(value)) {
            dump_string(output, std::get<std::string>(value));
        } else if (std::holds_alternative<array_t>(value)) {
            output.push_back('[');
            bool first = true;
            for (const auto& item : std::get<array_t>(value)) {
                if (not first)
                    output.push_back(',');
                first = false;
                item.dump_to(output);
            }
            output.push_back(']');
        } else {
            output.push_back('{');
            bool first = true;
            for (const auto& [key, item] : std::get<object_t>(value)) {
                if (not first)
                    output.push_back(',');
                first = false;
                dump_string(output, key);
                output.push_back(':');
                item.dump_to(output);
            }
            output.push_back('}');
        }
    }
};

}  // namespace deep_jit
