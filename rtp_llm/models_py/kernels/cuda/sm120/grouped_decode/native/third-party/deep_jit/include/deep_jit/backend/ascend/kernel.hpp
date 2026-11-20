#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <deep_jit/backend/ascend/driver.hpp>
#include <deep_jit/backend/ascend/options.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/gil.hpp>

namespace deep_jit::ascend {

template <typename T>
inline constexpr size_t kernel_arg_alignment = alignof(T) < 4 ? 4 : alignof(T);

template <typename T>
constexpr size_t align_kernel_arg(const size_t offset) {
    return (offset + kernel_arg_alignment<T> - 1) & ~(kernel_arg_alignment<T> - 1);
}

// Immutable Ascend kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
public:
    aclrtBinHandle binary_handle{};
    aclrtFuncHandle kernel_handle{};

    Kernel() = default;

    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    static std::vector<std::string> parse_kernel_names(const std::filesystem::path& path) {
        // Read header
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        DJ_HOST_ASSERT(input.is_open(), "failed to open Ascend binary: {}", path.string());
        const auto file_size = input.tellg();
        DJ_HOST_ASSERT(file_size >= static_cast<std::streamoff>(sizeof(Elf64_Ehdr)),
                       "invalid Ascend binary: {}", path.string());
        input.seekg(0);

        Elf64_Ehdr header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        DJ_HOST_ASSERT(input and std::memcmp(header.e_ident, ELFMAG, SELFMAG) == 0,
                       "not an ELF file: {}", path.string());
        DJ_HOST_ASSERT(header.e_ident[EI_CLASS] == ELFCLASS64 and header.e_ident[EI_DATA] == ELFDATA2LSB,
                       "unsupported ELF format: {}", path.string());
        DJ_HOST_ASSERT(header.e_shentsize == sizeof(Elf64_Shdr) and header.e_shnum > 0,
                       "invalid ELF section table: {}", path.string());

        // Read sections
        const auto section_table_size = static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr);
        DJ_HOST_ASSERT(header.e_shoff <= static_cast<uint64_t>(file_size) and
                       section_table_size <= static_cast<uint64_t>(file_size) - header.e_shoff,
                       "truncated ELF section table: {}", path.string());
        std::vector<Elf64_Shdr> sections(header.e_shnum);
        input.seekg(static_cast<std::streamoff>(header.e_shoff));
        input.read(reinterpret_cast<char*>(sections.data()), static_cast<std::streamsize>(section_table_size));
        DJ_HOST_ASSERT(input, "failed to read ELF section table: {}", path.string());

        // Read section names
        const auto string_table_index = header.e_shstrndx == SHN_XINDEX
            ? sections.front().sh_link
            : header.e_shstrndx;
        DJ_HOST_ASSERT(string_table_index < sections.size(),
                       "invalid ELF string table index: {}", path.string());
        const auto& string_table_section = sections[string_table_index];
        DJ_HOST_ASSERT(string_table_section.sh_offset <= static_cast<uint64_t>(file_size) and
                       string_table_section.sh_size <= static_cast<uint64_t>(file_size) - string_table_section.sh_offset,
                       "truncated ELF string table: {}", path.string());

        std::string string_table(string_table_section.sh_size, '\0');
        input.seekg(static_cast<std::streamoff>(string_table_section.sh_offset));
        input.read(string_table.data(), static_cast<std::streamsize>(string_table.size()));
        DJ_HOST_ASSERT(input, "failed to read ELF string table: {}", path.string());

        // Collect kernel names
        constexpr std::string_view prefix = ".ascend.meta.";
        std::vector<std::string> kernel_names;
        for (const auto& section: sections) {
            if (section.sh_name >= string_table.size())
                continue;
            const auto end = string_table.find('\0', section.sh_name);
            if (end == std::string::npos)
                continue;
            std::string_view name(string_table.data() + section.sh_name, end - section.sh_name);
            if (not name.starts_with(prefix))
                continue;

            name.remove_prefix(prefix.size());
            for (const std::string_view suffix: {"_mix_aic", "_mix_aiv"}) {
                if (name.ends_with(suffix)) {
                    name.remove_suffix(suffix.size());
                    break;
                }
            }
            kernel_names.emplace_back(name);
        }
        std::ranges::sort(kernel_names);
        kernel_names.erase(std::ranges::unique(kernel_names).begin(), kernel_names.end());
        return kernel_names;
    }

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Check existence
        const auto binary_path = dir / "kernel.o";
        DJ_HOST_ASSERT(std::filesystem::is_regular_file(binary_path),
                       "missing Ascend binary: {}", binary_path.string());

        // Record start time
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        if (debug)
            std::fputs(std::format("Loading Ascend binary: {}\n", binary_path.string()).c_str(), stdout);
        const auto start_time = std::chrono::steady_clock::now();

        // Load kernel
        const auto kernel_names = parse_kernel_names(binary_path);
        DJ_HOST_ASSERT(kernel_names.size() == 1,
                       "expected exactly one kernel in {}, found {}",
                       binary_path.string(), kernel_names.size());

        auto kernel = std::make_shared<Kernel>();
        DJ_ACL_CHECK(driver::lazy_aclrtBinaryLoadFromFile(binary_path.c_str(), nullptr, &kernel->binary_handle));
        DJ_ACL_CHECK(driver::lazy_aclrtBinaryGetFunction(
            kernel->binary_handle, kernel_names.front().c_str(), &kernel->kernel_handle));

        // Print and return
        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(std::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return kernel;
    }

    template <typename... Args>
    void launch(const LaunchOptions& launch_options, const Args&... args) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Checks
        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        DJ_HOST_ASSERT(launch_options.num_blocks.has_value() and *launch_options.num_blocks > 0,
                       "Ascend block count must be positive");
        DJ_HOST_ASSERT(launch_options.num_ubuf_bytes.has_value() and *launch_options.num_ubuf_bytes >= 0,
                       "Ascend dynamic UB size must not be negative");
        DJ_HOST_ASSERT(launch_options.num_launch_timeout_secs.has_value() and
                       *launch_options.num_launch_timeout_secs >= 0 and
                       *launch_options.num_launch_timeout_secs <= std::numeric_limits<uint16_t>::max(),
                       "Ascend launch timeout is out of range");

        // Pack arguments
        constexpr size_t parameter_size = [] {
            size_t offset = 0;
            ((offset = align_kernel_arg<Args>(offset) + sizeof(Args)), ...);
            return (offset + 7) & ~static_cast<size_t>(7);
        }();
        std::array<uint8_t, parameter_size> parameter_buffer{};
        size_t offset = 0;
        ((offset = align_kernel_arg<Args>(offset),
          std::memcpy(parameter_buffer.data() + offset, &args, sizeof(args)),
          offset += sizeof(args)), ...);

        // Set launch attributes
        std::array<aclrtLaunchKernelAttr, 2> attributes{};
        size_t num_attributes = 0;
        if (*launch_options.num_ubuf_bytes > 0) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
            attribute.value.dynUBufSize = static_cast<uint32_t>(*launch_options.num_ubuf_bytes);
        }
        if (*launch_options.num_launch_timeout_secs > 0) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
            attribute.value.timeout = static_cast<uint16_t>(*launch_options.num_launch_timeout_secs);
        }

        // Launch
        aclrtLaunchKernelCfg config{};
        aclrtLaunchKernelCfg* config_ptr = nullptr;
        if (num_attributes > 0) {
            config.attrs = attributes.data();
            config.numAttrs = num_attributes;
            config_ptr = &config;
        }

        const aclrtStream stream = launch_options.stream
            ? *launch_options.stream
            : c10_npu::getCurrentNPUStream();
        DJ_ACL_CHECK(driver::lazy_aclrtLaunchKernelWithHostArgs(
            kernel_handle, *launch_options.num_blocks, stream, config_ptr,
            sizeof...(Args) == 0 ? nullptr : parameter_buffer.data(),
            parameter_size, nullptr, 0));

        // Launch blocking
        if (get_env<bool>("ASCEND_LAUNCH_BLOCKING", false))
            DJ_ACL_CHECK(driver::lazy_aclrtSynchronizeDevice());
    }

    void unload() noexcept {
        if (binary_handle == nullptr)
            return;

        try {
            DJ_ACL_CHECK(driver::lazy_aclrtBinaryUnLoad(binary_handle));
        } catch (...) {
        }
        binary_handle = nullptr;
        kernel_handle = nullptr;
    }
};

}  // namespace deep_jit::ascend
