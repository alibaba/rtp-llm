#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <deep_jit/backend/ascend/device.hpp>
#include <deep_jit/backend/ascend/kernel.hpp>
#include <deep_jit/backend/ascend/options.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit {

class Ascend {
public:
    using Device = ascend::Device;
    using Kernel = ascend::Kernel;
    using CompilerOptions = ascend::CompilerOptions;
    using LaunchOptions = ascend::LaunchOptions;

    struct CompilerInfo {
        std::filesystem::path path;
        std::string version;
        std::filesystem::path linker_path;
        std::string linker_version;

        [[nodiscard]] std::string get_hash() const {
            return hash::FNV1a().update(version).update(linker_version).get_hex_digest();
        }

        [[nodiscard]] json to_json() const {
            return json::object_t {
                {"path", path.string()},
                {"version", version},
                {"linker_path", linker_path.string()},
                {"linker_version", linker_version},
            };
        }
    };

    struct Toolkit {
        std::filesystem::path bisheng;
        std::filesystem::path ld_lld;
        std::filesystem::path asc_devkit_include;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;

    explicit Ascend(const Env&)
        : toolkit(find_ascend_toolkit()),
          compiler_info(get_compiler_info()) {}

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        return {
            .path = toolkit.bisheng,
            .version = call_external_command(toolkit.bisheng.string() + " --version"),
            .linker_path = toolkit.ld_lld,
            .linker_version = call_external_command(toolkit.ld_lld.string() + " --version"),
        };
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Paths
        dir = std::filesystem::absolute(dir).lexically_normal();
        const auto source_path = dir / "kernel.asc";
        const auto relocatable_path = dir / "kernel.rel.o";
        const auto binary_path = dir / "kernel.o";
        const bool print_compiler_command = env.get<bool>("JIT_DEBUG", false) or
                                            env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

        // Write source code
        write_file_sync(source_path, source);

        // Build commands
        std::vector<std::string> base_bisheng_args = {toolkit.bisheng.string()};
        const auto bisheng_flags = options.get_bisheng_flags();
        base_bisheng_args.insert(base_bisheng_args.end(), bisheng_flags.begin(), bisheng_flags.end());
        for (const auto& include_dir: config.include_dirs)
            base_bisheng_args.emplace_back("-I" + include_dir.string());
        base_bisheng_args.emplace_back("-I" + toolkit.asc_devkit_include.string());

        auto bisheng_args = base_bisheng_args;
        bisheng_args.insert(bisheng_args.end(), {
            source_path.string(),
            "-o",
            relocatable_path.string(),
        });
        const auto bisheng_command = str::join(bisheng_args);

        std::vector<std::string> linker_args = {toolkit.ld_lld.string()};
        const auto linker_flags = options.get_linker_flags();
        linker_args.insert(linker_args.end(), linker_flags.begin(), linker_flags.end());
        linker_args.insert(linker_args.end(), {
            relocatable_path.string(),
            "-o",
            binary_path.string(),
        });
        const auto linker_command = str::join(linker_args);

        // Compile and link
        const auto cd_command = "cd " + dir.string() + " && ";
        call_external_command(cd_command + bisheng_command, print_compiler_command);
        call_external_command(cd_command + linker_command, print_compiler_command);
        safe_remove_all(relocatable_path);
        DJ_HOST_ASSERT(std::filesystem::is_regular_file(binary_path) and std::filesystem::file_size(binary_path) != 0,
                       "Bisheng did not produce a valid Ascend binary: {}", binary_path.string());

        // Dump assembly
        if (options.dump_asm.value_or(false)) {
            const auto asm_dir = dir / "asm";
            make_dirs(asm_dir);
            auto asm_args = base_bisheng_args;
            asm_args.insert(asm_args.end(), {
                "-save-temps",
                source_path.string(),
                "-o",
                (asm_dir / "kernel.rel.o").string(),
            });
            call_external_command("cd " + asm_dir.string() + " && " + str::join(asm_args),
                                  print_compiler_command);
        }

        // Write metadata
        const json metadata = json::object_t {
            {"command", bisheng_command + " && " + linker_command},
            {"config", config.to_json()},
            {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()},
        };
        write_file_sync(dir / "meta.json", metadata.dump());
    }

    [[nodiscard]] static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        return Kernel::load(dir, env);
    }

    static Toolkit find_ascend_toolkit() {
        // Find Ascend home
        std::filesystem::path home_path = get_env<std::string>("ASCEND_HOME_PATH");
        if (home_path.empty())
            home_path = get_env<std::string>("ASCEND_TOOLKIT_HOME");
        if (home_path.empty() and std::filesystem::exists("/usr/local/Ascend/ascend-toolkit/latest"))
            home_path = "/usr/local/Ascend/ascend-toolkit/latest";
        if (home_path.empty() and std::filesystem::exists("/usr/local/Ascend/cann"))
            home_path = "/usr/local/Ascend/cann";

        DJ_HOST_ASSERT(not home_path.empty() and std::filesystem::is_directory(home_path),
                       "Ascend toolkit home was not found");
        home_path = std::filesystem::absolute(home_path).lexically_normal();

        const auto bisheng = home_path / "bin/bisheng";
        const auto ld_lld = home_path / "bin/ld.lld";
        DJ_HOST_ASSERT(is_executable(bisheng), "Bisheng compiler is not executable: {}", bisheng.string());
        DJ_HOST_ASSERT(is_executable(ld_lld), "Ascend linker is not executable: {}", ld_lld.string());

        const auto asc_devkit_include = home_path / "aarch64-linux/asc/include/adv_api";
        DJ_HOST_ASSERT(std::filesystem::is_directory(asc_devkit_include),
                       "Ascend adv_api include directory was not found: {}", asc_devkit_include.string());
        return {
            .bisheng = bisheng,
            .ld_lld = ld_lld,
            .asc_devkit_include = asc_devkit_include,
        };
    }
};

}  // namespace deep_jit
