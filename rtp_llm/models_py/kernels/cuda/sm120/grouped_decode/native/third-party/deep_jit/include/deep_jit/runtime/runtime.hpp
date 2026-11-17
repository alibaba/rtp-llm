#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <deep_jit/cache/disk.hpp>
#include <deep_jit/cache/memory.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/lazy.hpp>
#include <deep_jit/utils/parser.hpp>

namespace deep_jit {

template <typename Backend>
class Runtime {
public:
    using Device = Backend::Device;
    using Kernel = Backend::Kernel;
    using CompilerOptions = Backend::CompilerOptions;
    using LaunchOptions = Backend::LaunchOptions;

    Config config;
    Env env;
    Device device;
    CompilerOptions default_compiler_options;
    LaunchOptions default_launch_options;
    DiskCache disk_cache;
    Backend backend;
    Parser parser;
    MemCache<std::string, Kernel> mem_cache;
    hash::FNV1a hash_base;

    explicit Runtime(Config config)
        : config(std::move(config)),
          env(this->config.env_prefix),
          default_compiler_options(CompilerOptions::default_options(env, device)),
          default_launch_options(LaunchOptions::default_options(env)),
          disk_cache(DiskCache::from_env(env)),
          backend(env),
          parser(this->config.include_dirs, this->config.include_prefixes) {
        // Hash contains 5 parts:
        //   - Extra signature (base, unchanged)
        //   - Compiler version (base, unchanged)
        //   - Compiler options
        //   - Post hook
        //   - Code (contains its includes)
        hash_base.update(this->config.extra_signature);
        hash_base.update(this->backend.compiler_info.get_hash());
    }

    std::shared_ptr<Kernel> compile(const std::string& name, const std::string& source, const CompilerOptions& override_options = {}) {
        const auto options = default_compiler_options.override_with(override_options);
        const auto key = cache_key(source, options);
        return mem_cache.get_or_create(key, [&] {
            return Backend::load(compile(name, source, key, options), env);
        });
    }

    std::filesystem::path compile_without_load(const std::string& name, const std::string& source, const CompilerOptions& override_options = {}) {
        const auto options = default_compiler_options.override_with(override_options);
        return compile(name, source, cache_key(source, options), options);
    }

    template <typename... Args>
    void launch(const std::shared_ptr<Kernel>& kernel, const LaunchOptions& override_options, const Args&... args) {
        DJ_HOST_ASSERT(kernel != nullptr, "kernel must not be null");
        kernel->launch(default_launch_options.override_with(override_options), args...);
    }

    std::filesystem::path compile(const std::string& name,
                                  const std::string& source,
                                  const std::string& key,
                                  const CompilerOptions& options) const {
        // Try disk cache firstly
        auto entry = disk_cache.entry(name, key);
        if (entry.hit)
            return entry.path;

        // Compile
        backend.compile(source, entry.path, env, config, options);

        // Publish
        GilScopedRelease gil_release;
        return entry.commit();
    }

    std::string cache_key(const std::string& source, const CompilerOptions& options) {
        auto hash = hash_base;
        options.update_hash(hash);
        return hash.update(options.get_post_hook_hash(config))
                   .update(parser.parse_into_hash(source))
                   .get_hex_digest();
    }
};

template <typename Backend>
inline LazyInit<Runtime<Backend>> create_lazy_jit(Config config) {
    return LazyInit<Runtime<Backend>>([config = std::move(config)] {
        return std::make_shared<Runtime<Backend>>(config);
    });
}

}  // namespace deep_jit
