#pragma once

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <sys/wait.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

inline std::string call_external_command(std::string command, const bool print_command = false) {
    DJ_HOST_ASSERT(not command.empty(), "command must not be empty");
    if (print_command)
        std::printf("Running command: %s\n", command.c_str());

    command += " 2>&1";
    const auto deleter = [](FILE* file) {
        if (file != nullptr)
            pclose(file);
    };
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(command.c_str(), "r"), deleter);
    DJ_HOST_ASSERT(pipe != nullptr, "failed to run command: {}", command);

    std::array<char, 512> buffer;
    std::string output;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        output += buffer.data();

    // NOTES: if the child was killed by a signal (e.g., SIGINT from Ctrl+C),
    // WEXITSTATUS would incorrectly return 0. Treat signal death as failure.
    const auto status = pclose(pipe.release());
    const auto exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    DJ_HOST_ASSERT(exit_code == 0, "command failed with exit code {}:\n{}\n{}", exit_code, command, output);
    return output;
}

}  // namespace deep_jit
