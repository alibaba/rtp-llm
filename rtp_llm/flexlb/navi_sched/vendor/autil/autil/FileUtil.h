#pragma once

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace autil {

class FileUtil {
public:
    static std::string joinPath(const std::string &directory, const std::string &name) {
        return (std::filesystem::path(directory) / name).string();
    }

    static bool readFile(const std::string &path, std::string &content) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) {
            return false;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) {
            return false;
        }
        content.swap(data);
        return true;
    }

    static bool writeFile(const std::string &path, const std::string &content) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file.write(content.data(), content.size());
        file.close();
        return !file.fail();
    }

    static bool writeFileAtomically(const std::string &path, const std::string &content) {
        const auto temporary = path + "_tmp";
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file.write(content.data(), content.size());
        file.close();
        std::error_code error;
        if (!file.fail()) {
            std::filesystem::rename(temporary, path, error);
            if (!error) {
                return true;
            }
        }
        std::filesystem::remove(temporary, error);
        return false;
    }

    static bool makeDirs(const std::string &path) {
        std::error_code error;
        std::filesystem::create_directories(path, error);
        return !error;
    }

    static bool listDir(const std::string &path, std::vector<std::string> &names) {
        std::error_code error;
        std::vector<std::string> result;
        std::filesystem::directory_iterator entry(path, error), end;
        for (; !error && entry != end; entry.increment(error)) {
            result.push_back(entry->path().filename().string());
        }
        if (error) {
            return false;
        }
        names.swap(result);
        return true;
    }

    static bool removePath(const std::string &path) {
        std::error_code error;
        std::filesystem::remove_all(path, error);
        return !error;
    }
};

} // namespace autil
