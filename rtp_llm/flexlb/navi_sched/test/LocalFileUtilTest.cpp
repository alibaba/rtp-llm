#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <stdlib.h>
#include <unistd.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "autil/FileRecorder.h"
#include "autil/FileUtil.h"

int main() {
    using autil::FileUtil;
    char pattern[] = "/tmp/navi-file-util-XXXXXX";
    char *temporary = mkdtemp(pattern);
    assert(temporary);
    const std::string root(temporary);
    const auto directory = FileUtil::joinPath(root, "nested/records");
    assert(FileUtil::makeDirs(directory));
    assert(FileUtil::makeDirs(directory));
    const auto path = FileUtil::joinPath(directory, "data.bin");
    const std::string binary("abc\0def\n", 8);
    std::string result;
    assert(FileUtil::writeFile(path, binary));
    assert(FileUtil::readFile(path, result) && result == binary);
    assert(FileUtil::writeFile(path, ""));
    assert(FileUtil::readFile(path, result) && result.empty());
    assert(FileUtil::writeFile(path, binary));
    assert(FileUtil::writeFile(path, "x"));
    assert(FileUtil::readFile(path, result) && result == "x");
    result = "unchanged";
    assert(!FileUtil::readFile(root + "/missing", result));
    assert(result == "unchanged");
    assert(!FileUtil::readFile(directory, result));
    assert(result == "unchanged");
    assert(!FileUtil::makeDirs(path));
    assert(!FileUtil::writeFile(root + "/missing/child", binary));
    assert(!FileUtil::writeFile(directory, binary));
    if (std::filesystem::exists("/dev/full")) {
        assert(!FileUtil::writeFile("/dev/full", binary));
    }

    std::vector<std::string> names;
    assert(FileUtil::listDir(directory, names));
    assert(names == std::vector<std::string>{"data.bin"});
    assert(!FileUtil::listDir(path, names));
    assert(names == std::vector<std::string>{"data.bin"});
    assert(!FileUtil::listDir(root + "/missing", names));

    assert(FileUtil::writeFileAtomically(path, binary));
    assert(FileUtil::readFile(path, result) && result == binary);
    assert(!std::filesystem::exists(path + "_tmp"));
    assert(FileUtil::makeDirs(path + "_tmp"));
    assert(!FileUtil::writeFileAtomically(path, "must not replace"));
    assert(FileUtil::readFile(path, result) && result == binary);
    assert(std::filesystem::is_directory(path + "_tmp"));
    assert(FileUtil::removePath(path + "_tmp"));
    assert(!FileUtil::writeFileAtomically(directory, binary));
    assert(std::filesystem::is_directory(directory));
    assert(!std::filesystem::exists(directory + "_tmp"));

    const auto link = root + "/data-link";
    std::filesystem::create_symlink(path, link);
    assert(FileUtil::readFile(link, result) && result == binary);
    assert(FileUtil::removePath(link));
    assert(FileUtil::readFile(path, result) && result == binary);

    const auto records = root + "/recorder";
    assert(FileUtil::makeDirs(records));
    assert(FileUtil::writeFile(records + "/keep.txt", "keep"));
    for (size_t id = 1; id <= 4; ++id) {
        autil::FileRecorder::newRecord(binary, records, "snapshot", id, 1);
    }
    assert(FileUtil::listDir(records, names));
    size_t snapshots = 0;
    for (const auto &name : names) {
        if (!autil::FileRecorder::hasSuffix(name, "snapshot")) continue;
        ++snapshots;
        assert(FileUtil::readFile(FileUtil::joinPath(records, name), result) && result == binary);
    }
    // FileRecorder keeps the previous keepCount records before adding the new one.
    assert(snapshots == 2);
    assert(FileUtil::readFile(records + "/keep.txt", result) && result == "keep");
    autil::FileRecorder::newRecord(binary, path, "snapshot", 5, 1);
    assert(FileUtil::readFile(path, result) && result == binary);
    assert(FileUtil::removePath(root));
    assert(!std::filesystem::exists(root));
    assert(FileUtil::removePath(root));
    std::cout << "Local FileUtil and FileRecorder checks passed\n";
}
