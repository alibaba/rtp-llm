#include "autil/FileRecorder.h"

#include <algorithm>
#include <vector>

#include "autil/FileUtil.h"
#include "autil/TimeUtility.h"

using namespace std;

namespace autil {

const size_t FileRecorder::_maxCount = 200;

void FileRecorder::newRecord(
    const string &content, const string &dirname, const string &suffix, size_t id, size_t keepCount) {
    string dir = FileUtil::joinPath(".", dirname);
    if (!FileUtil::makeDirs(dir)) {
        return;
    }
    vector<string> allfiles;
    if (FileUtil::listDir(dir, allfiles)) {
        vector<string> records;
        for (size_t i = 0; i < allfiles.size(); i++) {
            if (hasSuffix(allfiles[i], suffix)) {
                records.push_back(allfiles[i]);
            }
        }
        if (records.size() > keepCount) {
            sort(records.begin(), records.end());
            for (size_t i = 0; i < records.size() - keepCount; i++) {
                string removeFile = FileUtil::joinPath(dir, records[i]);
                FileUtil::removePath(removeFile);
            }
        }
    }
    string filename =
        TimeUtility::currentTimeString("%Y-%m-%d-%H-%M-%S") + "_" + std::to_string(id) + string("_") + suffix;
    string finalPath = FileUtil::joinPath(dir, filename);
    FileUtil::writeFile(finalPath, content);
}

bool FileRecorder::hasSuffix(const string &str, const string &suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace autil
