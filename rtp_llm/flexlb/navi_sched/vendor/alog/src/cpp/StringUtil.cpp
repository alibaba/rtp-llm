#include "StringUtil.h"

#include <iterator>

using namespace std;

namespace alog {

string StringUtil::trim(const string &str) {
    static const char *whiteSpace = " \t\r\n";

    if (str.empty())
        return str;

    string::size_type beginPos = str.find_first_not_of(whiteSpace);
    if (beginPos == string::npos)
        return "";

    string::size_type endPos = str.find_last_not_of(whiteSpace);

    return string(str, beginPos, endPos - beginPos + 1);
}

unsigned int StringUtil::split(vector<string> &vec, const string &str, char delimiter, unsigned int maxSegments) {
    vec.clear();
    back_insert_iterator<vector<string>> it(vec);
    return split(it, str, delimiter, maxSegments);
}

} // namespace alog
