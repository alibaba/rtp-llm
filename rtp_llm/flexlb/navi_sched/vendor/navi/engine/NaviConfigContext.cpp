#include "navi/engine/NaviConfigContext.h"
#include "navi/engine/StringHashTable.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/util/CommonUtil.h"
#include "navi/proto/NaviSummary.pb.h"

using namespace autil::legacy;

namespace navi {

NaviConfigContext::NaviConfigContext() {
}

NaviConfigContext::NaviConfigContext(
        const std::string &configPath,
        const RapidValue *jsonAttrs,
        const RapidValue *jsonConfig,
        const NodeDef *nodeDef,
        const autil::legacy::RapidValue *jsonConfig2)
    : _configPath(&configPath)
    , _jsonAttrs(jsonAttrs)
    , _jsonConfig(jsonConfig)
    , _nodeDef(nodeDef)
    , _jsonConfig2(jsonConfig2)
{
}

NaviConfigContext::NaviConfigContext(const std::string &configPath,
                                     const RapidValue *jsonAttrs,
                                     const NodeDef *nodeDef)
    : _configPath(&configPath)
    , _jsonAttrs(jsonAttrs)
    , _jsonConfig(nullptr)
    , _nodeDef(nodeDef)
    , _jsonConfig2(nullptr)
{
}

NaviConfigContext::~NaviConfigContext() {
}

std::map<std::string, std::string> NaviConfigContext::getBinaryAttrs() const {
    static std::map<std::string, std::string> defaultAttrs;
    if (!_nodeDef) {
        return defaultAttrs;
    }
    const auto &attrMap = _nodeDef->binary_attrs();
    if (attrMap.empty()) {
        return defaultAttrs;
    }
    std::map<std::string, std::string> binaryAttrs;
    for (const auto &attr : attrMap) {
        binaryAttrs.emplace(attr.first, attr.second);
    }
    return binaryAttrs;
}

bool NaviConfigContext::getIntegerArrayAttr(int32_t index, int64_t &value) const {
    if (!_nodeDef) {
        return false;
    }
    if (index >= _nodeDef->integer_array_attrs_size()) {
        return false;
    }
    value = _nodeDef->integer_array_attrs(index);
    return true;
}

NaviConfigContext NaviConfigContext::enter(const std::string &key) {
    const RapidValue *jsonAttrs = nullptr;
    Jsonize(key, jsonAttrs);
    return NaviConfigContext(*_configPath, jsonAttrs, _nodeDef);
}

const std::string &NaviConfigContext::getConfigPath() const {
    return *_configPath;
}

bool NaviConfigContext::isArray() const {
    if (!_jsonAttrs) {
        return false;
    }
    return _jsonAttrs->IsArray();
}

size_t NaviConfigContext::size() const {
    if (!_jsonAttrs) {
        return 0;
    }
    return _jsonAttrs->Size();
}

NaviConfigContext NaviConfigContext::enter(size_t index) {
    if (_jsonAttrs) {
        return NaviConfigContext(*_configPath, &(*_jsonAttrs)[index], _nodeDef);
    }

    return NaviConfigContext(*_configPath, nullptr, _nodeDef);
}

bool NaviConfigContext::hasKey(const std::string &key) const {
    return hasJsonKey(key) || hasBinaryKey(key) || hasIntegerKey(key);
}

void NaviConfigContext::JsonizeAsString(const std::string &key,
                                        std::string &value)
{
    const RapidValue *jsonValue = nullptr;
    Jsonize(key, jsonValue);
    if (jsonValue) {
        value = FastToJsonString(*jsonValue);
    }
}

void NaviConfigContext::JsonizeAsString(const std::string &key,
                                        std::string &value,
                                        const std::string &defaultValue)
{
    const RapidValue *jsonValue = nullptr;
    Jsonize(key, jsonValue, jsonValue);
    if (jsonValue) {
        value = FastToJsonString(*jsonValue);
    } else {
        value = defaultValue;
    }
}

bool NaviConfigContext::hasJsonKey(const std::string &key) const {
    return (_jsonAttrs && _jsonAttrs->HasMember(key.c_str())) || (_jsonConfig && _jsonConfig->HasMember(key.c_str())) ||
           (_jsonConfig2 && _jsonConfig2->HasMember(key.c_str()));
}

bool NaviConfigContext::hasBinaryKey(const std::string &key) const {
    if (!_nodeDef) {
        return false;
    }
    const auto &attrMap = _nodeDef->binary_attrs();
    return attrMap.end() != attrMap.find(key);
}

template <>
bool NaviConfigContext::JsonizeBinary<std::string>(const std::string &key,
                                                   std::string &value)
{
    if (!_nodeDef) {
        return false;
    }
    const auto &attrMap = _nodeDef->binary_attrs();
    auto it = attrMap.find(key);
    if (attrMap.end() != it) {
        value = it->second;
        return true;
    }
    return false;
}

bool NaviConfigContext::hasIntegerKey(const std::string &key) const {
    if (!_nodeDef) {
        return false;
    }
    const auto &attrMap = _nodeDef->integer_attrs();
    return attrMap.end() != attrMap.find(key);
}

template <>
bool NaviConfigContext::JsonizeInteger<int64_t>(const std::string &key,
                                                int64_t &value)
{
    if (!_nodeDef) {
        return false;
    }
    const auto &attrMap = _nodeDef->integer_attrs();
    auto it = attrMap.find(key);
    if (attrMap.end() != it) {
        value = it->second;
        return true;
    }
    return false;
}

template <>
bool NaviConfigContext::JsonizeBuildin(const std::string &key, std::string &value) {
    if (NAVI_BUILDIN_ATTR_NODE == key) {
        if (_nodeDef) {
            value = _nodeDef->name();
            return true;
        } else {
            AUTIL_LEGACY_THROW(autil::legacy::NotJsonizableException,
                               std::string("invalid buildin Jsonize key [") + key + "], not supported");
        }
    } else if (NAVI_BUILDIN_ATTR_KERNEL == key) {
        if (_nodeDef) {
            value = _nodeDef->kernel_name();
            return true;
        } else {
            AUTIL_LEGACY_THROW(autil::legacy::NotJsonizableException,
                               std::string("invalid buildin Jsonize key [") + key + "], not supported");
        }
    }
    return false;
}

void NaviJsonizeInfo::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("member", _memberName);
    json.Jsonize("member_type", CommonUtil::parseSymbol(_memberType.c_str()));
    json.Jsonize("json_key", _jsonKey);
    json.Jsonize("has_default", _hasDefault);
}

void NaviJsonizeInfo::fillProto(StringHashTable &stringHashTable, NaviJsonizeInfoDef &def) const {
    def.set_member_name_hash(stringHashTable.getHash(_memberName));
    def.set_member_type_hash(stringHashTable.getHash(CommonUtil::parseSymbol(_memberType.c_str())));
    def.set_function_name_hash(stringHashTable.getHash(_functionName));
    def.set_json_key_hash(stringHashTable.getHash(_jsonKey));
    def.set_has_default(_hasDefault);
    def.set_file_hash(stringHashTable.getHash(_sourceFile));
    def.set_line(_line);
}

std::ostream &operator<<(std::ostream &os, const NaviConfigContext &ctx) {
    os << "NaviConfigContext ";
    if (ctx._jsonAttrs) {
        os << "attr: " << FastToJsonString(*ctx._jsonAttrs) << std::endl;
    }
    if (ctx._jsonConfig) {
        os << "config: " << FastToJsonString(*ctx._jsonConfig) << std::endl;
    }
    if (ctx._jsonConfig2) {
        os << "config2: " << FastToJsonString(*ctx._jsonConfig2) << std::endl;
    }
    auto nodeDef = ctx._nodeDef;
    if (nodeDef) {
        const auto &binaryAttrMap = nodeDef->binary_attrs();
        os << "binary attrs: " << std::endl;
        for (const auto &pair : binaryAttrMap) {
            os << " " << pair.first << ": <...>" << std::endl;
        }
        const auto &integerAttrMap = nodeDef->integer_attrs();
        os << "integer attrs: " << std::endl;
        for (const auto &pair : integerAttrMap) {
            os << " " << pair.first << ": " << pair.second << std::endl;
        }
        const auto &integerArrayAttrs = nodeDef->integer_array_attrs();
        os << "integer array attrs: " << std::endl;
        for (const auto &value : integerArrayAttrs) {
            os << " " << value;
        }
        std::cout << std::endl;
    }
    return os;
}

}
