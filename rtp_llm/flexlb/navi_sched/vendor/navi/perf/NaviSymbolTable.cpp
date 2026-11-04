#include "navi/perf/NaviSymbolTable.h"
#include "navi/engine/StringHashTable.h"
#include "navi/engine/StackTable.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/util/CommonUtil.h"
#include <algorithm>
#include <autil/StringUtil.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fstream>
#include <libelf.h>
#include <limits.h>
#include <stdlib.h>

namespace navi {

NaviSymbolTable::NaviSymbolTable()
    : _totalSymbolCount(0)
{
}

NaviSymbolTable::~NaviSymbolTable() {
    _dsoMap.clear();
    _dsoVec.clear();
}

bool NaviSymbolTable::init(const NaviLoggerPtr &logger, const NaviSymbolTable *baseSymbolTable) {
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix("navi symbol table");
    if (!initElfVersion()) {
        return false;
    }
    initFileMaps();
    _baseSymbolTable = baseSymbolTable;
    if (!initLoadLibs()) {
        _baseSymbolTable = nullptr;
        return false;
    }
    _baseSymbolTable = nullptr;
    NAVI_LOG(
        INFO,
        "init navi symbol table success, lib count: %lu, symbol count: %lu",
        _dsoVec.size(), _totalSymbolCount);
    return true;
}

bool NaviSymbolTable::initElfVersion() const {
    if (elf_version(EV_CURRENT) == EV_NONE) {
        return false;
    }
    return true;
}

void NaviSymbolTable::initFileMaps() {
    auto file = fopen("/proc/self/maps", "r");
    if (!file) {
        return;
    }
    while (true) {
        char buffer[4096];
        char prot[5];
        char execName[4096];
        if (!fgets(buffer, sizeof(buffer), file)) {
            break;
        }
        strcpy(execName, "");
        uint64_t start, end, pgoff, maj, min, inode;
        /* 00400000-0040c000 r-xp 00000000 fd:01 41038  /bin/cat */
        auto n = sscanf(
            buffer, "%lx-%lx %s %lx %lx:%lx %lu %[^\n]\n", &start, &end, prot, &pgoff, &maj, &min, &inode, execName);
        if (n < 8) {
            continue;
        }
        MmapFile mmapFile;
        mmapFile.start = start;
        mmapFile.end = end;
        mmapFile.filePath = execName;
        mmapFile.entry = buffer;
        _fileMaps.emplace(start, mmapFile);
        NAVI_KERNEL_LOG(DEBUG, "file maps: 0x%lx-0x%lx, entry: %s", start, end, buffer);
    }
    fclose(file);
}

const MmapFile *NaviSymbolTable::getMmapFile(size_t addr) const {
    auto it = _fileMaps.find(addr);
    if (_fileMaps.end() == it) {
        return nullptr;
    }
    return &it->second;
}

std::string NaviSymbolTable::getBinaryName() const {
    char buff[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
    if (len != -1) {
        buff[len] = '\0';
        return std::string(buff);
    } else {
        return std::string();
    }
}

const NaviLoggerPtr &NaviSymbolTable::getLogger() const {
    return _logger.logger;
}

int NaviSymbolTable::dlCallback(struct dl_phdr_info *info, size_t size, void *data) {
    NaviSymbolTable *symbolTable = (NaviSymbolTable *)data;
    std::string filePath;
    if (info->dlpi_name) {
        filePath = info->dlpi_name;
    }
    auto base = info->dlpi_addr;
    if (filePath.empty()) {
        if (0 == base) {
            filePath = symbolTable->getBinaryName();
        } else {
            auto mmapFile = symbolTable->getMmapFile(base);
            if (!mmapFile) {
                filePath = "[vdso]";
            } else {
                filePath = mmapFile->filePath;
            }
        }
    }
    auto phnum = info->dlpi_phnum;
    uint64_t minAddr = std::numeric_limits<uint64_t>::max();
    uint64_t maxAddr = 0;
    for (size_t i = 0; i < phnum; i++) {
        const auto &phdr = info->dlpi_phdr[i];
        auto start = phdr.p_vaddr;
        auto end = start + phdr.p_memsz;
        minAddr = std::min(minAddr, start);
        maxAddr = std::max(maxAddr, end);
    }
    NaviDsoPtr dso;
    if (symbolTable->_baseSymbolTable) {
        auto oldDso = symbolTable->_baseSymbolTable->getDso(filePath);
        if (oldDso && oldDso->isSame(base, minAddr, maxAddr)) {
            dso = oldDso;
        }
    }
    if (!dso) {
        dso = std::make_shared<NaviDso>(filePath, base, minAddr, maxAddr);
        if (!dso->init(symbolTable->getLogger())) {
            return 0;
        }
    }
    symbolTable->addDso(dso);
    return 0;
}

void NaviSymbolTable::addDso(const NaviDsoPtr &dso) {
    _totalSymbolCount += dso->getSymbolCount();
    _dsoMap[dso->getPath()] = dso;
    _dsoVec.push_back(dso);
}

NaviDsoPtr NaviSymbolTable::getDso(const std::string &filePath) const {
    auto it = _dsoMap.find(filePath);
    if (_dsoMap.end() == it) {
        return nullptr;
    }
    return it->second;
}

bool NaviSymbolTable::initLoadLibs() {
    dl_iterate_phdr(dlCallback, this);
    std::sort(_dsoVec.begin(), _dsoVec.end(), NaviDso::compareDsos);
    NAVI_LOG(INFO, "dso count: %lu", _dsoVec.size());
    return true;
}

const char *NaviSymbolTable::findSymbol(uint64_t addr, const char *&dso,
                                        bool &plt, uint64_t *offset) const
{
    auto it = std::upper_bound(_dsoVec.begin(), _dsoVec.end(), addr,
                               NaviDso::compareDsoAndAddr);
    if (_dsoVec.end() == it) {
        return nullptr;
    }
    auto naviDso = *it;
    auto ret = naviDso->findSymbol(addr, plt, offset);
    if (ret) {
        dso = naviDso->getPath().c_str();
    }
    return ret;
}

const char *NaviSymbolTable::findFileEntry(uint64_t addr) const {
    auto it = _fileMaps.lower_bound(addr);
    if (_fileMaps.end() == it) {
        return nullptr;
    }
    const auto &mmapFile = it->second;
    if (mmapFile.start <= addr && mmapFile.end > addr) {
        return it->second.entry.c_str();
    } else {
        return nullptr;
    }
}

void NaviSymbolTable::resolveSymbol(
    const StackTable &stackTable,
    PerfSymbolTableDef *perfSymbolTableDef,
    StringHashTable *stringHashTable) const
{
    const auto &addrMap = stackTable.getAddrMap();
    int32_t symbolCount = addrMap.size();
    auto symbolVecDef = perfSymbolTableDef->mutable_symbols();
    symbolVecDef->Reserve(symbolCount);
    for (size_t i = 0; i < symbolCount; i++) {
        symbolVecDef->Add();
    }
    for (const auto &pair : addrMap) {
        auto ip = pair.first;
        uint64_t offset = 0;
        auto symbolStr = resolveAddr(ip, &offset);
        auto symbolDef = symbolVecDef->Mutable(pair.second);
        symbolDef->set_ip(ip);
        symbolDef->set_symbol(stringHashTable->getHash(std::move(symbolStr)));
        symbolDef->set_offset(offset);
    }
}

std::string NaviSymbolTable::resolveAddr(uint64_t ip, uint64_t *offset) const {
    const char *dso = nullptr;
    bool plt = false;
    auto symbol = findSymbol(ip, dso, plt, offset);
    if (!symbol) {
        symbol = findFileEntry(ip);
    }
    std::string symbolStr;
    if (symbol) {
        symbolStr = CommonUtil::parseSymbol(symbol);
        if (plt) {
            symbolStr += "@plt";
        }
    } else {
        char buffer[128];
        autil::StringUtil::uint64ToHexStr(ip, buffer, sizeof(buffer));
        symbolStr = std::string("0x") + buffer;
    }
    return symbolStr;
}

}
