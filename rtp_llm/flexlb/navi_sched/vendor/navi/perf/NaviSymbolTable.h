#pragma once

#include "navi/log/NaviLogger.h"
#include "navi/perf/NaviDso.h"
#include <link.h>

namespace navi {

class PerfSymbolTableDef;
class StringHashTable;
class StackTable;

struct MmapFile {
    size_t start;
    size_t end;
    std::string filePath;
    std::string entry;
};

class NaviSymbolTable
{
public:
    NaviSymbolTable();
    ~NaviSymbolTable();
private:
    NaviSymbolTable(const NaviSymbolTable &);
    NaviSymbolTable &operator=(const NaviSymbolTable &);
public:
    bool init(const NaviLoggerPtr &logger, const NaviSymbolTable *baseSymbolTable);
    NaviDsoPtr getDso(const std::string &filePath) const;
    void resolveSymbol(const StackTable &stackTable,
                       PerfSymbolTableDef *perfSymbolTableDef,
                       StringHashTable *stringHashTable) const;
    std::string resolveAddr(uint64_t ip, uint64_t *offset = nullptr) const;
private:
    const char *findSymbol(uint64_t addr, const char *&dso, bool &plt, uint64_t *offset) const;
    const char *findFileEntry(uint64_t addr) const;
    bool initElfVersion() const;
    void initFileMaps();
    bool initLoadLibs();
    void addDso(const NaviDsoPtr &dso);
    std::string getBinaryName() const;
    const MmapFile *getMmapFile(size_t addr) const;
    const NaviLoggerPtr &getLogger() const;
private:
    static int dlCallback(struct dl_phdr_info *info, size_t size, void *data);
private:
    DECLARE_LOGGER();
    const NaviSymbolTable *_baseSymbolTable = nullptr;
    std::map<size_t, MmapFile> _fileMaps;
    std::map<std::string, NaviDsoPtr> _dsoMap;
    std::vector<NaviDsoPtr> _dsoVec;
    uint64_t _totalSymbolCount;
};

NAVI_TYPEDEF_PTR(NaviSymbolTable);

}
