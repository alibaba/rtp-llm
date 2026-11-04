#pragma once

#include "navi/common.h"
#include "navi/log/NaviLogger.h"
#include <gelf.h>

namespace navi {

struct ElfSymbolTable {
    size_t index;
    GElf_Shdr shdr;
    Elf_Scn *sec = nullptr;
    Elf_Data *syms = nullptr;
    Elf_Data *symStrs = nullptr;
};

struct DsoSymbol {
    uint64_t start;
    uint64_t end;
    uint32_t size;
    bool plt;
    uint64_t nameOffset;
    const char *strTable = nullptr;
};

class NaviDso;
NAVI_TYPEDEF_PTR(NaviDso);

class NaviDso
{
public:
    NaviDso(const std::string &path, uint64_t base, uint64_t start,
            uint64_t end);
    ~NaviDso();
private:
    NaviDso(const NaviDso &);
    NaviDso &operator=(const NaviDso &);
public:
    bool init(const NaviLoggerPtr &logger);
    const char *findSymbol(uint64_t addr, bool &plt, uint64_t *offsetPtr) const;
    bool match(uint64_t addr) const {
        if (addr < _start + _base || addr > _end + _base) {
            return false;
        } else {
            return true;
        }
    }
    bool isSame(uint64_t base, uint64_t start, uint64_t end) const {
        return _base == base && _start == start && _end == end;
    }
    const std::string &getPath() const;
    uint64_t getSymbolCount() const;
private:
    bool loadElf();
    bool loadSymbolTable(const char *name, ElfSymbolTable &elfSymTab);
    bool readSymbolTable();
    bool readPltSymbols();
    Elf_Scn *elf_section_by_name(Elf *elf, GElf_Ehdr *ep, GElf_Shdr *shp,
                                 const char *name, size_t *idx);
private:
    static bool compareSymbols(const DsoSymbol &left, const DsoSymbol &right);
    static bool compareSymbolAndAddr(uint64_t addr, const DsoSymbol &symbol);
    static bool getVdsoRange(uint64_t &begin, uint64_t &end);
public:
    static bool compareDsos(const NaviDsoPtr &left, const NaviDsoPtr &right);
    static bool compareDsoAndAddr(uint64_t addr, const NaviDsoPtr &dso);
private:
    DECLARE_LOGGER();
    std::string _path;
    uint64_t _base;
    uint64_t _start;
    uint64_t _end;
    int32_t _fd;
    Elf *_elf;
    GElf_Ehdr _ehdr;
    ElfSymbolTable _symTab;
    ElfSymbolTable _dynSymTab;
    std::vector<DsoSymbol> _symbolVec;
    bool _hasPltSec = false;
};

}

