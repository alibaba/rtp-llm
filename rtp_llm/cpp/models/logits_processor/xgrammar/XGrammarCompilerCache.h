#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#if RTP_LLM_ENABLE_XGRAMMAR_CPP
#include <xgrammar/xgrammar.h>
#endif

namespace rtp_llm {

struct XGrammarDeviceTable {
    std::string cache_key;
    std::string lowering_blob;
    size_t      bytes = 0;
#if RTP_LLM_ENABLE_XGRAMMAR_CPP
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_grammar;
#endif
};

class XGrammarCompilerCache {
public:
    static XGrammarCompilerCache& instance();

    void init(size_t capacity = 1024);
    size_t capacity() const;
    size_t size() const;
    size_t evictionCount() const;

    XGrammarDeviceTable getOrInsertDeviceTable(const std::string& cache_key, const std::string& lowering_blob);

private:
    XGrammarCompilerCache() = default;

    using LruList = std::list<XGrammarDeviceTable>;

    mutable std::mutex                                             mutex_;
    size_t                                                         capacity_ = 1024;
    size_t                                                         evictions_ = 0;
    LruList                                                        entries_;
    std::unordered_map<std::string, LruList::iterator>             index_;
};

}  // namespace rtp_llm
