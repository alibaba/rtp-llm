#pragma once
#ifndef _GLIBCXX_PERMIT_BACKWARD_HASH
#define _GLIBCXX_PERMIT_BACKWARD_HASH
#endif
#include <algorithm>
#include <hash_map>

#undef _GLIBCXX_PERMIT_BACKWARD_HASH
#include <string.h>
#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {
class DataBuffer;

struct eqstr {
    bool operator()(const char *s1, const char *s2) const { return strcmp(s1, s2) == 0; }
};
typedef std::unordered_map<const char *, const char *, __gnu_cxx::hash<const char *>, eqstr> PSTR_MAP;
typedef PSTR_MAP::iterator PSTR_MAP_ITER;

class HttpRequestPacket : public Packet {
    friend class HttpRequestAndResponsePacketTest_testDecodeAndEncode_Test;

public:
    /*
     * 构造函数
     */
    HttpRequestPacket();

    /*
     * 析构函数
     */
    ~HttpRequestPacket();

    /*
     * 计算出数据包的长度
     */
    void countDataLen();
    /*
     * 组装
     */
    bool encode(DataBuffer *output);

    /*
     * 解开
     */
    bool decode(DataBuffer *input, PacketHeader *header);

    /*
     * 查询串
     */
    char *getQuery();

    /*
     * 是否keepalive
     */
    bool isKeepAlive();

    /*
     * 寻找其他头信息
     */
    const char *findHeader(const char *name);

private:
    char *_strHeader;    // 保存头内容的buffer
    char *_strQuery;     // 查询串
    bool _isKeepAlive;   // 是否支持keepalive
    int _method;         // get - 1
    int _version;        // http version 1-"HTTP/1.0"; 2="HTTP/1.1+"
    PSTR_MAP _headerMap; // 其他头信息的map
};

} // namespace anet

