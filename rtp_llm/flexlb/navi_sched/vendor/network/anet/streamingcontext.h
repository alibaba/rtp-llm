#pragma once
namespace anet {
class Packet;

class StreamingContext {
public:
    StreamingContext();
    virtual ~StreamingContext();

    virtual bool isCompleted();
    void setCompleted(bool compleled);
    virtual bool isBroken();
    void setBroken(bool broken);
    void setEndOfFile(bool eof);
    bool isEndOfFile();
    Packet *getPacket();

    virtual void setErrorNo(int errorNo);
    int getErrorNo();
    void setErrorString(const char *errorString);
    const char *getErrorString();

    /**
     * get the packet from context, then set _packet to NULL.
     **/
    virtual Packet *stealPacket();
    void setPacket(Packet *packet);
    virtual void reset();

protected:
    Packet *_packet;
    bool _completed;
    bool _broken;
    bool _eof;
    int _errorNo;
    const char *_errorString;
};

} /*end namespace anet*/
