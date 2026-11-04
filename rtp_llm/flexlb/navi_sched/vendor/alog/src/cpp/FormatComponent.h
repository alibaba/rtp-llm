#pragma once

#include <sstream>
#include <string>

namespace alog {
struct LoggingEvent;

class FormatComponent {
public:
    inline virtual ~FormatComponent(){};
    virtual void append(std::ostringstream &out, const LoggingEvent &event) = 0;
};

class StringComponent : public FormatComponent {
public:
    StringComponent(const std::string &literal);

    virtual void append(std::ostringstream &out, const LoggingEvent &event);

private:
    std::string m_literal;
};

class MessageComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class NameComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class DateComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class ZoneDateComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class LevelComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class ProcessIdComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class ThreadIdComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class FileComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class LineComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

class FunctionComponent : public FormatComponent {
public:
    virtual void append(std::ostringstream &out, const LoggingEvent &event);
};

} // namespace alog
