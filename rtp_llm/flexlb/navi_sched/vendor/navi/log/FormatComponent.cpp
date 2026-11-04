#include "navi/log/FormatComponent.h"

namespace navi {

StringComponent::StringComponent(const std::string& literal) : m_literal(literal) {}

void StringComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << m_literal;
}

void MessageComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.message;
}

void NameComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.name;
}

void ObjectComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.object;
}

void PrefixComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.prefix;
}

void BtComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.bt;
}

void DateComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    constexpr int32_t timeLen = 27;
    char buffer[timeLen];
    event.formatTime(buffer, timeLen);

    out << buffer;
}

void LevelComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.levelStr();
}

void ProcessIdComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.pid;
}

void ThreadIdComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.tid;
}

void FileComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.file;
}

void LineComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.line;
}

void FunctionComponent::append(std::ostringstream& out, const LoggingEvent& event)
{
    out << event.func;
}

}
