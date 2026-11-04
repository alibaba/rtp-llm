#include "FormatComponent.h"

#include "LoggingEvent.h"

namespace alog {
StringComponent::StringComponent(const std::string &literal) : m_literal(literal) {}

void StringComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << m_literal; }

void MessageComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.message; }

void NameComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.loggerName; }

void DateComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.loggingTime; }

void ZoneDateComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.loggingZoneTime; }

void LevelComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.levelStr; }

void ProcessIdComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.pid; }

void ThreadIdComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.tid; }

void FileComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.file; }

void LineComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.line; }

void FunctionComponent::append(std::ostringstream &out, const LoggingEvent &event) { out << event.function; }

} // namespace alog
