#include "alog/Layout.h"

#include <sstream>
#include <stddef.h>
#include <string>
#include <vector>

#include "FormatComponent.h"
#include "LoggingEvent.h"

namespace alog {

std::string BasicLayout::format(const LoggingEvent &event) {
    std::ostringstream message;

    message << event.loggingTime << " " << event.levelStr << " " << event.loggerName << " : " << event.message
            << std::endl;
    return message.str();
}

std::string SimpleLayout::format(const LoggingEvent &event) {
    std::ostringstream message;
    message << event.loggingTime << " " << event.levelStr << " " << event.loggerName << " (" << event.pid << "/"
            << event.tid << ") : " << event.message << std::endl;
    return message.str();
}

std::string BinaryLayout::format(const LoggingEvent &event) { return event.message; }

PatternLayout::PatternLayout() {
    std::string defaultPattern = "%%d %%l %%c (%%p/%%t) : %%m";
    setLogPattern(defaultPattern);
}

PatternLayout::~PatternLayout() { clearLogPattern(); }

void PatternLayout::clearLogPattern() {
    for (ComponentVector::const_iterator i = m_components.begin(); i != m_components.end(); i++) {
        delete (*i);
    }
    m_components.clear();
    m_logPattern = "";
}

void PatternLayout::setLogPattern(const std::string &logPattern) {
    std::string literal;
    FormatComponent *component = NULL;
    clearLogPattern();
    for (std::string::const_iterator i = logPattern.begin(); i != logPattern.end(); i++) {
        if (*i == '%' && i + 2 != logPattern.end()) {
            char c = *(++i);
            if (c == '%') {
                char cc = *(++i);
                switch (cc) {
                case 'm':
                    component = new MessageComponent();
                    break;
                case 'c':
                    component = new NameComponent();
                    break;
                case 'd':
                    component = new DateComponent();
                    break;
                case 'z':
                    component = new ZoneDateComponent();
                    break;
                case 'l':
                    component = new LevelComponent();
                    break;
                case 'p':
                    component = new ProcessIdComponent();
                    break;
                case 't':
                    component = new ThreadIdComponent();
                    break;
                case 'F':
                    component = new FileComponent();
                    break;
                case 'f':
                    component = new FunctionComponent();
                    break;
                case 'n':
                    component = new LineComponent();
                    break;
                default:
                    literal += "%%";
                    literal += cc;
                }
                if (component) {
                    if (!literal.empty()) {
                        m_components.push_back(new StringComponent(literal));
                        literal = "";
                    }
                    m_components.push_back(component);
                    component = NULL;
                }
            } else {
                literal += '%';
                literal += c;
            }
        } else {
            literal += *i;
        }
    }
    if (!literal.empty()) {
        m_components.push_back(new StringComponent(literal));
    }
    m_logPattern = logPattern;
}

std::string PatternLayout::format(const LoggingEvent &event) {
    std::ostringstream message;

    for (ComponentVector::const_iterator i = m_components.begin(); i != m_components.end(); i++) {
        (*i)->append(message, event);
    }
    message << std::endl;
    return message.str();
}

} // namespace alog
