#pragma once
namespace anet {
class FileControl {
public:
    static bool setCloseOnExec(int fd);
    static bool clearCloseOnExec(int fd);
    static bool isCloseOnExec(int fd);
};
} // namespace anet

