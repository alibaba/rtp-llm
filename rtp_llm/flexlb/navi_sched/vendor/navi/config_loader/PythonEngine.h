#pragma once

#include "Python.h"
#include <string>

namespace navi {

class PythonEngine
{
public:
    PythonEngine(const std::string &naviPythonPath);
    ~PythonEngine();
private:
    PythonEngine(const PythonEngine &);
    PythonEngine &operator=(const PythonEngine &);
public:
    bool exec(const std::string &configLoader,
              const std::string &configPath,
              const std::string &loadParam,
              std::string &result);
private:
    std::string getErrorInfo();
private:
    static void setDontWriteBytecode();
    static std::string getExternalDir(const std::string &naviPythonPath);
    static std::string getObjectStr(PyObject *obj);
    static std::string getString(const char *str);
private:
    bool _needFinalizePython = false;
    std::string _pathStr;
    std::wstring _path;
private:
    static std::string NAVI_CONFIG_INIT_MODULE;
    static std::string NAVI_CONFIG_INIT_FUNCTION;
};

}
