#pragma once

#include <pybind11/pybind11.h>

void registerNaviPybind(pybind11::module_ &m);
void registerNaviRuntimeConfigPybind(pybind11::module_ &m);
