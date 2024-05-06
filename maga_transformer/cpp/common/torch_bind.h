#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>

#define DECLARE_TORCH_JIT_CLASS(class) \
    static auto class##THS = torch::jit::class_<class>("MagaTransformer", #class)

#define DECLARE_DEFAULT_CONSTRUCTOR(class) \
    .def(torch::init<>()) \

#define DECLARE_TORCH_JIT_CLASS_WITH_DEFAULT_CONSTRUCTOR(class) \
    DECLARE_TORCH_JIT_CLASS(class) \
    DECLARE_DEFAULT_CONSTRUCTOR(class)

#define ADD_TORCH_JIT_METHOD(class, method) \
    .def(#method, &class::method)

#define ADD_TORCH_JIT_PROPERTY(class, property) \
    .def_readwrite(#property, &class::property)
