#pragma once
#include <string>


// corresponds to cublasOperation_t
enum TransposeOperation {
    NONE,
    TRANSPOSE,
};

std::string inline enumToString(TransposeOperation type) {
    if (type == NONE) {
        return "NONE";
    } else {
        return "TRANSPOSE";
    }
};