#pragma once

#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/utils/ShapeCheck.h"


#include <optional>
#include <functional>
#include <algorithm>
#include <sstream>

namespace fastertransformer {


// target independence params check
void GemmParams::check() const {

    // check dim
    auto dim = A.dim();
    FT_CHECK_WITH_INFO((dim >= 1), 
                        "Gemm op param A dim %d should greater than 2.", A.dim());
    
    FT_CHECK_WITH_INFO((B.dim() == dim), 
                        "Gemm op param B dim %d should be equal to A", B.dim());
    
    if (C != std::nullopt) {
        auto c_dim = C.value().get().dim();
        FT_CHECK_WITH_INFO((C.value().get().dim() == dim), 
                            "Gemm op param C dim %d should be equal to A and B", c_dim);
    }

    if (dim > 2) {
        bool batch_dim_same = std::equal(A.shape().begin(), 
                                         A.shape().end() -2, 
                                         B.shape().begin(), 
                                         B.shape().end() -2);

        FT_CHECK_WITH_INFO(batch_dim_same, 
                           "Batch Gemm op A [%s] and B [%s] need batch shape same!",
                           ShapeStringView(A.shape()), ShapeStringView(B.shape()));
        
        if (C != std::nullopt) {
            bool batch_dim_same = std::equal(A.shape().begin(), 
                                             A.shape().end() -2, 
                                             C.value().get().shape().begin(), 
                                             C.value().get().shape().end() -2);
            FT_CHECK_WITH_INFO(batch_dim_same, 
                               "Batch Gemm op C [%s] need batch shape same!",
                               ShapeStringView(C.value().get().shape()));
        }
    }


    auto m_a = (transA == TransposeOperation::NONE) ? A.shape()[dim -2] : A.shape()[dim -1];
    auto k_a = (transA == TransposeOperation::NONE) ? A.shape()[dim -1] : A.shape()[dim -2];

    auto k_b = (transB == TransposeOperation::NONE) ? B.shape()[dim -2] : B.shape()[dim -1];
    auto n_b = (transB == TransposeOperation::NONE) ? B.shape()[dim -1] : B.shape()[dim -2];

    FT_CHECK_WITH_INFO((k_a == k_b), 
                        "Gemm op A (%s) [%s] need compact with B (%s) [%s]!",
                        enumToString(transA),
                        ShapeStringView(A.shape()),
                        enumToString(transB),
                        ShapeStringView(B.shape()));
    
    if (C != std::nullopt) {
        auto m_c = C.value().get().shape()[dim - 2];
        auto n_c = C.value().get().shape()[dim - 1];

        FT_CHECK_WITH_INFO((m_a == m_c) && (n_c == n_b), 
                          "Gemm op A (%s) [%s] and B (%s) [%s] need compact with C [%s]!",
                          enumToString(transA),
                          ShapeStringView(A.shape()),
                          enumToString(transB),
                          ShapeStringView(B.shape()),
                          ShapeStringView(C.value().get().shape()));
    }

}

}  // namespace fastertransformer
