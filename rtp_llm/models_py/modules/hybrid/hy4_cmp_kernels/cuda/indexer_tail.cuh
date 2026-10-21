#pragma once
#include <ATen/ATen.h>
#include <cstdint>
void indexer_q_launch(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                     const at::Tensor&,const at::Tensor&,bool,bool,bool,std::uint64_t,
                     const at::Tensor&,const at::Tensor&,float);
void indexer_test_copy(const at::Tensor&,const at::Tensor&,bool,bool);
