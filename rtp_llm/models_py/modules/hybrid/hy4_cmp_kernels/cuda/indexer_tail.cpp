#include <torch/extension.h>
#include <string>
#include <array>
#include <optional>
#include "indexer_tail.cuh"
void run(const at::Tensor& q,const at::Tensor& positions,const at::Tensor& cs,
         const at::Tensor& rounded,const at::Tensor& q8,const at::Tensor& scale,
         bool rope,bool pdl,const std::string& release,std::uint64_t event,
         const std::optional<at::Tensor>& raw_gate,const std::optional<at::Tensor>& weighted_gate,
         float gate_scale) {
 TORCH_CHECK(release=="early" || release=="late","invalid release");
 TORCH_CHECK(q.is_cuda() && q.dim()==3 && q.size(0)>0 && q.size(0)<=256 && q.size(1)==32 && q.size(2)==128,
             "Q requires [M,32,128]");
 const int64_t m=q.size(0);
 TORCH_CHECK(positions.sizes()==at::IntArrayRef({m}),"positions require [M]");
 TORCH_CHECK(cs.dim()==2 && cs.size(0)>0 && cs.size(1)==64,"cos_sin requires [P,64]");
 TORCH_CHECK(rounded.sizes()==q.sizes() && q8.sizes()==q.sizes() && scale.sizes()==at::IntArrayRef({m,32,1}),
             "output shapes mismatch");
 TORCH_CHECK(scale.strides()==at::IntArrayRef({32,1,m*32}),"q_scale requires native unpacked strides [32,1,M*32]");
 std::array<at::Tensor,6> tensors={q,positions,cs,rounded,q8,scale};
 const at::ScalarType types[]={at::kBFloat16,at::kInt,at::kFloat,at::kBFloat16,at::kFloat8_e4m3fn,at::kFloat};
 for(int i=0;i<6;++i) {
  TORCH_CHECK(tensors[i].device()==q.device() && tensors[i].scalar_type()==types[i] && tensors[i].is_contiguous(),
              "requires same-device contiguous exact-dtype tensors");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr())%16==0,"base alignment requires16bytes");
  for(int j=0;j<i;++j) {
   auto a=reinterpret_cast<std::uintptr_t>(tensors[i].data_ptr()),b=reinterpret_cast<std::uintptr_t>(tensors[j].data_ptr());
   TORCH_CHECK(a+tensors[i].nbytes()<=b || b+tensors[j].nbytes()<=a,"input/output alias unsupported");
  }
 }
 TORCH_CHECK(raw_gate.has_value()==weighted_gate.has_value(),"raw and weighted gate must both be supplied");
 if(raw_gate) {
  TORCH_CHECK(raw_gate->sizes()==at::IntArrayRef({m,32}) && raw_gate->strides()==at::IntArrayRef({32,1}),"raw gate requires [M,32] rowmajor");
  TORCH_CHECK(weighted_gate->sizes()==at::IntArrayRef({m,32,1}) && weighted_gate->strides()==at::IntArrayRef({32,1,1}),"weighted gate requires native [M,32,1] strides");
  const std::array<at::Tensor,2> gates={*raw_gate,*weighted_gate};
  for(int i=0;i<2;++i) {
   TORCH_CHECK(gates[i].device()==q.device() && gates[i].scalar_type()==at::kFloat,"gate device/dtype mismatch");
   auto a=reinterpret_cast<std::uintptr_t>(gates[i].data_ptr());
   for(const auto& tensor:tensors) {
    auto b=reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(a+gates[i].nbytes()<=b || b+tensor.nbytes()<=a,"gate aliases Q input/output");
   }
  }
  auto a=reinterpret_cast<std::uintptr_t>(raw_gate->data_ptr()),b=reinterpret_cast<std::uintptr_t>(weighted_gate->data_ptr());
  TORCH_CHECK(a+raw_gate->nbytes()<=b || b+weighted_gate->nbytes()<=a,"raw/weighted gate overlap");
 }
 indexer_q_launch(q,positions,cs,rounded,q8,scale,rope,pdl,release=="early",event,
                  raw_gate.value_or(at::Tensor()),weighted_gate.value_or(at::Tensor()),gate_scale);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("run",&run,pybind11::arg("q"),pybind11::arg("positions"),pybind11::arg("cs"),
       pybind11::arg("rounded"),pybind11::arg("q8"),pybind11::arg("scale"),pybind11::arg("rope"),
       pybind11::arg("pdl"),pybind11::arg("release"),pybind11::arg("event"),
       pybind11::arg("raw_gate")=std::nullopt,pybind11::arg("weighted_gate")=std::nullopt,
       pybind11::arg("gate_scale")=1.0f);
 m.def("_test_copy",&indexer_test_copy);
}
