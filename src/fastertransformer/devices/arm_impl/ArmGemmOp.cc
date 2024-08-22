#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>
#include "autil/StringUtil.h"

namespace fastertransformer {

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm_acl(const GemmParams& params) {

    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();
    batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;
    if (data_type != params.A.type()) {
        std::cout << "[Warning] GEMM compute type differs from input type. Not supported" << std::endl;
        data_type = params.A.type();
    }
    arm_compute::DataType acl_data_type = getAclDataType(data_type);

    arm_compute::TensorInfo src_data_info = arm_compute::TensorInfo(arm_compute::TensorShape(k, m), 1, acl_data_type);

    arm_compute::TensorInfo wei_data_info = arm_compute::TensorInfo(arm_compute::TensorShape(n, k), 1, acl_data_type);

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    arm_compute::NEGEMM gemm;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor output_tensor;

    src_tensor.allocator()->init(src_data_info);
    wei_tensor.allocator()->init(wei_data_info);
    output_tensor.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(n, m), 1, acl_data_type));

    arm_compute::NETranspose transA;
    arm_compute::NETranspose transB;
    arm_compute::Tensor      src_acc_tensor;
    arm_compute::Tensor      wei_acc_tensor;
    arm_compute::TensorInfo  src_acc_info;
    arm_compute::TensorInfo  wei_acc_info;

    bool is_transA = params.transA == TransposeOperation::TRANSPOSE;
    bool is_transB = params.transB == TransposeOperation::TRANSPOSE;

    src_acc_info = arm_compute::TensorInfo(arm_compute::TensorShape(m, k), 1, acl_data_type);
    wei_acc_info = arm_compute::TensorInfo(arm_compute::TensorShape(k, n), 1, acl_data_type);
    src_acc_tensor.allocator()->init(src_acc_info);
    wei_acc_tensor.allocator()->init(wei_acc_info);

    for (size_t batch = 0; batch < batch_size; batch++) {
        if (is_transA && !is_transB) {
            src_tensor.allocator()->allocate();
            src_acc_tensor.allocator()->import_memory(params.A.dataWithOffset(batch * m * k));
            transA.configure(&src_acc_tensor, &src_tensor);
            transA.run();
            wei_tensor.allocator()->import_memory(params.B.dataWithOffset(batch * n * k));
        } else if (is_transB && !is_transA) {
            wei_tensor.allocator()->allocate();
            wei_acc_tensor.allocator()->import_memory(params.B.dataWithOffset(batch * n * k));
            transB.configure(&wei_acc_tensor, &wei_tensor);
            transB.run();
            src_tensor.allocator()->import_memory(params.A.dataWithOffset(batch * m * k));
        } else if (is_transA && is_transB) {
            src_tensor.allocator()->allocate();
            src_acc_tensor.allocator()->import_memory(params.A.dataWithOffset(batch * m * k));
            wei_tensor.allocator()->allocate();
            wei_acc_tensor.allocator()->import_memory(params.B.dataWithOffset(batch * n * k));
            transA.configure(&src_acc_tensor, &src_tensor);
            transB.configure(&wei_acc_tensor, &wei_tensor);
            transA.run();
            transB.run();
        } else {
            src_tensor.allocator()->import_memory(params.A.dataWithOffset(batch * m * k));
            wei_tensor.allocator()->import_memory(params.B.dataWithOffset(batch * n * k));
        }
        output_tensor.allocator()->import_memory(output->dataWithOffset(batch * m * n));

        gemm.configure(&src_tensor, &wei_tensor, nullptr, &output_tensor, params.alpha, params.beta);
        gemm.run();
    }

    src_tensor.allocator()->free();
    wei_tensor.allocator()->free();

    return output;
}

}  // namespace fastertransformer
