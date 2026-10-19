# aclnn_custom_ops — 自定义 AscendC 算子目录

本目录用于存放 rtp-llm 用到的**自定义 AscendC 算子**源码与构建脚本。

历史说明：本目录原存放 ApplyTopKTopPCustom 算子，起该算子已替换为 CANN
内置算子 ApplyTopKTopPWithSorted（aclnn 接口 aclnnApplyTopKTopP，声明头
$ASCEND_HOME_PATH/include/aclnnop/aclnn_apply_top_k_top_p.h，CANN >= 9.0.0 内置）。

## 如何接入一个新的自定义 AscendC 算子

1. 【本目录】放置算子工程：每个算子一个子目录（op_host/ op_kernel/ 及工程 CMake 脚手架），
   或放置完整自定义算子仓的源码；如需 bazel 参与构建，恢复 BUILD + .bzl 构建规则
   （参考 git 历史中本目录的 build_aclnn_ops 写法）；
2. 【部署】按 CANN 自定义算子 run 包方式安装：
   bash <vendor>.run --install-path=$ASCEND_HOME_PATH --quiet
   （vendor 落在 $ASCEND_HOME_PATH/vendors/<vendor_name>/，含 libcust_opapi.so 与 opp/）
3. 【bindings 层】rtp_llm/models_py/bindings/ascend/ops/ 下新增：
   - aclnn_<op>.h：算子两段式接口声明（aclnnXxxGetWorkspaceSize / aclnnXxx），
     若 CANN 已内置同名算子可直接 include 官方头 aclnnop/aclnn_<op>.h；
   - ascend_<op>.h：torch 包装函数，核心一行 EXEC_NPU_CMD(aclnnXxx, ...)；
4. 【运行时】source $ASCEND_HOME_PATH/vendors/<vendor_name>/bin/set_env.bash
   （设置 ASCEND_CUSTOM_OPP_PATH 与 LD_LIBRARY_PATH）。

## 调用机制（无需改动）

EXEC_NPU_CMD（op_api_common.h）按符号名动态查找：优先 dlopen libcust_opapi.so
（自定义/vendor 算子），找不到自动 fallback 到内置 libopapi.so（CANN 内置算子）。
因此自定义与内置算子对调用方形态完全一致，仅命中的库不同。
