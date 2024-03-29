# Docker发布历史
包含cuda11和cuda12两个版本镜像
## 开发镜像
开发镜像已配置cuda，bazel等基础环境，但不包含已安装的wheel包，支持用户从源码开始构建/测试

###  0.0.1
```
cuda11: cn-hangzhou.aliyuncs.com/havenask/rtp_llm:cuda11
cuda12: cn-hangzhou.aliyuncs.com/havenask/rtp_llm:cuda12
``` 


## 服务镜像
包含基础镜像和已安装的wheel包，支持用户直接使用

### version 0.1.8 
relased on 2024.3.25
```
cuda11: registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:0.1.8_cuda11
cuda12: registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:0.1.8_cuda12
```