# model test
## 环境要求
* 镜像: reg.docker.alibaba-inc.com/isearch/py_dev:0.0.7
## 执行测试
### 简单测试(model只有两层 head等参数按比例裁剪)
```
bazel test //maga_transformer/test/model_test/fake_test:*
```

## 添加/运行单个case

### case目录说明
* testdata 存放case 数据
* data/{model_type}/ 每个model的fake和real的测试结果(hidden states)
* 如果需要新增测试用例，在all_fake_model_test.py中添加模型参数后，在model_test_base.py:ModelTestBase:_test_score下解注释save段即可储存运行结果
