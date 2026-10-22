# 独立 Mock 启动入口

`standalone_mock.py` 在一个 Pod 内启动一个 Mock JVM。部署模式统一设置
`MOCK_DEPLOYMENT_MODE=single_pod` 或 `per_engine`；前者承载全部逻辑 P/D，
后者每 Pod 仅承载一个 P 或 D。

配置、兼容变量、端口、VipServer 注册与健康检查契约统一见
[Whale Mock 配置](../online_eval/docs/whale/configuration.md)。
入口生成运行配置后以 `exec` 启动 Java，由 JVM 直接接收容器终止信号。
