# 出包与镜像适配

## 从现有 Frontend 镜像构建

从 Whale 读取目标 Role 当前运行的镜像、架构、启动命令和 Python 路径。以同架构的完整 RTP 镜像 digest 为基底。根据源码选择 Hook 接口。提供 `install_from_config` 的镜像用 `current`，使用 `install_if_enabled` 且 visitor 由 `get_master_route_addrs` 直接路由的镜像用 `legacy`。入口与路由函数以 [configure.py](../../../mock-framework/scripts/traffic-capture/configure.py) 为准。

从仓库根目录运行：

```bash
capture=rtp_llm/flexlb/mock-framework/scripts/traffic-capture
docker buildx build --platform "$target_platform" \
  --build-arg FRONTEND_IMAGE="$base_image_at_digest" \
  --build-arg RTP_PYTHON="$frontend_python" \
  --build-arg RTP_CAPTURE_RUNTIME="$capture_runtime" \
  --tag "$capture_image" --load "$capture"
```

适配层将脚本和隔离安装的 Zstd 依赖放到 `/opt/rtp-frontend-capture`，生成 `http.json`、`grpc.json` 和 `both.json`。继承基础镜像的启动方式。构建阶段验证解释器、依赖与命令入口。使用与目标架构相同的构建环境。单架构本地验证后，通过已授权仓库发布带唯一标签的镜像，记录 adapter commit、基础 digest 和成品 digest。

在适配镜像内部用实际启动解释器加载所选 JSON，检查 `HotHookRuntime` 成功加载全部目标。启动目标进程并检查私网采集端口。窗口打开前验证普通健康探针及一个 P/D 路由请求。

```bash
"$frontend_python" -c 'import sys; from rtp_llm.utils.hot_hook_runtime import install_from_config; assert install_from_config(sys.argv[1])' \
  "/opt/rtp-frontend-capture/$capture_profile.json"
```

旧版镜像使用同一配置文件，并在探测命令和服务启动环境中设置 `RTP_HOT_HOOK=1` 和 `RTP_HOT_HOOK_FILE=/opt/rtp-frontend-capture/hook.py`。验证时调用 `install_if_enabled()`。

## Mock ViT 镜像

按 [Mock Engine 打包](../../../mock-framework/docs/whale-engine.md) 从同一 capture 分支构建 Engine 镜像，只更新承担 Mock ViT 的角色。只有部署需要单独 ViT 几何归档时才启用该角色。V4.1 Frontend 内联几何及纯文本采集可使用 `--vit-pods 0`。

## 源码出包

发布已提交版本的补丁层和 skill 时，可用单一 Git 归档固定内容：

```bash
git archive --format=tar.gz --prefix=rtp-frontend-capture/ \
  -o "$capture_bundle" HEAD \
  rtp_llm/flexlb/mock-framework/scripts/traffic-capture \
  rtp_llm/flexlb/skills/mock-traffic-capture
sha256sum "$capture_bundle"
```
