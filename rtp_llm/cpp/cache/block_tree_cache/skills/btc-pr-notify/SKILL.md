---
name: btc-pr-notify
description: 预览、部署或排查 RTP-LLM 的 BTC 钉钉群通知。用于目标分支为 main、改动涉及 block_tree_cache 的新 PR 和合入提醒。
---

# BTC PR 群通知

## 通知规则

监听 `alibaba/rtp-llm` 中目标分支为 `main` 的 PR。按 GitHub 的变更文件清单匹配 `rtp_llm/cpp/cache/block_tree_cache/` 前缀。文件移入、移出或删除都算。

| 群消息标题 | 触发条件 |
| --- | --- |
| `【BTC 新 PR】RTP-LLM #<编号>` | 新建的 PR 首次命中 BTC 目录，草稿也包括在内。最初没有改动 BTC 的 PR，后续推送命中时也会提醒。 |
| `【BTC 已合入】RTP-LLM #<编号>` | PR 合入 `main`，且最终改动仍命中 BTC 目录。 |

每条消息依次显示标题、PR 创建者的 GitHub 用户名、一至两句中文摘要和 GitHub 链接。摘要由 Qoder 根据该阶段能取到的全部 commit message 生成，重复提交说明会合并。机器人每 15 分钟检查一次，同一 PR 的新建和合入分别去重。启用前创建的 PR 后续合入时，仍可触发合入通知。

## 预览与排查

在运行定时任务的主机上，以部署用户执行。以下命令用同一个 PR 对照两种通知，只在终端显示结果。已合入 PR 的新建预览使用当前可见的提交记录。

```sh
cd ~/.local/share/rtp-btc-notify
python3 notify.py preview 1777 --kind opened
python3 notify.py preview 1777 --kind merged
```

若预期的消息未出现，先核对目标分支、事件时间和变更文件，再查定时器、日志及对应事件的投递状态。

```sh
systemctl --user list-timers --all rtp-btc-notify.timer
journalctl --user -u rtp-btc-notify.service -n 30 --no-pager
python3 notify.py status
sqlite3 state.sqlite3 'SELECT number,kind,state FROM notifications WHERE number=1777;'
```

`skip` 表示这次检查未命中；`new` 等待处理，`ready` 表示消息已生成，`sent` 表示钉钉已确认接收。若停在 `sending`，先在目标群按 PR 编号和标题核实是否送达，避免重复投递。

## 首次部署

从本 Skill 目录执行。运行主机需要 Python 3.6 或更高版本、Git、可用的 systemd 用户服务，以及安装在 `~/.local/bin/qodercli` 的 Qoder CLI。部署用户需能访问 GitHub、Qoder 和钉钉。

```sh
app="$HOME/.local/share/rtp-btc-notify"
units="$HOME/.config/systemd/user"
install -d -m 700 "$app" "$app/qoder-teams"
install -d "$units"
install -m 700 scripts/notify.py "$app/notify.py"
install -m 644 assets/rtp-btc-notify.service assets/rtp-btc-notify.timer "$units/"

QODER_CONFIG_DIR="$app/qoder-teams" "$HOME/.local/bin/qodercli" login
python3 "$app/notify.py" configure
# 若机器人启用了自定义关键词，将其中一个关键词单独写入 "$app/keyword"，再启用定时器。
python3 "$app/notify.py" init
systemctl --user daemon-reload
systemctl --user enable --now rtp-btc-notify.timer
```

通过 `configure` 的隐藏输入粘贴目标群机器人的 Webhook。Webhook、Qoder 登录状态和 SQLite 投递记录都保存在部署用户的运行目录。更新现有部署时只替换此 Skill 中的脚本和 unit 文件，保留运行目录中的凭据和投递记录。需要退出 SSH 后持续运行的主机，可用 `loginctl show-user "$USER" -p Linger` 检查用户服务是否常驻。

如果日志显示钉钉回执码 `310000` 且提示「关键词不匹配」，核对运行目录下的 `keyword` 文件与群机器人的安全设置。预览和正式投递使用同一段文本。修改关键词后，下一次检查会使用新配置重试已生成的消息。
