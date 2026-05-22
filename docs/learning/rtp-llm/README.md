# RTP-LLM Codebase Learning — 学习笔记仓

学习者：caihaowen.chw  ·  起点：2026-05-22  ·  目标：合格推理框架工程师

> 课程设计哲学、单节模板、标杆问题清单见 **[00-curriculum.md](00-curriculum.md)**

## 进度

| # | 节 | 主题 | 状态 | 工时 | 笔记 |
|---|----|------|------|------|------|
| 1 | B1.1 | 入口与对象转换 | ☐ pending | - | - |
| 2 | B1.2 | Engine 主循环 | ☐ pending | - | - |
| 3 | B1.3 | FIFOScheduler + continuous batching | ☐ pending | - | - |
| 4 | B1.4 | PD 分离 | ☐ pending | - | - |
|   | -- | **B1 块末抽测** | ☐ pending | - | - |
| 5 | B2.1 | Executor 三件套 | ☐ pending | - | - |
| 6 | B2.2 | PyWrappedModel C++↔Py 桥 | ☐ pending | - | - |
| 7 | B2.3 | Qwen3 一遍 forward | ☐ pending | - | - |
|   | -- | **B2 块末抽测** | ☐ pending | - | - |
| 8 | B3.1 | Attention 多实现 | ☐ pending | - | - |
| 9 | B3.2 | Kernel 调用面 | ☐ pending | - | - |
| 10 | B3.3 | SM 分流 | ☐ pending | - | - |
| 11 | B3.4 | CUDA Graph + MTP | ☐ pending | - | - |
|   | -- | **B3 块末抽测** | ☐ pending | - | - |
| 12 | B4.1 | Loader 主流程 | ☐ pending | - | - |
| 13 | B4.2 | 量化变体 | ☐ pending | - | - |
| 14 | B4.3 | 构建系统全家桶 | ☐ pending | - | - |
|   | -- | **B1-B4 综合抽测** | ☐ pending | - | - |
| 15 | Cap | 真实小改动 PR | ☐ pending | - | - |

**图例**：☐ pending · 🟡 in-progress · ✅ completed · ⚠️ needs-retry

## 当前状态

🟡 **下一节：B1.1 入口与对象转换** — 等待 spec 用户复核通过后开课

## 快速指引

- **开新一节**：讲师按节号创建 `NN-<slug>.md`，按单节模板（curriculum §6）填 Section 1-3，然后开讲
- **节结束**：学习者填 Section 4，讲师批 Section 5-6，更新本表的状态/工时/笔记列
- **块末抽测**：在 `block-quiz-BN.md` 单独成文件
- **跳过/倒回**：在对应节笔记顶部记录决策与日期

## 关联文档

- **课程设计**：`00-curriculum.md`
- **Superpowers spec 指针**：`docs/superpowers/specs/2026-05-22-rtp-llm-codebase-learning-curriculum.md`
- **学习者背景 memory**：`~/.claude/projects/-data0-caihaowen-chw-RTP-LLM-github-opensource/memory/MEMORY.md`
