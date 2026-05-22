# RTP-LLM Codebase Learning Curriculum (Spec Pointer)

- **type**: brainstorming-output spec, pointer
- **date**: 2026-05-22
- **owner**: caihaowen.chw

## 目的

把用户"循序渐进融会贯通整个代码库"的诉求转成一份 15 节的可执行学习课程，标准是"学完成为合格的推理框架工程师，能执行各种各样的开发任务"。

## 设计本体

实际课程设计在：

**[`docs/learning/rtp-llm/00-curriculum.md`](../../learning/rtp-llm/00-curriculum.md)**

进度追踪与每节笔记在：

**[`docs/learning/rtp-llm/`](../../learning/rtp-llm/)**

## 为什么 spec 和操作笔记分离

- spec 是 "**为什么要这么学**"（设计哲学、决策记录、范围/非范围）
- 操作笔记是 "**实际怎么学**"（每节 lesson plan、学习者答案、错题本）
- 两者读者频次不同：spec 一次定型；操作笔记每节读写一次

## 关键决策摘要（详见 00-curriculum.md §4）

| # | 决策 | 一句话理由 |
|---|---|---|
| D1 | 自顶向下深度优先（方案 A），不走广度浅扫 | 硬验收解决"被问会懵" |
| D2 | B1.4 只讲 PD 分离，去掉 force_batch | 用户已熟 |
| D3 | 验收任务全部动手类，非画图/口述 | 目标升到"能执行开发任务" |
| D4 | 加 Capstone 第 15 节真实改动 PR | 全部知识在真实工作流走一遍 |
| D5 | 每节独立 `NN-<slug>.md` 文件 | 课程 doc 不膨胀，session 文件作错题本 |
| D6 | 写到 docs/learning/，superpowers/specs/ 下放 pointer | 操作与设计分离 |

## 流程定位

本 spec 来自 `superpowers:brainstorming` skill。标准的后续步骤是 `superpowers:writing-plans`，但本课程设计自身已经是 plan（按节执行的可操作大纲），因此跳过额外的 implementation plan 文档，直接进入"开课"执行。
