# FlexLB

用户明确请求使用 `$flexlb-development` 时，读取仓库中的 [FlexLB 开发规范](skills/flexlb-development/SKILL.md)。

Java 成员顺序和 Javadoc 检查在提交时运行。也可提前检查暂存文件：

```bash
python3 -m pre_commit run flexlb-java-conventions --hook-stage pre-commit
```
