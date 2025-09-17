# 添加文档手册

## 1. 添加文档

## 2. 添加索引

## 3. 增加中文

1. 进入 docs 目录。
2. 运行命令 `make update`。此命令会提取所有可翻译的消息并更新 PO 翻译文件。
3. 手动修改新增的翻译文件，文件位于 docs/locales/zh_CN/LC_MESSAGES 下，补全对应语言的 msgstr 字段。
4. 运行命令 `make html`，此命令会同时生成中英两语的 HTML 版本。您也可以分别运行 `make html-zh` 和 `make html-en` 命令生成中英文两个版本。
