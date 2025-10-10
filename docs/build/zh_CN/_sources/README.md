# 添加文档手册

## 1. 添加文档
* 如果是设计可以添加MD文档，如果是使用文档可以在Jupyter Notebook 上运行则建议添加ipynb文档


## 2. 添加索引
* docs 的索引由.rst 文件构建，添加好文档之后根据文档的用途将文档放到对应的目录结构


## 3. 增加中文

1. 进入 docs 目录。
2. 运行命令 `make update`。此命令会提取所有可翻译的消息并更新 PO 翻译文件。
3. 手动修改新增的翻译文件，文件位于 docs/locales/zh_CN/LC_MESSAGES 下，补全对应语言的 msgstr 字段。
4. 运行命令 `make html`，此命令会同时生成中英两语的 HTML 版本。您也可以分别运行 `make html-zh` 和 `make html-en` 命令生成中英文两个版本。
