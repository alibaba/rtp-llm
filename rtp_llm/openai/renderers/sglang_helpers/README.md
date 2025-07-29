# SGLang 代码修改说明


## 修改内容

### 1. 导入路径调整
**修改对应 import 的绝对路径**
- 将原有的绝对导入路径调整为适配当前项目结构的相对路径
- 确保模块间的依赖关系正确建立

### 2. 中文编码问题修复
**修复 `qwen25_detector` 中的中文转义问题**

**问题描述：**
- 流式处理中对 `arguments` 执行 `json.dumps()` 时缺少 `ensure_ascii=False` 参数
- 导致中文字符被错误转义

**解决方案：**
```python
cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
```

### 3. 工具索引处理修复
**修复 `qwen25_detector` 非流式场景的 `tool_index` 处理**

**问题描述：**
- 非流式场景下没有正确处理 `tool_index`

**解决方案：**
```python
# XinshiFix: 修复 tool_index 错误被分配的情况
for i, call in enumerate(calls):
    call.tool_index = i
```

### 4. 索引处理优化
**修复 `qwen3_coder_detector` 的索引处理问题**

**问题描述：**
- 没有正确处理 `index` 参数
- 索引处理逻辑不完善

**解决方案：**
```python
# old
res.extend(self.parse_base_json(raw, tools))

# new
# XinshiFix: 修复 tool_index 错误被分配的情况
parsed_calls = self.parse_base_json(raw, tools)
# 手动设置正确的 tool_index（父类注释要求的）
for call in parsed_calls:
    call.tool_index = self._current_tool_index
    self._current_tool_index += 1
res.extend(parsed_calls)
```