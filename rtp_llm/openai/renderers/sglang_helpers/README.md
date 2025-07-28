有几个直接copy sglang对应代码之后的修改, 在这里补充一下

- 修改对应import的绝对路径
- 移除了对应的structure_info和build_ebnf方法依赖
- partial_json_parser目前还没被加入到系统依赖, 本地测试中手动import
- qwen25_detector中间流式会对arguments做json.dumps()但是没有ensure_ascii=False, 会错误的转义中文, 故在base_format_detector中间修改
- qwen25_detector中间非流式场景没有正确的处理tool_index, 做了fix
- qwen3_detector中间没有正确的处理index