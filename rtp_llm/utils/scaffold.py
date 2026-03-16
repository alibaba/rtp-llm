# 这个文件中定义并初始化一个全局单例对象。
#
# 它的作用是作为开发阶段的临时脚手架，用来保存一些需要在不同模块之间共享的信息。
# 由于 rtpllm 工程较大，直接修改原有代码结构会比较麻烦，
# 所以这里提供一个简单的全局入口，方便在不影响原有代码的前提下，
# 快速注入和传递一些状态、配置或调试信息。
#
# 这个对象主要用于快速原型开发，不建议长期承载正式业务逻辑。


class Scaffold:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.duplicated_kv_head = False
        return cls._instance


SCAFFOLD_QWEN35_MI355X = Scaffold()
