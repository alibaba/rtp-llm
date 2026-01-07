"""
测试 markers 模块。
提供与 pytest.mark 兼容的 marker 装饰器，即使没有 pytest 也能工作。

用法:
    from rtp_llm.test.markers import mark

    @mark.gpu
    class MyGpuTest(TestCase):
        pass

    @mark.cpu
    class MyCpuTest(TestCase):
        pass
"""

try:
    import pytest

    mark = pytest.mark
except ImportError:
    # 如果没有 pytest，创建一个 dummy marker 系统
    class DummyMark:
        """当 pytest 不可用时的 dummy marker"""

        def __getattr__(self, name):
            """返回一个保存 marker 名称的装饰器"""

            def decorator(cls_or_func):
                # 在类/函数上设置 _markers 属性
                if not hasattr(cls_or_func, "_test_markers"):
                    cls_or_func._test_markers = set()
                cls_or_func._test_markers.add(name)
                return cls_or_func

            return decorator

    mark = DummyMark()


def has_marker(cls, marker_name):
    """检查类是否有指定的 marker"""
    # 检查 pytest markers
    marks = getattr(cls, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    for m in marks:
        if hasattr(m, "name") and m.name == marker_name:
            return True

    # 检查 dummy markers
    test_markers = getattr(cls, "_test_markers", set())
    if marker_name in test_markers:
        return True

    return False
