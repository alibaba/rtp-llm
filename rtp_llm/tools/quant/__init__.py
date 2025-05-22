from .base_quanter import BaseQuanter
import pkgutil
import importlib

package_name = __name__

# 遍历当前包下的所有模块
for _, module_name, _ in pkgutil.iter_modules(__path__, package_name + '.'):
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 遍历模块中的所有属性
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # 检查属性是否是 BaseQuanter 的子类
        if isinstance(attribute, type) and issubclass(attribute, BaseQuanter) and attribute is not BaseQuanter:
            # 在这里执行注册逻辑，例如调用 register 方法
            attribute.register()