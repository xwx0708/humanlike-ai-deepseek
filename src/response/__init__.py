"""回应模块：组织每轮交给思考模型的内容，把对话交给记忆。

平台是 NapCat + AstrBot（D28），AstrBot 插件在 response/astrbot_plugin/；这里的代码不依赖 AstrBot，可以单独测试。
"""

from response import attachments, requests, responder  # noqa: F401  加载时登记各自的参数与功能开关（D42）
