"""全部测试共用：不读用户的 settings.json，一律用代码里的默认值（D40）。"""

import os

os.environ["HUMANLIKE_SETTINGS"] = os.path.join(os.path.dirname(__file__), "_no_settings.json")
