# AstrBot 官方镜像 + 本项目在 AstrBot 里运行需要的依赖（见 pyproject.toml）
FROM soulter/astrbot:latest
RUN python -m pip install --no-cache-dir "jieba>=0.42"
