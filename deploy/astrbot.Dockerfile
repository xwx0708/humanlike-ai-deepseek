# AstrBot 官方镜像 + 本项目在 AstrBot 里运行需要的依赖（见 pyproject.toml）
FROM soulter/astrbot:latest
RUN python -m pip install --no-cache-dir "jieba>=0.42" "pypdf>=5" "python-docx>=1.1" "openpyxl>=3.1" "python-pptx>=1.0"
