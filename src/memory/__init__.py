"""记忆模块。别的模块只认 contracts.memory.MemoryService；这里导出它的实现，供组装处（插件、测试）构造。"""

from memory.service import DefaultMemoryService, RecentWindowMemory, background_job

__all__ = ["DefaultMemoryService", "RecentWindowMemory", "background_job"]
