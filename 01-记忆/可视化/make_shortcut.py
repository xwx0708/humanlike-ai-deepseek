"""在桌面创建「记忆可视化」快捷方式：双击即启动服务器，并在副屏打开可视化窗口。

用法：python make_shortcut.py。已存在同名快捷方式时会覆盖。

直接调用 Windows 的 IShellLinkW 接口。不用 WScript.Shell，因为它在这台机器的系统代码页下
处理不了路径里的中文（"工程""记忆"等）。
"""

import ctypes
import ctypes.wintypes as wt
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "记忆可视化.lnk"


def _guid(s: str) -> ctypes.c_byte * 16:
    return (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(s).bytes_le)


CLSID_SHELL_LINK = _guid("00021401-0000-0000-C000-000000000046")
IID_SHELL_LINK_W = _guid("000214F9-0000-0000-C000-000000000046")
IID_PERSIST_FILE = _guid("0000010B-0000-0000-C000-000000000046")

# 虚函数表中的位置（前 3 个是 IUnknown 的 QueryInterface、AddRef、Release）
QUERY_INTERFACE, RELEASE = 0, 2
SET_DESCRIPTION, SET_WORKING_DIRECTORY, SET_ARGUMENTS, SET_ICON_LOCATION, SET_PATH = 7, 9, 11, 17, 20
PERSIST_SAVE = 6


def _call(obj: ctypes.c_void_p, index: int, *args, argtypes=()):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])
    return fn(obj, *args)


def _desktop() -> Path:
    buf = ctypes.create_unicode_buffer(wt.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf)  # CSIDL_DESKTOPDIRECTORY
    return Path(buf.value)


def create_shortcut(
    lnk: Path, target: str, arguments: str, workdir: str, icon: str, icon_index: int, description: str
) -> None:
    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)
    link = ctypes.c_void_p()
    ole32.CoCreateInstance(
        ctypes.byref(CLSID_SHELL_LINK), None, 1, ctypes.byref(IID_SHELL_LINK_W), ctypes.byref(link)
    )
    try:
        _call(link, SET_PATH, target, argtypes=(wt.LPCWSTR,))
        _call(link, SET_ARGUMENTS, arguments, argtypes=(wt.LPCWSTR,))
        _call(link, SET_WORKING_DIRECTORY, workdir, argtypes=(wt.LPCWSTR,))
        _call(link, SET_ICON_LOCATION, icon, icon_index, argtypes=(wt.LPCWSTR, ctypes.c_int))
        _call(link, SET_DESCRIPTION, description, argtypes=(wt.LPCWSTR,))
        persist = ctypes.c_void_p()
        _call(
            link,
            QUERY_INTERFACE,
            ctypes.byref(IID_PERSIST_FILE),
            ctypes.byref(persist),
            argtypes=(ctypes.c_void_p, ctypes.c_void_p),
        )
        try:
            _call(persist, PERSIST_SAVE, str(lnk), True, argtypes=(wt.LPCWSTR, wt.BOOL))
        finally:
            _call(persist, RELEASE)
    finally:
        _call(link, RELEASE)
        ole32.CoUninitialize()


def main() -> None:
    pythonw = Path(sys.executable).with_name("pythonw.exe")  # 用 pythonw，双击时不弹黑色命令行窗口
    lnk = _desktop() / NAME
    create_shortcut(
        lnk,
        target=str(pythonw),
        arguments=f'"{HERE / "open_on_second_screen.py"}"',
        workdir=str(HERE),
        icon=os.path.expandvars(r"%SystemRoot%\System32\imageres.dll"),
        icon_index=144,
        description="启动记忆可视化服务器，并在副屏打开可视化窗口",
    )
    print(f"已创建 {lnk}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()
