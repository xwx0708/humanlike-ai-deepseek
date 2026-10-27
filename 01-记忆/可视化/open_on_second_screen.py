"""启动自动刷新服务器（如果没在运行），并把可视化页面用独立窗口最大化显示在副屏上。

用法：python open_on_second_screen.py [页面]，也可以直接双击「打开副屏.bat」。
不指定页面时：窗口已开着就只移回副屏，没开就打开 forgetting.html。
指定页面时：窗口已开着就关掉，在指定页面重新打开。
窗口用单独的 Edge 配置（%LOCALAPPDATA%\\memory-viz-edge），不影响日常使用的 Edge。
"""

import ctypes
import ctypes.wintypes as wt
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):  # 有的控制台编码打不出中文，至少别因此报错
    stream.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
PORT = 8790
PAGE_GIVEN = len(sys.argv) > 1
PAGE = sys.argv[1] if PAGE_GIVEN else "forgetting.html"
URL = f"http://127.0.0.1:{PORT}/{PAGE}"
PROFILE = Path(os.environ["LOCALAPPDATA"]) / "memory-viz-edge"
EDGE = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft/Edge/Application/msedge.exe"

user32 = ctypes.windll.user32
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.SetWindowPos.argtypes = [
    wt.HWND,
    wt.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wt.UINT,
]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def ensure_server() -> None:
    if port_open():
        return
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    subprocess.Popen(
        [str(pythonw), str(HERE / "serve.py"), str(PORT)],
        cwd=HERE,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    for _ in range(50):
        if port_open():
            return
        time.sleep(0.1)
    sys.exit(f"服务器没能在端口 {PORT} 启动")


def viz_pids() -> set[int]:
    """用本专用配置启动的 Edge 主进程（不含渲染等子进程）。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
        "Where-Object { $_.CommandLine -like '*memory-viz-edge*' -and $_.CommandLine -notlike '*--type=*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    cmd = ["powershell", "-NoProfile", "-Command", ps]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True).stdout
    return {int(x) for x in out.split()}


def windows_of(pids: set[int]) -> list:
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT), ("rcWork", wt.RECT), ("dwFlags", wt.DWORD)]


def secondary_work_area() -> wt.RECT:
    """副屏的可用区域；只有一块屏时退回主屏。"""
    areas = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM)
    def cb(hmon, _hdc, _rect, _):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        areas.append((bool(info.dwFlags & 1), info.rcWork))  # 1 = MONITORINFOF_PRIMARY
        return True

    user32.EnumDisplayMonitors(None, None, cb, 0)
    secondary = [r for primary, r in areas if not primary]
    return (secondary or [r for _, r in areas])[0]


def main() -> None:
    ensure_server()

    wins = windows_of(viz_pids())
    if wins and PAGE_GIVEN:  # 换页面：关掉旧窗口再开
        for hwnd in wins:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        deadline = time.time() + 10
        while wins and time.time() < deadline:
            time.sleep(0.5)
            wins = windows_of(viz_pids())
    if not wins:
        subprocess.Popen(
            [
                str(EDGE),
                f"--user-data-dir={PROFILE}",
                "--no-first-run",
                "--no-default-browser-check",
                f"--app={URL}",
            ]
        )
        deadline = time.time() + 15
        while not wins and time.time() < deadline:
            time.sleep(0.5)
            wins = windows_of(viz_pids())
        if not wins:
            sys.exit("Edge 窗口没有出现")

    area = secondary_work_area()
    hwnd = wins[0]
    user32.ShowWindow(hwnd, 9)  # 先还原，才能移动
    user32.SetWindowPos(hwnd, None, area.left + 40, area.top + 40, 800, 500, 0x0014)  # 不改层级、不抢焦点
    user32.ShowWindow(hwnd, 3)  # 在副屏上最大化
    print(f"已在副屏打开 {URL}")


if __name__ == "__main__":
    main()
