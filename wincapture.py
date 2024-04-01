import numpy as np
from ctypes import windll
import win32gui
import win32ui



def capture_win_alt(window_name: str):
    """
    Capture a screenshot of a specified window.

    Args:
        window_name (str): The name of the window to capture.

    Returns:
        numpy.ndarray: The captured screenshot as a numpy array.

    Raises:
        RuntimeError: If unable to acquire the screenshot.
        ValueError: If the specified window is not found.
    """

    #Depreciated
    #if not windll.user32.GetProcessDPIAware():
    windll.user32.SetProcessDPIAware()

    hwnd = win32gui.FindWindow(None, window_name)

    if hwnd is None:
        raise ValueError(f"Window '{window_name}' not found")

    if hwnd is not None:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top
    else:
        # handle the case when hwnd is None
        ...

    hwnd_dc = win32gui.GetWindowDC(hwnd)

    try:
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    except Exception as e:
        print(f"Error creating device context: {e}")
        return None

    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()

    try:
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    except Exception as e:
        print(f"Error creating compatible bitmap: {e}")
        return None

    save_dc.SelectObject(bitmap)

    result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel

    if not result:
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")

    return img