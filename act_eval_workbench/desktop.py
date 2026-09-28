"""Use Ubuntu's Xft-enabled Tk so Chinese fonts render correctly in conda."""
import ctypes
import os
import sys
from pathlib import Path

def main():
    for library in ('libtcl8.6.so','libtk8.6.so'):
        path=Path('/usr/lib/x86_64-linux-gnu')/library
        if not path.exists():raise RuntimeError(f'需要系统 Tk/Xft：{path}')
        ctypes.CDLL(str(path),mode=ctypes.RTLD_GLOBAL)
    os.environ['TCL_LIBRARY']='/usr/share/tcltk/tcl8.6'
    os.environ['TK_LIBRARY']='/usr/share/tcltk/tk8.6'
    if '--self-test' in sys.argv:
        from .test_ui import main as run
    else:
        from .app import main as run
    run()
if __name__=='__main__':main()
