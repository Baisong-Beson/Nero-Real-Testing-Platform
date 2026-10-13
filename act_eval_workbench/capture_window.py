"""Capture an application X11 drawable (Xwayland root screenshots may be black)."""
import ctypes as c
from PIL import Image

def capture(widget,path):
    widget.update_idletasks()
    class XImage(c.Structure):
        _fields_=[('width',c.c_int),('height',c.c_int),('xoffset',c.c_int),('format',c.c_int),
                  ('data',c.c_void_p),('byte_order',c.c_int),('bitmap_unit',c.c_int),('bitmap_bit_order',c.c_int),
                  ('bitmap_pad',c.c_int),('depth',c.c_int),('bytes_per_line',c.c_int),('bits_per_pixel',c.c_int),
                  ('red_mask',c.c_ulong),('green_mask',c.c_ulong),('blue_mask',c.c_ulong)]
    lib=c.CDLL('libX11.so.6');lib.XOpenDisplay.argtypes=[c.c_char_p];lib.XOpenDisplay.restype=c.c_void_p
    lib.XGetImage.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_int,c.c_uint,c.c_uint,c.c_ulong,c.c_int]
    lib.XGetImage.restype=c.POINTER(XImage);lib.XDestroyImage.argtypes=[c.POINTER(XImage)];lib.XCloseDisplay.argtypes=[c.c_void_p]
    d=lib.XOpenDisplay(None)
    if not d:raise RuntimeError('X display unavailable')
    ptr=None
    try:
        ptr=lib.XGetImage(d,widget.winfo_id(),0,0,widget.winfo_width(),widget.winfo_height(),c.c_ulong(-1).value,2)
        if not ptr:raise RuntimeError('XGetImage failed')
        im=ptr.contents
        if im.bits_per_pixel!=32 or im.byte_order!=0:raise RuntimeError('unexpected X image layout')
        data=c.string_at(im.data,im.bytes_per_line*im.height)
        image=Image.frombytes('RGB',(im.width,im.height),data,'raw','BGRX',im.bytes_per_line,1);image.save(path)
    finally:
        if ptr:lib.XDestroyImage(ptr)
        lib.XCloseDisplay(d)
