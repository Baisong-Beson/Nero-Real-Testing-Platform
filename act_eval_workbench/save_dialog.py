"""Directory picker with an explicit create-folder action on Linux Tk."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox,simpledialog
from . import storage


class ExportDirectoryDialog:
    def __init__(self,parent):
        self.result=None;self.current=storage.initial_export_directory()
        self.window=tk.Toplevel(parent);self.window.title('保存实验结果 · 选择目录')
        self.window.geometry('850x520');self.window.minsize(650,400);self.window.transient(parent)
        self.path=tk.StringVar(value=str(self.current));self.status=tk.StringVar()
        body=ttk.Frame(self.window,padding=16);body.pack(fill='both',expand=True)
        ttk.Label(body,text='优先使用上次成功保存且仍可写的目录；不可用时使用 qbs 默认目录。',wraplength=780).pack(anchor='w')
        row=ttk.Frame(body);row.pack(fill='x',pady=10)
        self.entry=ttk.Entry(row,textvariable=self.path);self.entry.pack(side='left',fill='x',expand=True)
        self.entry.bind('<Return>',lambda _:self.navigate(self.path.get()))
        ttk.Button(row,text='前往',command=lambda:self.navigate(self.path.get())).pack(side='left',padx=(8,0))
        actions=ttk.Frame(body);actions.pack(fill='x',pady=(0,8))
        ttk.Button(actions,text='上一级',command=lambda:self.navigate(self.current.parent)).pack(side='left')
        ttk.Button(actions,text='默认目录',command=self.default_directory).pack(side='left',padx=8)
        self.new_button=ttk.Button(actions,text='新建文件夹',command=self.new_folder);self.new_button.pack(side='left')
        frame=ttk.Frame(body);frame.pack(fill='both',expand=True)
        self.folders=ttk.Treeview(frame,columns=('name',),show='headings',selectmode='browse')
        self.folders.heading('name',text='文件夹（双击进入）');self.folders.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(frame,orient='vertical',command=self.folders.yview);scroll.pack(side='right',fill='y')
        self.folders.configure(yscrollcommand=scroll.set);self.folders.bind('<Double-1>',self.enter_selected)
        status_label=ttk.Label(body,textvariable=self.status,wraplength=780)
        bottom=ttk.Frame(body);bottom.pack(side='bottom',fill='x',before=frame)
        status_label.pack(side='bottom',anchor='w',pady=8,before=frame)
        ttk.Button(bottom,text='取消',command=self.cancel).pack(side='right')
        self.save_button=ttk.Button(bottom,text='保存到此目录',command=self.accept);self.save_button.pack(side='right',padx=8)
        self.window.protocol('WM_DELETE_WINDOW',self.cancel)
        self.window.bind('<Escape>',lambda _:self.cancel())
        self.navigate(self.current)

    def navigate(self,path):
        try:
            path=Path(path).expanduser().resolve()
            if not path.is_dir():raise ValueError('目录不存在，请使用“新建文件夹”创建')
            children=sorted((p for p in path.iterdir() if p.is_dir()),key=lambda p:p.name.casefold())
            self.current=path;self.path.set(str(path));self.folders.delete(*self.folders.get_children())
            self.children={str(i):p for i,p in enumerate(children)}
            for key,p in self.children.items():self.folders.insert('',tk.END,iid=key,values=(p.name,))
            self.status.set('保存时会在此目录下生成独立的 ACT_results_时间戳 文件夹，保留已有导出。')
            return True
        except (OSError,ValueError) as exc:
            self.status.set(str(exc));return False

    def default_directory(self):
        from .common import DEFAULT_EXPORTS
        try:DEFAULT_EXPORTS.mkdir(parents=True,exist_ok=True);self.navigate(DEFAULT_EXPORTS)
        except OSError as exc:self.status.set(str(exc))

    def enter_selected(self,_=None):
        selected=self.folders.selection()
        if selected:self.navigate(self.children[selected[0]])

    def new_folder(self):
        if not self.navigate(self.path.get()):return
        name=simpledialog.askstring('新建文件夹','文件夹名称：',parent=self.window)
        if name is None:return
        try:self.navigate(storage.create_folder(self.current,name.strip()))
        except (OSError,ValueError) as exc:messagebox.showerror('无法新建文件夹',str(exc),parent=self.window)

    def accept(self):
        if not self.navigate(self.path.get()):return
        if not storage.usable_directory(self.current):
            self.status.set('目录不可写，请选择其他位置。');return
        self.result=str(self.current);self.window.destroy()

    def cancel(self):self.result=None;self.window.destroy()

    def show(self):
        # Tk on Xwayland may reject a grab before the window is mapped.
        try:self.window.wait_visibility();self.window.grab_set();self.window.wait_window()
        except tk.TclError:
            if self.window.winfo_exists():raise
        return self.result


def choose_export_directory(parent):return ExportDirectoryDialog(parent).show()
