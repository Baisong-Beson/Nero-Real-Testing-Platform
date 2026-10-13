"""Camera role selection UI. Device discovery stays outside the Tk event loop."""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .camera_devices import binding
from .camera_preview import VIEWS


class CameraSettingsDialog:
    def __init__(self, app):
        self.app=app;self.manager=getattr(app.backend,'camera_manager',None)
        if self.manager is None:raise RuntimeError('当前会话未启用设备管理')
        self.window=tk.Toplevel(app.root);self.window.title('相机设置');self.window.transient(app.root)
        self.window.geometry('670x340');self.window.minsize(620,330)
        self.messages=queue.Queue();self.busy=False;self.dirty=False;self.signature=None
        frame=ttk.Frame(self.window,padding=20);frame.pack(fill='both',expand=True)
        title=ttk.Frame(frame);title.pack(fill='x',pady=(0,16))
        ttk.Label(title,text='相机分配',font=('Noto Sans CJK SC',15,'bold')).pack(side='left')
        ttk.Button(title,text='刷新设备',command=self.manager.refresh.set).pack(side='right')
        self.vars={};self.boxes={}
        for role in ('main','right','left'):
            row=ttk.Frame(frame);row.pack(fill='x',pady=6)
            ttk.Label(row,text=VIEWS[role],width=13).pack(side='left')
            var=tk.StringVar(value='未分配');self.vars[role]=var
            box=ttk.Combobox(row,textvariable=var,state='readonly',width=49)
            box.pack(side='left',fill='x',expand=True);box.bind('<<ComboboxSelected>>',self.changed)
            self.boxes[role]=box
        self.automatic=tk.BooleanVar(value=self.manager.config.get('auto_rebind',True))
        ttk.Checkbutton(frame,text='仅有一个替换设备时自动更新序列号',variable=self.automatic,command=self.changed).pack(anchor='w',pady=(14,6))
        bottom=ttk.Frame(frame);bottom.pack(side='bottom',fill='x',pady=(12,0))
        self.status=tk.StringVar(value='正在扫描相机…')
        ttk.Label(bottom,textvariable=self.status,wraplength=425).pack(side='left',fill='x',expand=True)
        self.save_button=ttk.Button(bottom,text='应用',command=self.apply);self.save_button.pack(side='right')
        self.poll()

    def changed(self,_=None):self.dirty=True

    def poll(self):
        if not self.window.winfo_exists():return
        try:
            kind,text=self.messages.get_nowait();self.busy=False
            if kind=='error':messagebox.showerror('相机设置',text,parent=self.window)
            else:self.dirty=False
            self.status.set(text);self.signature=None
        except queue.Empty:pass
        roles=self.manager.config.get('roles',{})
        devices=list(self.manager.devices);present={d['id'] for d in devices}
        signature=repr((devices,roles))
        if signature!=self.signature and not self.busy:
            selected={role:self.by_label.get(var.get()) for role,var in self.vars.items()} if hasattr(self,'by_label') else {}
            all_devices={d['id']:d for d in devices}
            all_devices.update({d['id']:d for d in roles.values() if d and d['id'] not in all_devices})
            self.by_label={'未分配':None};self.by_id={}
            for device in all_devices.values():
                label=device['name']+' · '+device['serial']+('（离线）' if device['id'] not in present else '')
                self.by_label[label]=device;self.by_id[device['id']]=label
            for role,box in self.boxes.items():
                box.configure(values=list(self.by_label))
                row=selected.get(role) if self.dirty else roles.get(role)
                self.vars[role].set(self.by_id.get(row['id'],'未分配') if row else '未分配')
            self.signature=signature
            if not self.dirty:self.status.set(f'已发现 {len(devices)} 台相机')
        if self.manager.error and not self.busy:self.status.set(self.manager.error[:100])
        if not self.busy and self.manager.pending and not self.dirty:
            self.status.set('对应关系不唯一，请手动分配' if self.manager.devices else '未检测到相机，接入后自动刷新')
        self.save_button.configure(state='disabled' if self.busy else 'normal')
        for box in self.boxes.values():box.configure(state='disabled' if self.busy else 'readonly')
        self.window.after(600,self.poll)

    def apply(self):
        roles={role:binding(self.by_label[var.get()],role) if self.by_label.get(var.get()) else None
               for role,var in self.vars.items()}
        automatic=self.automatic.get();self.busy=True;self.status.set('正在更新相机…')
        def work():
            try:self.manager.apply(roles,automatic);self.messages.put(('ok','已保存，正在连接所选相机'))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()
