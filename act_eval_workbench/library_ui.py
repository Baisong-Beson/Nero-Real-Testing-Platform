"""Model/task library controls. File validation and copies run off the Tk loop."""
import json
from pathlib import Path
import tempfile
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox,scrolledtext
from . import common as c,library,storage

class LibraryPanel:
    def __init__(self,app,parent):
        self.app=app;self.busy=False;self.buttons=[]
        ttk.Label(parent,text='模型与任务库',font=('Noto Sans CJK SC',15,'bold')).pack(anchor='w')
        ttk.Label(parent,text='导入文件会复制到平台 library 目录；原文件移动后仍可使用。其他架构通过 VLA 推理服务接入。\n任务定义与模型独立管理；只列出模型声明支持的任务组合。',wraplength=850).pack(anchor='w',pady=8)
        row=ttk.Frame(parent);row.pack(fill='x')
        for label,fn in [('导入模型',lambda:self.import_file('model')),('接入 VLA 服务',self.service_form),('导入任务',lambda:self.import_file('task')),('新建任务',self.task_form),('导出模板',self.templates),('刷新',self.refresh)]:
            b=ttk.Button(row,text=label,command=fn);b.pack(side='left',padx=(0,5));self.buttons.append(b)
        self.status=tk.StringVar(value='库目录：'+str(library.root()))
        ttk.Label(parent,textvariable=self.status,wraplength=850).pack(anchor='w',pady=8)
        self.table=ttk.Treeview(parent,columns=('kind','label','id','adapter'),show='headings',height=9)
        for key,label,width in [('kind','类别',60),('label','名称',210),('id','ID',180),('adapter','接口 / 任务',220)]:
            self.table.heading(key,text=label);self.table.column(key,width=width)
        self.table.pack(fill='both',expand=True);self.table.bind('<<TreeviewSelect>>',self.details)
        self.detail=scrolledtext.ScrolledText(parent,height=9,wrap='word',font=('Noto Sans Mono CJK SC',9));self.detail.pack(fill='x',pady=8)
        self.refresh()

    def idle(self):
        if self.busy or self.app.backend.busy or (self.app.dialog and self.app.dialog.winfo_exists()):raise RuntimeError('请先结束当前操作，再修改模型与任务库')

    def work(self,fn,kind):
        self.idle();self.busy=True;self.status.set('正在校验并导入，请稍候…')
        def run():
            try:self.app.backend.events.put(dict(type='library_import',key=fn(),kind=kind))
            except Exception as exc:self.app.backend.events.put(dict(type='library_import',error=str(exc),kind=kind))
        threading.Thread(target=run,daemon=True).start()

    def completed(self,event):
        self.busy=False
        if event.get('error'):
            self.status.set('导入未完成：'+event['error']);messagebox.showerror('导入未完成',event['error'],parent=self.app.root);return
        self.refresh();key=event['key']
        if event['kind']=='task':self.app.task.set(c.TASKS[key])
        else:
            compatible=[k for k in c.TASKS if library.compatible(k,key)]
            if compatible:
                current=next((k for k,v in c.TASKS.items() if v==self.app.task.get()),None)
                self.app.task.set(c.TASKS[current if current in compatible else compatible[0]])
                self.app.model.set(c.MODELS[key])
        self.app.selection_changed();self.status.set('导入完成：'+key+'。已复制到 '+str(library.root()))
        self.app.log(self.status.get())

    def refresh(self):
        c.refresh_catalog();self.table.delete(*self.table.get_children());self.entries={}
        for kind,data in [('task',c.TASKS),('model',c.MODELS)]:
            for key,label in data.items():
                builtin=key in (library.BUILTIN_TASKS if kind=='task' else library.BUILTIN_MODELS)
                d=library.task_definition(key) if kind=='task' else dict(id=key,label=label,adapter='内置推理',tasks=list(library.BUILTIN_TASKS)) if builtin else library.package(kind,key)[0]
                iid=kind+':'+key;self.entries[iid]=d
                self.table.insert('',tk.END,iid=iid,values=('任务' if kind=='task' else '模型',label,key,'内置' if builtin else d.get('adapter','任务定义')))
        if hasattr(self.app,'task_box'):self.app.task_box.configure(values=list(c.TASKS.values()))

    def details(self,_=None):
        selected=self.table.selection()
        if not selected:return
        self.detail.configure(state='normal');self.detail.delete('1.0',tk.END)
        self.detail.insert('1.0',json.dumps(self.entries[selected[0]],ensure_ascii=False,indent=2));self.detail.configure(state='disabled')

    def import_file(self,kind):
        try:
            self.idle()
            previous=storage.preferences().get('last_import_directory',str(c.PLATFORM))
            path=filedialog.askopenfilename(parent=self.app.root,title='选择模型描述 JSON / 已封存 policy.pt' if kind=='model' else '选择任务 task.json',initialdir=previous,
                filetypes=[('模型或任务描述','*.json *.pt')] if kind=='model' else [('任务描述','*.json')])
            if not path:return
            prefs=storage.preferences();prefs['last_import_directory']=str(Path(path).parent);c.write(c.PREFERENCES,prefs)
            if kind=='model' and Path(path).suffix=='.pt':self.sealed_form(path)
            else:self.work(lambda:library.import_package(path,kind),kind)
        except Exception as exc:messagebox.showerror('无法导入',str(exc),parent=self.app.root)

    def form(self,title,fields,submit,description=''):
        self.idle();window=tk.Toplevel(self.app.root);window.title(title);window.transient(self.app.root);window.geometry('820x660')
        frame=ttk.Frame(window,padding=16);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text=description,wraplength=770).pack(anchor='w',pady=(0,10))
        values={}
        for key,label,value in fields:
            row=ttk.Frame(frame);row.pack(fill='x',pady=4);ttk.Label(row,text=label,width=24).pack(side='left')
            var=tk.StringVar(value=value);values[key]=var;ttk.Entry(row,textvariable=var).pack(side='left',fill='x',expand=True)
        def save():
            try:submit({k:v.get().strip() for k,v in values.items()});window.destroy()
            except Exception as exc:messagebox.showerror('请检查输入',str(exc),parent=window)
        ttk.Button(frame,text='校验并导入',command=save).pack(side='bottom',anchor='e',pady=10)
        return window

    def import_data(self,data,kind):
        # The temporary descriptor is ours; no external code is evaluated.
        def action():
            with tempfile.TemporaryDirectory(prefix='nero-descriptor-') as tmp:
                path=Path(tmp)/f'{kind}.json';c.write(path,data);return library.import_package(path,kind)
        self.work(action,kind)

    def task_form(self):
        fields=[('id','任务 ID','my_task'),('label','任务名称','新任务'),('instruction','语言指令（传给模型）','Pick up the object and place it in the tray.'),
            ('success','成功标准','物体释放后在目标区域稳定 2 秒'),('right','右臂 7 关节 / 度','0,90,90,90,0,0,0'),('left','左臂 7 关节 / 度','0,90,-90,90,0,0,0'),('grippers','右、左夹爪宽度 / mm','100,100')]
        def submit(v):
            arrays={k:[float(x) for x in v[k].replace('，',',').split(',')] for k in ('right','left','grippers')}
            self.import_data(dict(schema='nero.task.v1',id=v['id'],label=v['label'],instruction=v['instruction'],success_definition=v['success'],
                ready=dict(method='explicit',joints_deg=[arrays['right'],arrays['left']],gripper_width_m=[x/1000 for x in arrays['grippers']])),'task')
        self.form('新建测试任务',fields,submit,'此处填写参考起手位，保存不会移动机器人。若要自动计算示教均值，请用带 start_states.npz 的任务包导入。')

    def service_form(self):
        fields=[('id','模型 ID','my_vla'),('label','模型名称','新 VLA'),('endpoint','推理地址','ws://127.0.0.1:9000'),('service_id','服务内模型 ID','my_vla'),
            ('sha','权重 SHA256',''),('tasks','支持任务 ID（逗号分隔）','*'),('horizon','每次预测步数','16'),('steps','每次执行步数','4'),('dt','动作时间间隔 / 秒','0.05'),('timeout','单次推理超时 / 秒','2'),('cameras','相机（逗号分隔）','main')]
        def submit(v):
            self.import_data(dict(schema='nero.model.v1',id=v['id'],label=v['label'],adapter='websocket',endpoint=v['endpoint'],service_model_id=v['service_id'],checkpoint_sha256=v['sha'],
                tasks=[x.strip() for x in v['tasks'].split(',')],action_contract=library.CONTRACT,
                runtime=dict(action_horizon=int(v['horizon']),steps_per_replan=int(v['steps']),dt_s=float(v['dt']),inference_timeout_s=float(v['timeout']),cameras=[x.strip() for x in v['cameras'].split(',')])),'model')
        self.form('接入 VLA 模型服务',fields,submit,'先按导出的说明接入 NERO 适配器。服务负责加载权重、归一化及动作转换；平台负责反馈、执行与记录。相机可填 main,left_wrist,right_wrist。')

    def sealed_form(self,path):
        task=next(k for k,v in c.TASKS.items() if v==self.app.task.get())
        fields=[('id','模型 ID','imported_policy'),('label','模型名称','导入训练模型'),('tasks','支持任务 ID（逗号分隔）',task)]
        self.form('导入训练完成的策略',fields,lambda v:self.work(lambda:library.import_sealed(path,v['id'],v['label'],[x.strip() for x in v['tasks'].split(',')]),'model'),
            '权重旁需有 verification.json；自动寻找同一导出包的离线样本和参考输出。模型仍按原训练方式推理；改任务名称不会改变它学到的技能。')

    def templates(self):
        try:
            self.idle()
            from .save_dialog import choose_export_directory
            path=choose_export_directory(self.app.root)
            if path:
                target=library.export_templates(path);self.status.set('模板已导出：'+str(target));self.app.log(self.status.get())
        except Exception as exc:messagebox.showerror('无法导出模板',str(exc),parent=self.app.root)
