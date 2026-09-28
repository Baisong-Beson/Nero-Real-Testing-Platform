"""Chinese desktop UI. Physical operations require a local plan confirmation."""
import argparse
import os
from pathlib import Path
import queue
import subprocess
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
from .common import *
from .backend import Backend
from . import records as record_store
from .camera_health import indicator
from .save_dialog import choose_export_directory
from . import storage
from . import library
from .library_ui import LibraryPanel

class App:
    def __init__(self,root,backend=None):
        self.root=root;self.backend=backend or Backend()
        self.last_session=None;self.last_kind=None;self.pending_disable=False;self.pending_reset=None;self.closing=False
        self.dialog=None;self.photo=None;self.image_stamp=None;self.last_progress='';self.simulation=[];self.result_dirs={};self.tick_handle=None
        root.title('nero真机评测工作台');root.geometry('1400x920');root.minsize(1180,820);root.configure(bg='#edf2f7')
        root.protocol('WM_DELETE_WINDOW',self.on_close);root.bind('<Escape>',lambda _:self.emergency())
        style=ttk.Style();style.theme_use('clam')
        style.configure('.',font=('Noto Sans CJK SC',10),background='#edf2f7')
        style.configure('TButton',padding=(10,5));style.configure('TLabel',foreground='#183655')
        style.configure('TLabelframe.Label',font=('Noto Sans CJK SC',11,'bold'))
        style.configure('Header.TLabel',font=('Noto Sans CJK SC',20,'bold'))
        style.configure('Treeview',rowheight=27,background='white',fieldbackground='white')
        selected=storage.initial_selection()
        self.task=tk.StringVar(value=TASKS[selected['task']]);self.model=tk.StringVar(value=MODELS[selected['model']])
        self.duration=tk.StringVar(value='180');self.excursion=tk.StringVar(value='45');self.tcp=tk.StringVar(value='20')
        self.relative_limits=tk.BooleanVar(value=False)
        self.operator=tk.StringVar(value='现场操作员');self.layout=tk.StringVar(value='layout_001')
        self.state=tk.StringVar(value='就绪 · 只读连接中');self.model_state=tk.StringVar(value='尚未加载本工作台模型')
        self.connection=tk.StringVar(value='等待 ROS 反馈');self.outcome=tk.StringVar(value='未判定')
        self.hardware_status=tk.StringVar(value='启动时自动检查 CAN、双臂与主相机连接')
        self.reason=tk.StringVar();self.session_label=tk.StringVar(value='尚无本工作台实验');self.buttons={}
        self.build();self.selection_changed();self.refresh_records();self.tick()

    def build(self):
        outer=ttk.Frame(self.root,padding=16);outer.pack(fill='both',expand=True)
        head=ttk.Frame(outer);head.pack(fill='x',pady=(0,12))
        ttk.Label(head,text='nero真机评测工作台',style='Header.TLabel').pack(side='left')
        ttk.Label(head,text='任务 · 模型 · 起手位 · 推理 · 实验记录',foreground='#637990').pack(side='left',padx=22)
        ttk.Label(head,textvariable=self.state,wraplength=560).pack(side='right')
        columns=ttk.Frame(outer);columns.pack(fill='both',expand=True)
        left=ttk.Frame(columns,width=330);left.pack(side='left',fill='y',padx=(0,14));left.pack_propagate(False)
        right=ttk.Frame(columns);right.pack(side='left',fill='both',expand=True)
        footer=ttk.Frame(left);footer.pack(side='bottom',fill='x',pady=(8,0))
        self.buttons['disable']=ttk.Button(footer,text='张开夹爪 → 双臂及夹爪失能',command=lambda:self.request('disable'))
        self.buttons['disable'].pack(fill='x',pady=(0,6))
        self.stop_button=tk.Button(footer,text='■ 紧急停止测试 / Esc',font=('Noto Sans CJK SC',14,'bold'),bg='#bd2939',fg='white',
            activebackground='#941d2a',activeforeground='white',relief='flat',command=self.emergency,pady=12)
        self.stop_button.pack(fill='x',pady=(0,5))
        ttk.Label(footer,text='软件停止：关闭控制并请求保持。\n硬件急停按钮仍可直接使用。',foreground='#6b7280').pack(anchor='w')
        scroll_frame=ttk.Frame(left);scroll_frame.pack(fill='both',expand=True)
        scroll=tk.Canvas(scroll_frame,bg='#edf2f7',highlightthickness=0)
        bar=ttk.Scrollbar(scroll_frame,orient='vertical',command=scroll.yview);bar.pack(side='right',fill='y')
        scroll.pack(side='left',fill='both',expand=True);scroll.configure(yscrollcommand=bar.set)
        controls=ttk.Frame(scroll);window=scroll.create_window((0,0),window=controls,anchor='nw')
        controls.bind('<Configure>',lambda _:scroll.configure(scrollregion=scroll.bbox('all')))
        scroll.bind('<Configure>',lambda e:scroll.itemconfigure(window,width=e.width))
        setup=ttk.LabelFrame(controls,text='评测配置',padding=10);setup.pack(fill='x')
        for label,var,values in [('任务',self.task,list(TASKS.values())),('模型',self.model,list(MODELS.values()))]:
            ttk.Label(setup,text=label).pack(anchor='w')
            box=ttk.Combobox(setup,textvariable=var,values=values,state='readonly');box.pack(fill='x',pady=(2,7))
            box.bind('<<ComboboxSelected>>',self.selection_changed)
            if label=='任务':self.task_box=box
            else:self.model_box=box
        for label,var in [('最长推理秒数（1–600）',self.duration),('操作员',self.operator),('布局编号',self.layout)]:
            row=ttk.Frame(setup);row.pack(fill='x',pady=3);ttk.Label(row,text=label).pack(side='left')
            ttk.Entry(row,textvariable=var,width=14).pack(side='right')
        ttk.Label(setup,textvariable=self.model_state,wraplength=286,foreground='#476983').pack(anchor='w',pady=(5,0))
        ttk.Label(setup,text='运行速度由机械臂驱动控制\n已取消工作台低速插值',wraplength=286,foreground='#476983').pack(anchor='w',pady=(5,0))
        for title,items in [
            ('准备与模拟',[('load','加载 / 切换模型'),('pose','计算任务起手位'),('offline','离线回放与轨迹模拟'),('shadow','实时相机只读推理')]),
            ('真机操作',[('ready','复位到任务起手位'),('default','恢复指定默认姿态'),('formal','启动一次真机测试')])]:
            frame=ttk.LabelFrame(controls,text=title,padding=8);frame.pack(fill='x',pady=(10,0))
            for key,label in items:
                b=ttk.Button(frame,text=label,command=lambda k=key:self.request(k));b.pack(fill='x',pady=2);self.buttons[key]=b
            if title=='真机操作':
                self.reset_hint=ttk.Label(frame,text='',wraplength=274,foreground='#476983');self.reset_hint.pack(anchor='w',pady=(5,0))
        self.notebook=ttk.Notebook(right);self.notebook.pack(fill='both',expand=True)
        live=ttk.Frame(self.notebook,padding=10);pose=ttk.Frame(self.notebook,padding=12);records=ttk.Frame(self.notebook,padding=12)
        for frame,title in [(live,'实时监视'),(pose,'姿态与模拟'),(records,'实验记录')]:self.notebook.add(frame,text=title)
        ttk.Label(live,textvariable=self.connection,wraplength=950).pack(anchor='w',pady=(0,8))
        connection_tools=ttk.Frame(live);connection_tools.pack(fill='x',pady=(0,8))
        self.reconnect_button=ttk.Button(connection_tools,text='重新检查全部连接',command=self.recover_connection)
        self.reconnect_button.pack(side='right',padx=(8,0))
        self.camera_reconnect_button=ttk.Button(connection_tools,text='重连主相机',command=lambda:self.recover_connection(camera_only=True))
        self.camera_reconnect_button.pack(side='right')
        ttk.Label(connection_tools,textvariable=self.hardware_status,wraplength=560).pack(side='left',fill='x',expand=True)
        camera_status=ttk.Frame(live);camera_status.pack(fill='x',pady=(0,8));self.wrist_indicators={}
        self.main_camera_indicator=ttk.Label(camera_status,text='● 主相机：检测中',foreground='#7b8794')
        self.main_camera_indicator.pack(side='left',padx=(0,24))
        for side,label in [('left','左夹爪相机'),('right','右夹爪相机')]:
            item=ttk.Frame(camera_status);item.pack(side='left',padx=(0,24))
            light=tk.Canvas(item,width=18,height=18,bg='#edf2f7',highlightthickness=0);light.pack(side='left',padx=(0,5))
            dot=light.create_oval(3,3,15,15,fill='#7b8794',outline='')
            text=ttk.Label(item,text=label+'：检测中');text.pack(side='left')
            self.wrist_indicators[side]=(light,dot,text,label)
        self.camera=tk.Label(live,text='等待主相机画面',bg='#10263b',fg='#d4e4f4');self.camera.pack(fill='both',expand=True)
        cols=('arm',*[str(i) for i in range(1,8)],'grip')
        self.joints=ttk.Treeview(live,columns=cols,show='headings',height=2)
        for col in cols:
            self.joints.heading(col,text='手臂' if col=='arm' else '夹爪/mm' if col=='grip' else 'J'+col)
            self.joints.column(col,width=82 if col in ('arm','grip') else 68,anchor='center')
        for arm,label in [('right','右臂'),('left','左臂')]:self.joints.insert('',tk.END,iid=arm,values=[label]+['—']*8)
        self.joints.pack(side='bottom',before=self.camera,fill='x',pady=10)
        self.pose_text=scrolledtext.ScrolledText(pose,height=12,wrap='word',font=('Noto Sans Mono CJK SC',10),relief='flat')
        self.pose_text.pack(fill='x');self.pose_text.configure(state='disabled')
        bounds=ttk.LabelFrame(pose,text='可选调试范围',padding=10);bounds.pack(fill='x',pady=10)
        self.range_toggle=ttk.Checkbutton(bounds,text='启用相对起点的运动范围限制（仅小范围调试）',variable=self.relative_limits,command=self.update_range_controls)
        self.range_toggle.pack(anchor='w');self.range_entries=[]
        for label,var in [('相对起点最大关节变化 / °（5–90）',self.excursion),('末端最大位移 / cm（2–40）',self.tcp)]:
            row=ttk.Frame(bounds);row.pack(fill='x',pady=3);ttk.Label(row,text=label).pack(side='left')
            entry=ttk.Entry(row,textvariable=var,width=10);entry.pack(side='right');self.range_entries.append(entry)
        ttk.Label(bounds,text='默认关闭：不因 45° / 20 cm / 连杆 25 cm 的相对位移截停。开启时连杆上限为末端上限 +5 cm。\n保留绝对关节限位、工作空间、反馈与卡住检测；运行速度由驱动控制。不包含桌面和物体碰撞检测。',wraplength=770,foreground='#6b7280').pack(anchor='w',pady=5)
        self.update_range_controls()
        ttk.Label(pose,text='离线模拟：右/左 joint4 理想跟踪轨迹（不含接触动力学）').pack(anchor='w')
        self.plot=tk.Canvas(pose,bg='white',height=210,highlightthickness=0);self.plot.pack(fill='both',expand=True,pady=6)
        self.plot.bind('<Configure>',lambda _:self.draw_plot())
        ttk.Label(records,textvariable=self.session_label,wraplength=850).pack(anchor='w')
        row=ttk.Frame(records);row.pack(fill='x',pady=10)
        ttk.Button(row,text='刷新记录',command=self.refresh_records).pack(side='left')
        ttk.Button(row,text='打开当前记录目录',command=self.open_directory).pack(side='left',padx=8)
        self.record_buttons={}
        actions=ttk.Frame(records);actions.pack(fill='x',pady=(0,6))
        for key,label,command in [('export_selected','导出所选',lambda:self.export_records(False)),
                ('export_all','导出全部',lambda:self.export_records(True)),
                ('clear_selected','清除所选',lambda:self.clear_records(False)),
                ('clear_all','清除全部',lambda:self.clear_records(True)),
                ('restore','恢复已清除',self.restore_records)]:
            button=ttk.Button(actions,text=label,command=command);button.pack(side='left',padx=(0,6));self.record_buttons[key]=button
        self.record_count=ttk.Label(records,text='仅记录已开始的真机模型推理；支持 Ctrl / Shift 多选；清除可恢复。',wraplength=760)
        self.record_count.pack(anchor='w',pady=(0,4))
        table=ttk.Frame(records)
        self.records=ttk.Treeview(table,columns=('task','model','duration','state','outcome'),show='headings',height=9,selectmode='extended')
        for col,title in [('task','任务'),('model','模型'),('duration','上限/s'),('state','执行结果'),('outcome','人工判定')]:
            self.records.heading(col,text=title);self.records.column(col,width=110)
        record_scroll=ttk.Scrollbar(table,orient='vertical',command=self.records.yview);record_scroll.pack(side='right',fill='y')
        self.records.configure(yscrollcommand=record_scroll.set)
        self.records.pack(side='left',fill='both',expand=True);self.records.bind('<<TreeviewSelect>>',self.record_selected)
        verdict=ttk.LabelFrame(records,text='记录当前正式试验结果',padding=10)
        ttk.Combobox(verdict,textvariable=self.outcome,values=['未判定','成功','失败'],state='readonly',width=12).pack(side='left')
        ttk.Entry(verdict,textvariable=self.reason).pack(side='left',fill='x',expand=True,padx=8)
        ttk.Button(verdict,text='保存判定',command=self.save_outcome).pack(side='right')
        self.statistics=scrolledtext.ScrolledText(records,height=3,wrap='word',font=('Noto Sans CJK SC',9),relief='flat')
        self.statistics.pack(side='bottom',fill='x',pady=(4,0));self.statistics.configure(state='disabled')
        verdict.pack(side='bottom',fill='x',pady=(8,0));table.pack(fill='both',expand=True)
        library_frame=ttk.Frame(self.notebook,padding=12);self.notebook.add(library_frame,text='模型与任务库')
        self.library=LibraryPanel(self,library_frame)
        frame=ttk.LabelFrame(right,text='运行日志与诊断',padding=6);frame.pack(side='bottom',before=self.notebook,fill='x',pady=(10,0))
        self.logs=scrolledtext.ScrolledText(frame,height=6,wrap='word',font=('Noto Sans Mono CJK SC',9),relief='flat')
        self.logs.pack(fill='x');self.logs.configure(state='disabled')

    def settings(self):
        return Settings(task=next(k for k,v in TASKS.items() if v==self.task.get()),model=next((k for k,v in MODELS.items() if v==self.model.get()),'mpi-base'),
            duration_s=float(self.duration.get()),
            excursion_deg=float(self.excursion.get()) if self.relative_limits.get() else 45.,
            tcp_cm=float(self.tcp.get()) if self.relative_limits.get() else 20.,
            relative_limits_enabled=self.relative_limits.get(),operator=self.operator.get(),layout=self.layout.get()).checked()
    def update_range_controls(self):
        for entry in self.range_entries:entry.configure(state='normal' if self.relative_limits.get() else 'disabled')
    def log(self,text):
        self.logs.configure(state='normal');self.logs.insert(tk.END,time.strftime('%H:%M:%S')+'  '+str(text)+'\n')
        self.logs.see(tk.END);self.logs.configure(state='disabled')
    def selection_changed(self,_=None):
        try:
            task=next(k for k,v in TASKS.items() if v==self.task.get());choices=library.models_for(task)
            self.model_box.configure(values=list(choices.values()))
            if self.model.get() not in choices.values():self.model.set(next(iter(choices.values()),''))
            model=next((k for k,v in choices.items() if v==self.model.get()),'mpi-base')
            self.show_pose(ready_pose(task));self.update_task_labels(task)
            storage.remember_selection(task,model)
            self.model_state.set('配置已保存；加载时核对模型身份。' if choices else '此任务暂无兼容模型，请在“模型与任务库”导入模型。仍可查看姿态或复位。')
        except Exception as exc:self.log(exc)
    def update_task_labels(self,task):
        short={'banana':'香蕉','plate':'盘子','holder':'放笔'}.get(task,TASKS[task])
        widths=ready_gripper_target(task)
        state='右爪闭合' if task=='banana' else '双爪张开' if widths==[.1,.1] else f'右 {widths[0]*1000:g}/左 {widths[1]*1000:g} mm'
        self.buttons['ready'].configure(text=f'复位到{short}起手位（{state}）' if task in library.BUILTIN_TASKS else '复位到任务起手位')
        self.reset_hint.configure(text='当前任务：'+TASKS[task]+f'\n到位后夹爪：右 {widths[0]*1000:g} / 左 {widths[1]*1000:g} mm')
    def show_pose(self,pose):
        text=f'{TASKS[pose["task"]]} · {pose["train_episodes"]} 条训练示教\n{pose["estimator"]}\n\n'
        definition=library.task_definition(pose['task'])
        text+='语言指令：'+definition['instruction']+'\n成功标准：'+definition['success_definition']+'\n\n'
        for i,label in enumerate(('右臂','左臂')):text+=label+' 起手位 / °：'+', '.join(f'{x:.3f}' for x in pose['joints_deg'][i])+'\n'
        text+='\n训练起手夹爪均宽 / mm：'+', '.join(f'{x*1000:.1f}' for x in pose['gripper_width_mean_m'])+'\n'
        text+='复位会先保持关节、张开双夹爪到 100 mm，再移动关节；会松开持物。\n'
        if pose['task']=='banana':text+='香蕉任务：双臂到位后，右爪闭合度1（驱动目标0 mm），左爪100 mm。\n持物时实测宽度可以大于0；闭合到位或稳定接触后完成。推理前确认右手实际持蕉。\n'
        elif pose['task'] not in library.BUILTIN_TASKS:text+='关节到位后按任务定义恢复右、左夹爪宽度 / mm：'+str([x*1000 for x in ready_gripper_target(pose['task'])])+'\n'
        text+='训练起手位是建议参考，偏离时仍可从当前姿态启动推理，确认窗口会提示偏差。\n'
        text+='\n指定默认位 / °：\n右 [0, 90, 90, 90, 0, 0, 0]\n左 [0, 90, -90, 90, 0, 0, 0]\n'
        if pose['task'] in library.BUILTIN_TASKS:text+=f'\n训练中最大 joint4 偏移：右 {pose["train_excursion_deg"][0][3]:.1f}°，左 {pose["train_excursion_deg"][1][3]:.1f}°。\n该统计用于诊断，不会自动放宽范围。'
        self.pose_text.configure(state='normal');self.pose_text.delete('1.0',tk.END);self.pose_text.insert(tk.END,text);self.pose_text.configure(state='disabled')
    def request(self,kind):
        try:
            if self.library.busy:raise RuntimeError('请等待模型 / 任务导入完成')
            if kind in ('load','offline','shadow','formal') and not self.model.get():raise ValueError('请先导入兼容当前任务的模型')
            if kind=='disable' and self.backend.busy:
                self.pending_reset=None;self.pending_disable=True;self.backend.stop(physical=True);self.state.set('先停止任务，随后准备开爪并全失能确认');return
            if kind in ('ready','default') and self.backend.busy:
                self.pending_disable=False;self.pending_reset=kind;self.backend.stop(physical=True);self.state.set('先停止任务，随后从实时位置准备复位');return
            if self.dialog and self.dialog.winfo_exists():raise RuntimeError('请先处理当前动作确认窗口')
            settings=self.settings();storage.remember_selection(settings.task,settings.model)
            self.last_kind=kind;directory=self.backend.start(kind,settings)
            self.state.set('正在准备：'+kind);self.log(f'开始 {kind} · 日志目录：{directory}')
        except Exception as exc:messagebox.showerror('无法开始',str(exc),parent=self.root)
    def show_confirmation(self,plan,directory):
        dialog=tk.Toplevel(self.root);self.dialog=dialog;dialog.title('确认本次真机动作');dialog.transient(self.root);dialog.geometry('820x560')
        ttk.Label(dialog,text='本次动作',style='Header.TLabel').pack(anchor='w',padx=20,pady=14)
        body='任务：'+TASKS[plan['settings']['task']]+'\n模型：'+MODELS[plan['settings']['model']]+'\n\n'+plan['description']+'\n\n'
        if 'start_joints_rad' in plan:
            for i,label in enumerate(('右臂','左臂')):body+=label+' 当前 / °：'+', '.join(f'{x:.2f}' for x in np.rad2deg(plan['start_joints_rad'][i]))+'\n'
            if 'start_widths_m' in plan:body+='当前夹爪 / mm：'+', '.join(f'{x*1000:.1f}' for x in plan['start_widths_m'])+'\n'
        if plan.get('gripper_policy') in ('open_to_100mm','open_then_disable'):body+='双夹爪目标：100 mm（完全张开，会松开持物）\n'
        if plan.get('gripper_policy')=='task_target_after_joint_arrival':
            target=plan['trajectory']['gripper_goal_m']
            if plan['settings']['task']=='banana':body+=f'起手到位后夹爪指令：右闭合度1 / {target[0]*1000:.1f} mm（闭合目标），左 {target[1]*1000:.1f} mm（张开）\n持物实测开度可大于0；闭合到位或稳定接触后完成。\n'
            else:body+=f'起手到位后夹爪指令：右 {target[0]*1000:g} / 左 {target[1]*1000:g} mm。\n'
        if 'trajectory' in plan:
            for i,label in enumerate(('右臂','左臂')):body+=label+' 目标 / °：'+', '.join(f'{x:.2f}' for x in np.rad2deg(plan['trajectory']['goal'][i]))+'\n'
        if plan['kind']=='formal':
            s=plan['settings']
            for warning in plan.get('start_warnings',[]):body+='起手提示：'+warning+'\n'
            body+=('\n相对起点调试范围：'+f'关节 {s["excursion_deg"]:g}°，末端 {s["tcp_cm"]:g} cm，连杆 {s["tcp_cm"]+5:g} cm。\n' if s.get('relative_limits_enabled',True) else '\n相对起点调试范围：关闭；绝对限位、工作空间与运动保护仍开启。\n')
            body+=f'成功标准：{plan["success_definition"]}。\n'
        body+='\n停止会关闭控制并请求保持，不自动复位或开始下一次。失能会撤去电机支撑。'
        area=scrolledtext.ScrolledText(dialog,height=15,wrap='word',font=('Noto Sans CJK SC',11));area.pack(fill='both',expand=True,padx=20)
        area.insert('1.0',body);area.configure(state='disabled')
        check=tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog,text='我已核对动作，通路已清空；失能时已确保双臂和持物获支撑。',variable=check).pack(anchor='w',padx=20,pady=12)
        row=ttk.Frame(dialog);row.pack(fill='x',padx=20,pady=(0,16))
        def cancel():dialog.destroy();self.dialog=None;self.state.set('已取消，未执行动作')
        def execute():
            if not check.get():messagebox.showwarning('请核对现场','请先勾选现场确认。',parent=dialog);return
            try:
                approve(plan,directory,self.operator.get());dialog.destroy();self.dialog=None
                self.last_kind='execute'
                self.backend.start('execute',Settings(**plan['settings']),directory);self.state.set('真机动作执行中')
            except Exception as exc:messagebox.showerror('无法执行',str(exc),parent=self.root)
        ttk.Button(row,text='取消',command=cancel).pack(side='left');ttk.Button(row,text='确认并执行',command=execute).pack(side='right')
        dialog.protocol('WM_DELETE_WINDOW',cancel)
        self.confirmation_check=check;self.confirmation_execute=execute;self.confirmation_cancel=cancel
        # X11 may not map a transient window until its parent becomes visible.
        # Never block the UI heartbeat with wait_visibility / wait_window.
        def grab_when_visible(event=None):
            if self.dialog is not dialog or not dialog.winfo_exists():return
            if event is not None and event.widget is not dialog:return
            if not dialog.winfo_viewable():return
            try:dialog.grab_set()
            except tk.TclError:dialog.after(100,grab_when_visible)
        dialog.bind('<Map>',grab_when_visible)
        dialog.after_idle(grab_when_visible)
    def emergency(self):
        if self.dialog and self.dialog.winfo_exists():self.confirmation_cancel()
        self.pending_disable=False;self.pending_reset=None;self.backend.stop(physical=True);self.state.set('已发停止请求，正在核对收尾')
    def draw_plot(self):
        c=self.plot;c.delete('all');w=max(c.winfo_width(),100);h=max(c.winfo_height(),100)
        if not self.simulation:c.create_text(w/2,h/2,text='完成离线模拟后显示模型输出对应的关节轨迹',fill='#60778d');return
        data=np.array([x['joints_deg'] for x in self.simulation]);lo=float(data[:,:,3].min())-3;hi=float(data[:,:,3].max())+3
        c.create_line(45,15,45,h-28,w-10,h-28,fill='#8297aa')
        for v in np.linspace(lo,hi,5):
            y=h-28-(v-lo)/(hi-lo)*(h-43);c.create_line(45,y,w-10,y,fill='#e5edf4');c.create_text(25,y,text=f'{v:.0f}°',fill='#60778d')
        for arm,color,label in [(0,'#2371bf','右 J4'),(1,'#d78620','左 J4')]:
            points=[]
            for i,q in enumerate(data):points += [45+i/max(1,len(data)-1)*(w-65),h-28-(q[arm,3]-lo)/(hi-lo)*(h-43)]
            if len(points)>=4:c.create_line(*points,fill=color,width=2)
            c.create_text(w-65,18+arm*20,text=label,fill=color)
    def refresh_records(self):
        try:
            entries=record_store.scan(RUNS);hidden=sum(row['archived'] for row in record_store.scan(RUNS,True));groups=summarize_records()
        except (ValueError,OSError,TypeError) as exc:
            self.record_count.configure(text='记录读取失败，见运行日志');self.log('记录读取失败：'+str(exc));return
        self.records.delete(*self.records.get_children());self.result_dirs={}
        self.record_data={row['id']:row['result'] for row in entries}
        for row in entries:
            r=row['result'];s=r.get('settings',{});v=r.get('adjudication',{}).get('success');iid=row['id'];self.result_dirs[iid]=Path(row['directory'])
            self.records.insert('',tk.END,iid=iid,values=(TASKS.get(s.get('task'),s.get('task','—')),s.get('model','—'),s.get('duration_s','—'),
                (record_store.error_text(r) or (r.get('kind','')+' · '+('完成' if r.get('runtime_completed') else '未完成')))+(' · 录像缺帧提示' if r.get('video',{}).get('gap_warning_count') else ''),
                '成功' if v is True else '失败' if v is False else '未判定'))
        self.record_count.configure(text=f'真机模型推理：当前 {len(entries)} 条，已清除 {hidden} 条。复位、失能、预检及模拟不计入；支持多选与恢复。')
        lines=[f'{g["task"]}/{g["model"]} · {g["layout"]} · {g["duration_s"]:g}s · 参数 {g["protocol"]}：已做 {g["started"]}/{g["target"]}，成功 {g["success"]}，失败 {g["failed"]}，待判定 {g["pending"]}；成功/已开始 {g["success"]/g["started"]:.1%}' for g in groups]
        self.statistics.configure(state='normal');self.statistics.delete('1.0',tk.END)
        self.statistics.insert('1.0','\n'.join(lines) or '尚无本工作台正式试验；模拟和复位不计成功率。');self.statistics.configure(state='disabled')
    def record_selected(self,_=None):
        selected=self.records.selection()
        if not selected:return
        self.last_session=self.result_dirs[selected[0]];self.session_label.set(str(self.last_session));r=self.record_data[selected[0]]
        v=r.get('adjudication',{}).get('success');self.outcome.set('成功' if v is True else '失败' if v is False else '未判定')
        self.reason.set(r.get('adjudication',{}).get('reason',''))
    def save_outcome(self):
        try:
            if len(self.records.selection())>1:raise ValueError('判定结果时请选择一条记录')
            if not self.last_session:raise ValueError('请先选择一条正式试验')
            p=self.last_session/'result.json';r=read(p)
            if not record_store.is_experiment(r):raise ValueError('只有已开始的真机模型推理可记录任务成功；模拟/复位不计')
            value={'成功':True,'失败':False,'未判定':None}[self.outcome.get()]
            if value is not None and not self.reason.get().strip():raise ValueError('请填写判定依据或失败原因')
            entry=dict(success=value,reason=self.reason.get().strip(),operator=self.operator.get(),unix_s=time.time())
            r.setdefault('adjudication_history',[]).append(dict(previous=r.get('adjudication'),new=entry));r['adjudication']=entry
            write(p,r);self.refresh_records();self.log('已保存判定，原始执行日志保持不变')
        except Exception as exc:messagebox.showerror('无法保存',str(exc),parent=self.root)
    def record_operation_allowed(self):
        if self.backend.busy or (self.dialog and self.dialog.winfo_exists()):raise RuntimeError('请先结束当前任务或处理动作确认窗口，再管理记录')
    def clear_records(self,all_records=False):
        try:
            self.record_operation_allowed()
            ids=[row['id'] for row in record_store.scan(RUNS)] if all_records else list(self.records.selection())
            if not ids:raise ValueError('请先选择需要清除的记录' if not all_records else '没有可清除的记录')
            if not messagebox.askyesno('清除实验记录',f'将清除 {len(ids)} 条记录，使其退出列表和当前成功率统计。\n原始日志、结果和录像保留，可通过“恢复已清除”恢复。\n确认清除？',parent=self.root):return
            count=record_store.set_archived(RUNS,ids,True,self.operator.get())
            if self.last_session and self.last_session.name in ids:self.last_session=None;self.session_label.set('所选记录已清除，可恢复')
            self.refresh_records();self.log(f'已清除 {count} 条记录（可恢复，原始文件保留）')
        except Exception as exc:messagebox.showerror('无法清除',str(exc),parent=self.root)
    def restore_records(self):
        try:
            self.record_operation_allowed();ids=[r['id'] for r in record_store.scan(RUNS,True) if r['archived']]
            if not ids:raise ValueError('没有已清除的记录')
            if not messagebox.askyesno('恢复实验记录',f'恢复 {len(ids)} 条已清除记录，并重新纳入统计？',parent=self.root):return
            count=record_store.set_archived(RUNS,ids,False,self.operator.get());self.refresh_records();self.log(f'已恢复 {count} 条记录')
        except Exception as exc:messagebox.showerror('无法恢复',str(exc),parent=self.root)
    def export_records(self,all_records=False):
        try:
            self.record_operation_allowed();ids=None if all_records else list(self.records.selection())
            if ids==[]:raise ValueError('请先选择要导出的记录，可用 Ctrl / Shift 多选')
            if all_records and not record_store.scan(RUNS):raise ValueError('没有可导出的记录')
            destination=choose_export_directory(self.root)
            if not destination:return
            out=record_store.export(RUNS,destination,ids);self.last_export=out;self.log('实验结果已导出：'+str(out))
            try:storage.remember_export_directory(destination)
            except OSError as exc:self.log('导出成功，但无法记住保存位置：'+str(exc))
            messagebox.showinfo('导出完成',f'已保存逐条 CSV、汇总 CSV、完整 JSON 和校验清单：\n{out}\n\n录像仍保留在原始记录目录。',parent=self.root)
        except Exception as exc:messagebox.showerror('无法导出',str(exc),parent=self.root)
    def recover_connection(self,camera_only=False):
        if self.dialog and self.dialog.winfo_exists():
            self.log('请先关闭动作确认窗口，再重新检查连接');return
        try:
            if camera_only:self.backend.recover_connection(camera_only=True)
            else:self.backend.recover_connection()
        except Exception as exc:self.log(str(exc))

    def open_directory(self):subprocess.Popen(['xdg-open',str(self.last_session or RUNS)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    def tick(self):
        if self.tick_handle is not None:
            self.root.after_cancel(self.tick_handle);self.tick_handle=None
        try:self.refresh_ui()
        except Exception as exc:
            traceback.print_exc()
            self.pending_disable=False;self.pending_reset=None
            if self.dialog and self.dialog.winfo_exists():self.dialog.destroy();self.dialog=None
            try:self.backend.stop(physical=self.backend.busy and self.last_kind=='execute')
            except Exception:traceback.print_exc()
            self.state.set('界面刷新异常，任务已请求停止 · 见日志')
            self.log(f'界面刷新异常：{type(exc).__name__}: {exc}')
        finally:
            try:
                if self.root.winfo_exists():self.tick_handle=self.root.after(100,self.tick)
            except tk.TclError:pass  # Window closed normally.
    def refresh_ui(self):
        self.backend.tick()
        try:
            while True:
                e=self.backend.events.get_nowait()
                if e['type']=='log':self.log(e['text'])
                elif e['type']=='library_import':self.library.completed(e)
                elif e['type']=='hardware':
                    message=e['message']
                    camera=e.get('result',{}).get('camera_usb',{}).get('message')
                    if camera and not e.get('result',{}).get('main_camera'):message+='\n'+camera
                    self.hardware_status.set(message[:240])
                elif e['type']=='model':self.model_state.set(f'已加载 {e["task"]}/{e["model"]} · 端口 {e["port"]}')
                elif e['type']=='error':self.state.set('任务未完成 · 见日志');self.log(e['error'])
                elif e['type']=='done':
                    r=e['result'];directory=Path(e['directory'])
                    if e['kind'] in ('ready','default','formal','disable','micro'):
                        if self.pending_disable or self.pending_reset:self.log('已取消旧计划，等待停止后准备新动作')
                        else:self.state.set('等待动作确认');self.show_confirmation(r,directory)
                    else:
                        problem=r.get('error') or '; '.join(r.get('errors',[]))
                        self.state.set(('存在错误 · ' if problem else '任务结束 · ')+e['kind']);self.log(problem or '完成 '+e['kind'])
                        if e['kind']=='pose':self.show_pose(r);self.notebook.select(1)
                        if e['kind']=='offline':self.simulation=r['trajectory'];self.draw_plot();self.notebook.select(1)
                        if e['kind']=='execute':
                            self.refresh_records()
                            if record_store.is_experiment(r):
                                self.records.selection_set(directory.name);self.record_selected();self.notebook.select(2)
        except queue.Empty:pass
        t=self.backend.telemetry_state()
        from .camera_reconnect import fresh as main_camera_fresh
        main_ok=main_camera_fresh(t)
        self.main_camera_indicator.configure(text='● 主相机：'+('通畅' if main_ok else '未连接 / 图像过期'),foreground='#16a34a' if main_ok else '#bd2939')
        for side,(light,dot,text,label) in self.wrist_indicators.items():
            color,status=indicator(t.get('wrist_cameras',{}).get(side),time.time()-t.get('updated_unix_s',0))
            light.itemconfigure(dot,fill=color);text.configure(text=label+'：'+status)
        if t:
            arms=t.get('arms',{});stale=time.time()-t.get('updated_unix_s',0)>.8
            for arm,label in [('right','右臂'),('left','左臂')]:
                row=arms.get(arm)
                if row:stale=stale or row['age_s']>.3;self.joints.item(arm,values=[label]+[f'{x:.2f}' for x in row['joints_deg']]+[f'{row["width_m"]*1000:.1f}'])
            age=t.get('camera_age_s');count=sum(t.get('control_publishers',{}).values())
            grip=[]
            for arm,label in [('right','右'),('left','左')]:
                row=t.get('grippers',{}).get(arm,{})
                status='未知/过期' if stale or not -.05<=row.get('age_s',999)<=.3 else '故障' if row.get('fault') else '使能' if row['enabled'] else '失能'
                grip.append(label+status)
            self.connection.set(f'反馈：{"过期/异常" if stale or len(arms)<2 else "实时"}    相机：{str(round(age,2))+" s" if age is not None else "未连接"}    控制发布者：{count}    夹爪：'+ ' / '.join(grip))
            if self.state.get()=='就绪 · 只读连接中' and not stale and len(arms)==2 and age is not None and age<.5:
                self.state.set('就绪 · 相机与双臂已连接')
        path=self.backend.directory/'camera.jpg'
        if path.exists() and path.stat().st_mtime!=self.image_stamp:
            try:
                im=Image.open(path);im.thumbnail((max(700,self.camera.winfo_width()),450));self.photo=ImageTk.PhotoImage(im)
                self.camera.configure(image=self.photo,text='');self.image_stamp=path.stat().st_mtime
            except (OSError,ValueError):pass
        if self.backend.busy and self.backend.active:
            try:
                progress=read(self.backend.active/'progress.json')['text']
                if progress!=self.last_progress:self.state.set(progress[:110]);self.last_progress=progress
            except (OSError,ValueError,KeyError):pass
        hardware_busy=getattr(self.backend,'hardware_busy',False)
        for key,button in self.buttons.items():button.configure(state='normal' if not self.library.busy and not hardware_busy and (not self.backend.busy or key in ('disable','ready','default')) else 'disabled')
        blocked=self.backend.busy or self.library.busy or hardware_busy or bool(self.dialog and self.dialog.winfo_exists())
        for button in self.library.buttons:button.configure(state='disabled' if blocked else 'normal')
        for box in (self.task_box,self.model_box):box.configure(state='disabled' if blocked else 'readonly')
        self.reconnect_button.configure(state='disabled' if blocked or self.backend.stop_jobs else 'normal')
        self.camera_reconnect_button.configure(state='disabled' if blocked or self.backend.stop_jobs else 'normal')
        if not blocked and not self.closing and hasattr(self.backend,'check_camera_reconnect'):self.backend.check_camera_reconnect()
        for button in self.record_buttons.values():button.configure(state='disabled' if blocked else 'normal')
        for proc,directory in list(self.backend.stop_jobs):
            if proc.poll() is not None:
                try:
                    r=read(directory/'stop_result.json');self.log('停止服务：'+('已响应' if not r['errors'] else str(r['errors'])))
                except (OSError,ValueError):self.log('停止服务未返回记录，请查看日志/使用硬件急停')
                self.backend.stop_jobs.remove((proc,directory))
        if self.pending_disable and not self.backend.busy and not self.backend.stop_jobs:self.pending_disable=False;self.request('disable')
        if self.pending_reset and not self.backend.busy and not self.backend.stop_jobs:
            kind=self.pending_reset;self.pending_reset=None;self.request(kind)
        if self.closing and self.backend.close():self.root.destroy();return
    def on_close(self):
        if self.library.busy:
            self.state.set('正在导入，请完成后关闭窗口');return
        if self.dialog and self.dialog.winfo_exists():self.confirmation_cancel()
        if self.backend.busy:self.backend.stop(physical=self.last_kind=='execute')
        self.closing=True;self.state.set('正在停止后台任务并关闭')

def main():
    p=argparse.ArgumentParser();p.add_argument('--screenshot',type=Path);p.add_argument('--close-after',type=float);a=p.parse_args()
    import fcntl
    lock=open('/tmp/act_eval_workbench_desktop.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('nero真机评测工作台已经打开，请使用已有窗口。')
    root=tk.Tk();app=App(root)
    if a.screenshot:
        def capture():
            from .capture_window import capture as capture_drawable
            a.screenshot.parent.mkdir(parents=True,exist_ok=True)
            capture_drawable(root,a.screenshot)
        root.after(5000,capture)
    if a.close_after:root.after(int(a.close_after*1000),app.on_close)
    root.mainloop()
if __name__=='__main__':main()
