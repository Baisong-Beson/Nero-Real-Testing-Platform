"""Exercise real Tk import dialogs and persistence with no real backend."""
from pathlib import Path
import ctypes
import os
for _name in ('libtcl8.6.so','libtk8.6.so'):ctypes.CDLL('/usr/lib/x86_64-linux-gnu/'+_name,mode=ctypes.RTLD_GLOBAL)
os.environ['TCL_LIBRARY']='/usr/share/tcltk/tcl8.6';os.environ['TK_LIBRARY']='/usr/share/tcltk/tk8.6'
import tempfile
import time
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch
from . import common as c,library
from .app import App
from .test_ui import FakeBackend
from .capture_window import capture

def main():
    evidence=c.new_session('library_ui_validation')
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp)
        with patch.object(c,'PLATFORM',base),patch.object(c,'PREFERENCES',base/'preferences.json'),patch.object(c,'RUNS',base/'runs'),patch('act_eval_workbench.app.RUNS',base/'runs'),patch('subprocess.Popen',side_effect=AssertionError('UI test cannot launch processes')):
            c.refresh_catalog();root=tk.Tk();fake=FakeBackend(base);app=App(root,fake);root.update()
            def wait():
                deadline=time.monotonic()+8
                while app.library.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
                root.update();assert not app.library.busy,app.library.status.get()
            def form_data(window,data):
                def children(widget):
                    for child in widget.winfo_children():yield child;yield from children(child)
                entries=[w for w in children(window) if isinstance(w,ttk.Entry)]
                for entry,value in zip(entries,data):entry.delete(0,'end');entry.insert(0,value)
                next(w for w in children(window) if isinstance(w,ttk.Button) and w['text']=='校验并导入').invoke()
                wait()
            app.notebook.select(3);app.library.task_form();root.update()
            window=next(w for w in root.winfo_children() if isinstance(w,tk.Toplevel))
            form_data(window,['ui_task','界面新任务','Pick up the cup.','杯子留在篮内 2 秒','0,90,90,90,0,0,0','0,90,-90,90,0,0,0','40,100'])
            assert app.settings().task=='ui_task' and app.model.get()==''
            assert '右 40 / 左 100 mm' in app.reset_hint['text']
            app.library.service_form();root.update();window=next(w for w in root.winfo_children() if isinstance(w,tk.Toplevel))
            form_data(window,['ui_vla','界面 VLA','ws://127.0.0.1:9999','ui_vla','a'*64,'ui_task','3','2','.05','2','main'])
            assert app.settings().model=='ui_vla'
            assert c.read(c.PREFERENCES)['last_selection']==dict(task='ui_task',model='ui_vla')
            app.library.table.selection_set('model:ui_vla');app.library.details();root.update()
            capture(root,evidence/'model_library.png')
            assert 'ACT' not in app.library.detail.get('1.0','end')
            templates=library.export_templates(base);assert (templates/'使用说明.md').exists()
            # Import a JSON through the real dialog handler, including background completion.
            task=library.task_definition('ui_task');task.update(id='ui_imported',label='文件导入任务')
            c.write(base/'task.json',task)
            with patch('act_eval_workbench.library_ui.filedialog.askopenfilename',return_value=str(base/'task.json')):app.library.import_file('task')
            wait();assert 'ui_imported' in c.TASKS
            app.task.set(c.TASKS['ui_task']);app.model.set(c.MODELS['ui_vla']);app.selection_changed()
            app.on_close();app.tick()
            root=tk.Tk();app=App(root,FakeBackend(base));root.update()
            assert app.settings().task=='ui_task' and app.settings().model=='ui_vla'
            root.geometry('1180x820');app.notebook.select(3);root.update()
            for button in app.library.buttons:
                assert button.winfo_rootx()+button.winfo_width()<=root.winfo_rootx()+root.winfo_width()
            capture(root,evidence/'model_library_minimum.png');app.on_close();app.tick()
        c.refresh_catalog()
    c.write(evidence/'validation.json',dict(passed=True,physical_motion_executed=False,tests=['create_task','connect_vla_form','compatible_selection','task_reset_label','json_import','export_templates','restart_selection','minimum_layout']))
    print('LIBRARY_UI_PASSED',evidence)

if __name__=='__main__':main()
