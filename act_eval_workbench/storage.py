"""Persistent export locations, independent of shell cwd and robot services."""
import os
from pathlib import Path
import tempfile
from . import common as m


def usable_directory(path):
    try:
        path=Path(path).expanduser().resolve()
        if not path.is_dir() or not os.access(path,os.W_OK|os.X_OK):return False
        with tempfile.TemporaryFile(dir=path):pass
        return True
    except (OSError,ValueError,TypeError):return False


def preferences():
    try:
        data=m.read(m.PREFERENCES)
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError):return {}


def initial_selection():
    from .library import models_for
    selected=preferences().get('last_selection',{})
    if not isinstance(selected,dict):selected={}
    task,model=selected.get('task'),selected.get('model')
    task=task if isinstance(task,str) and task in m.TASKS else 'plate'
    models=models_for(task)
    return dict(task=task,model=model if isinstance(model,str) and model in models else next(iter(models),'mpi-base'))


def remember_selection(task,model):
    if task not in m.TASKS or model not in m.MODELS:raise ValueError('任务或模型无效，未保存选择')
    data=preferences();data['last_selection']=dict(task=task,model=model);m.write(m.PREFERENCES,data)


def initial_export_directory():
    previous=preferences().get('last_export_directory')
    if previous and usable_directory(previous):return Path(previous).expanduser().resolve()
    m.DEFAULT_EXPORTS.mkdir(parents=True,exist_ok=True)
    if not usable_directory(m.DEFAULT_EXPORTS):raise OSError('默认导出目录不可写：'+str(m.DEFAULT_EXPORTS))
    return m.DEFAULT_EXPORTS.resolve()


def remember_export_directory(path):
    path=Path(path).expanduser().resolve()
    if not usable_directory(path):raise OSError('导出位置不可用：'+str(path))
    data=preferences();data['last_export_directory']=str(path);m.write(m.PREFERENCES,data)


def create_folder(parent,name):
    if not name or name in ('.','..') or any(c in name for c in ('/','\\','\0')):
        raise ValueError('请输入单个文件夹名称，不能包含路径分隔符')
    parent=Path(parent).expanduser().resolve()
    if not usable_directory(parent):raise OSError('当前目录不可写：'+str(parent))
    folder=parent/name;folder.mkdir(exist_ok=False)
    return folder
