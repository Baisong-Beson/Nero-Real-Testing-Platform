"""Public NERO platform namespace; existing installations remain compatible."""
from pathlib import Path
__path__=[str(Path(__file__).resolve().parent.parent/'act_eval_workbench')]
