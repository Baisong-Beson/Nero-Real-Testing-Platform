"""Reconstruct train-only ready poses from sealed ROS bags; never publishes ROS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--audits',type=Path,required=True)
    p.add_argument('--bags',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extractor',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.extractor.parent))
    from extract_plate_ready_states import reconstruct
    a.output.mkdir(parents=True,exist_ok=True)
    for task in ('plate','banana','holder'):
        mp=a.audits/task/'manifest.json';manifest=json.loads(mp.read_text())
        states=[];evidence=[]
        for i,entry in enumerate(manifest['train']):
            ap=a.audits/task/(entry['episode']+'_audit.json')
            if sha(ap)!=entry['audit_sha256']:raise ValueError('audit hash mismatch')
            state,row=reconstruct(a.bags/f'session_20260920_{task}',entry,json.loads(ap.read_text()))
            states.append(state);evidence.append(row)
            print(f'{task}: {i+1}/{len(manifest["train"])} verified',flush=True)
        starts=np.stack([s[0] for s in states]).astype(float);all_states=np.concatenate(states)
        mean=starts.mean(0);ji=list(range(7))+list(range(8,15))
        report=dict(schema='act_eval.ready_pose.v1',task=task,estimator='equal-episode arithmetic mean of measured motion_ready frame; train split only',
            manifest_sha256=sha(mp),train_episodes=len(states),train_frames=len(all_states),
            ready_state_mean=mean.tolist(),ready_state_std=starts.std(0,ddof=1).tolist(),
            ready_state_min=starts.min(0).tolist(),ready_state_max=starts.max(0).tolist(),
            train_state_min=all_states.min(0).tolist(),train_state_max=all_states.max(0).tolist(),
            training_max_joint_excursion_from_mean_ready_deg=np.rad2deg(np.max(np.abs(all_states[:,ji]-mean[ji]),axis=0)).reshape(2,7).tolist(),
            episode_duration_s=[len(s)*.05 for s in states],evidence=evidence,physical_motion_executed=False)
        np.savez_compressed(a.output/f'{task}_train_states.npz',**{e['episode']:s for e,s in zip(manifest['train'],states)})
        report['states_file_sha256']=sha(a.output/f'{task}_train_states.npz')
        (a.output/f'{task}_ready.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='evidence'}),flush=True)
if __name__=='__main__':main()
