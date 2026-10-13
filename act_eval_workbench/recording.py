"""Record evidence without treating a brief dropped-frame gap as control loss."""
import json
from pathlib import Path
import queue
import threading
import cv2
import numpy as np

class VideoRecorder:
    MAX_SOURCE_GAP=.5
    WARN_SOURCE_GAP=.25
    def __init__(self,output,image_shape):
        self.output=Path(output);self.image_shape=tuple(image_shape);self.queue=queue.Queue(maxsize=64)
        self.error=None;self.frames=0;self.last_source_stamp=None;self.max_source_gap_s=0.
        self.gap_warnings=[];self.gap_warning_count=0
        self.thread=threading.Thread(target=self.work,name='act-video',daemon=True);self.thread.start()
    def submit(self,row):
        try:self.queue.put_nowait(row)
        except queue.Full:self.error='video queue overflow'
    def accept_stamp(self,stamp):
        gap=None if self.last_source_stamp is None else stamp-self.last_source_stamp
        if gap is not None:
            self.max_source_gap_s=max(self.max_source_gap_s,gap)
            if not 0<gap<=self.MAX_SOURCE_GAP:raise RuntimeError(f'video source gap {gap:.4f}s exceeds {self.MAX_SOURCE_GAP:g}s or timestamp regressed')
            if gap>self.WARN_SOURCE_GAP:
                self.gap_warning_count+=1
                if len(self.gap_warnings)<100:self.gap_warnings.append(dict(frame=self.frames,source_gap_s=gap))
        self.last_source_stamp=stamp
        return gap
    def work(self):
        writer=None
        try:
            h,w,_=self.image_shape;writer=cv2.VideoWriter(str(self.output/'main_camera.avi'),cv2.VideoWriter_fourcc(*'MJPG'),15.,(w,h))
            if not writer.isOpened():raise RuntimeError('MJPG video writer unavailable')
            with (self.output/'video_frames.jsonl').open('x') as log:
                while True:
                    row=self.queue.get()
                    if row is None:break
                    frame=cv2.imdecode(np.frombuffer(row['data'],np.uint8),cv2.IMREAD_COLOR)
                    if frame is None or frame.shape!=self.image_shape:raise RuntimeError('invalid/changed camera frame dimensions')
                    gap=self.accept_stamp(row['stamp']);writer.write(frame)
                    log.write(json.dumps(dict(frame=self.frames,source_unix_s=row['stamp'],received_monotonic_s=row['received'],source_gap_s=gap))+'\n');log.flush()
                    self.frames+=1
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
        finally:
            if writer:writer.release()
    def close(self):
        try:self.queue.put(None,timeout=2.)
        except queue.Full:self.error=self.error or 'video writer stuck'
        self.thread.join(3.)
        if self.thread.is_alive():self.error=self.error or 'video writer did not finish'
        return dict(frames=self.frames,max_source_gap_s=self.max_source_gap_s,error=self.error,
            gap_warning_count=self.gap_warning_count,gap_warnings=self.gap_warnings,
            quality='error' if self.error else 'minor_gaps' if self.gap_warning_count else 'complete',
            video_file='main_camera.avi',frame_times_file='video_frames.jsonl')
