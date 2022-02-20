"""
    Records frames from camera

    Client side driver interface
"""
import sys
import os
import threading
sys.path.append('../../')
from pipe import Pipe
from kernel_state import threads
from drivers.camera.camera_process import start_record

class CameraDriver:
    pipe = Pipe('camera')

    @classmethod
    def next_frame(cls):
        return cls.pipe.get()

    @classmethod
    def sync(cls):
        p = threading.Thread(target=start_record, args=(cls,), daemon=True)
        threads.append(p)
        p.start()
