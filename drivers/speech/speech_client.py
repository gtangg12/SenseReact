import sys
import os
import time

import numpy as np

sys.path.append('../../')
from pipe import RemoteTunnel, timestamp_name

class SpeechDriver:
    pipe = RemoteTunnel('speech')

    @classmethod
    def next(cls):
        client_path = f'dock/speech_{timestamp_name()}.txt'
        if not cls.pipe.get(client_path):
            return

        audio = np.loadtxt(client_path)
        os.remove(client_path)

        return audio # a numpy array describing audio that can be converted into .wav with tts_utils.save_audio

    @classmethod
    def sync(cls):
        pass
