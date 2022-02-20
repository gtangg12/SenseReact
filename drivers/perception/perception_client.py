"""
    Driver for translating video into captions

    Client side driver interface
"""
import sys
import time
sys.path.append('../../')
from pipe import RemoteTunnel, timestamp_name

class PerceptionDriver:
    pipe = RemoteTunnel('perception')

    @classmethod
    def next(cls):
        client_path = f'dock/perception_{timestamp_name()}.txt'
        if not cls.pipe.get(client_path):
            return
        with open(client_path, 'r') as fin:
            text = fine.read()
        os.remove(client_path)
        return text

    @classmethod
    def sync(cls):
        pass
