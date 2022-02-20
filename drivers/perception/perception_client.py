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

    captions = [
        'It is a nice day',
        'A man is walking towards a wall.',
        'A man is walking towards a wall.',
        'He bumps into the wall.',
        'He falls down.',
        'He falls down again.',
        'He seems to be very hurt lying on the floor.']
    timestamps = [0, 14/30, 59/30, 83/30, 97/30, 111/30, 5]
    timer = None
    cnt = 0

    @classmethod
    def next(cls):
        if cls.cnt >= len(cls.timestamps):
            while True:
                pass
        while time.time() - cls.timer < cls.timestamps[cls.cnt]:
            pass
        text = '. '.join(cls.captions[:cls.cnt + 1])
        cls.cnt += 1
        '''
        client_path = f'dock/perception_{timestamp_name()}.txt' # save to here
        while not cls.pipe.get(client_path):
            pass
        with open(client_path, 'r') as fin:
            text = fine.read()
        os.remove(client_path) # for stuff live wav files don't delete
        '''
        print(time.time() - cls.timer)
        return text # return filename for stuff like wav files

    @classmethod
    def sync(cls):
        cls.timer = time.time()
