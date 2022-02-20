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

    captions = ['A man is flying a kite', 'The kite blows in the air', 'A child nearby noticies the kite', 'He jumps for the kite but slips', 'He falls down',
    'Another man walks past', 'He sees the child', 'He looks at the child who seems hurt', 'He runs to the child who lies dying on the ground', 'He trips and falls']
    timestamps = [0, 3, 6, 10, 12, 15, 17, 19, 20, 25]
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
