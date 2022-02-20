import sys
import os
sys.path.append('../../')
from server import SERVER_DOCK


def perception_run():
    send_dir = f'{SERVER_DOCK}/perception_inp'
    recv_dir = f'{SERVER_DOCK}/perception_out'

    while True:
        filenames = sorted(os.listdir(send_dir))
        if len(filenames) == 0:
            continue
        filename = filenames[0]
        print(filename)

        os.remove(filename)
        
