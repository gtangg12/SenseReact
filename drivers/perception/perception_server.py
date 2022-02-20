import sys
import os
import time
sys.path.append('../../')
from server import SERVER_DOCK


def perception_run():
    inp_dir = f'{SERVER_DOCK}/perception_inp'
    out_dir = f'{SERVER_DOCK}/perception_out'

    while True:
        filenames = sorted(os.listdir(inp_dir))
        if len(filenames) == 0:
            continue
        filename = filenames[0]
        print(filename)

        os.remove(filename)


        """
        DO YOUR COMPUTATION HERE
        """

        with open(f'{out_dir}/{filename}', 'w') as fout:
            fout.write('AKSJGFHKJSAFGHJKFKASFGJ')
