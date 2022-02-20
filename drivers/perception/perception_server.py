import sys
import os
import shutil
import time
import torch
sys.path.append('../../')
from server_util import SERVER_DOCK


def perception_run():
    inp_dir = f'{SERVER_DOCK}/perception_inp'
    out_dir = f'{SERVER_DOCK}/perception_out'

    shutil.rmtree(inp_dir)
    shutil.rmtree(out_dir)
    os.makedirs(inp_dir)
    os.makedirs(out_dir)

    while True:
        filenames = sorted(os.listdir(inp_dir))
        if len(filenames) == 0:
            continue
        filename = filenames[0]
        print(filename)
        tensor = torch.load(f'{inp_dir}/{filename}')

        os.remove(f'{inp_dir}/{filename}')

        """
        DO YOUR COMPUTATION HERE
        """

        with open(f'{out_dir}/{filename}', 'w') as fout:
            fout.write('AKSJGFHKJSAFGHJKFKASFGJ')
