import sys
import os
import time

import numpy as np

sys.path.append('../../')
from server import SERVER_DOCK
from speech.multilingual_tts import MultiTTS


def speech_run():
    inp_dir = f'{SERVER_DOCK}/speech_inp'
    out_dir = f'{SERVER_DOCK}/speech_out'

    model = MultiTTS('en')

    while True:
        # input
        filenames = sorted(os.listdir(inp_dir))
        if len(filenames) == 0:
            continue
        filename = filenames[0]
        print(filename)

        with open('{}/{}'.format(inp_dir, filename)) as f:
            input_text = f.read()

        # computation
        audio = model.synthesize(input_text)

        os.remove('{}/{}'.format(inp_dir, filename))

        # output
        np.savetxt('{}/{}'.format(out_dir, filename), audio)
