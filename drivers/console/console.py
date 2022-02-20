import sys
import os
import time
import cv2
sys.path.append('../../')
from pipe import Pipe


class ConsoleDriver:
    pipe = Pipe('console')

    @classmethod
    def print(cls, output):
        print("Console print: ", output)
        time.sleep(1)

    @classmethod
    def sync(cls):
        pass
