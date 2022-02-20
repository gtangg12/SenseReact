import sys
sys.path.append('../../')
from server import SERVER_DOCK


def perception_run():
    send_dir = f'{SERVER_DOCK}/perception_inp'
    recv_dir = f'{SERVER_DOCK}/perception_out'
