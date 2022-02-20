import argparse
import os
import time
import torch
from queue import Queue
from paramiko.client import SSHClient
from dotenv import load_dotenv
from server_util import SERVER_DOCK


LOCAL_DOCK = 'dock'

load_dotenv()
client = SSHClient()
client.load_system_host_keys()
client.connect('satori-login-002.mit.edu',
               username='gtangg12',
               password=os.environ['GTANGG12_PASSWORD'])

remote_base_dir = '/nobackup/users/gtangg12/SenseReactServer'
remote_client = client.open_sftp()


def timestamp_name():
    return str(time.time() * 100000).split('.')[0]


class RemoteQueue():
    def __init__(self, name):
        self.name = name
        self.send_dir = f'{SERVER_DOCK}/{name}_inp'
        self.recv_dir = f'{SERVER_DOCK}/{name}_out'

    def put(self, client_path):
        filename = client_path.split('/')[1]
        remote_client_path = f'{remote_base_dir}/{self.send_dir}/{filename}'
        remote_client.put(client_path, remote_client_path)
        os.remove(client_path)

    def get(self, client_path):
        return
        """ DEADLOCK """
        remote_recv_dir = f'{remote_base_dir}/{self.recv_dir}'
        print(remote_recv_dir)
        print(remote_client.listdir(remote_recv_dir))
        return
        filenames = sorted(remote_client.listdir(remote_recv_dir))
        print(filenames)
        return
        if len(filenames) == 0:
            return False
        filename = filenames[0]
        print(filename)
        remote_client_path = f'{remote_recv_dir}/{filename}'
        remote_client.get(remote_client_path, client_path)
        remote_client.remove(remote_client_path)
        return True


pipes = {}


class Pipe:
    """ Unidirectional and local """
    def __init__(self, name):
        if name not in pipes.keys():
            pipes[name] = Queue()
        self.buffer = pipes[name]

    def put(self, value):
        self.buffer.put(value)

    def get(self):
        return self.buffer.get()

    def __len__(self):
        return self.buffer.qsize()


class Tunnel:
    """ Bidirectional through ego (essentially one send/one recv queues) """
    # should implement put, get (outer), read, write (inner)
    pass


class RemoteTunnel:
    """ Bidirectional through ego and remote """
    def __init__(self, name):
        if name not in pipes.keys():
            pipes[name] = RemoteQueue(name)
        self.buffer = pipes[name]

    def put(self, client_path):
        self.buffer.put(client_path)

    def get(self, client_path):
        return self.buffer.get(client_path)
