from kernel_util import drivers

"""
N Programming Language
"""

class Program:
    def __init__(self, filename):
        self.compile(f'programs/{filename}.n')
        self.drivers = []

    def compile(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines().split('\n')
        for line in lines:
            print(line)

    def execute(self, prompt):
        pass

    def sync_driver(self, driver):
        self.drivers.append()

    def exec_driver(self, driver, method):
        pass


    def load(self, query):
        pass

    def store(self, query):
        pass
