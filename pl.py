"""
N Programming Language
"""

    def add_driver(self, driver):
        pass

    def add_goal(self, goal):
        pass

    def add_rule(self, rule):
        pass

    def load(self, query):
        pass

    def store(self, query):
        pass


class Program:
    def __init__(self, filename):
        self.compile(filename)

    def compile(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines().split('\n')
        for line in lines:
            eval(line.strip('\n'))

    def execute(self, prompt):
        pass
