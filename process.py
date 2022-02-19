"""
    N Programming Language Parser
"""

import time
from kernel_util import drivers


class Program:
    def __init__(self, filename):
        self.frequency = 0.1
        self.prompts, self.loop = compile(f'programs/{filename}.n')
        self.states = {}


running_program = None


def execute():
    assert running_program, "No program has been run"
    timer = time.time()
    while True:
        for line in running_program.loop:
            eval(line)
        while (time.time() - timer < 1.0 / running_program.frequency):
            pass
        timer = time.time()
        print(running_program.states)


def exec_driver(name, method, *args):
    return eval(f'drivers[\'{name}\'].{method}()')


def exec_logic(name, inp):
    prompt = eval(f'running_program.prompts[\'{name}\']\n\nHuman: {inp}:')
    '''
    response = models['text_completion'].create(engine='text-davinci-001',
                                                prompt=prompt,
                                                temperature=0.3,
                                                max_tokens=512)
    text = get_response_text(response)
    '''
    start_sequence = "\nAI: "
    restart_sequence = "\nHuman: "
    start = s.rfind(start_sequence) + len(start_sequence)
    end = s.rfind(restart_sequence)
    output = text[start:end]
    return output


def load(var):
    return running_program.states[var]


def store(var, value):
    running_program.states[var] = value


def set_frequency(frequency):
    running_program.frequency = frequency


def compile(filename):
    with open(filename, 'r') as fin:
        lines = [line.strip(' \n\t') for line in fin.readlines()]

    def join_prompt(prompt):
        return '\n'.join(prompt).rstrip('\n')

    mode = None
    prompts, prompt_name = {}, None
    loop = []
    for line in lines:
        if not prompt_name and line == '':
            continue
        if line in ['logic:', 'setup:', 'loop:']:
            mode = line
            if mode == 'setup:':
                prompts[prompt_name] = join_prompt(prompts[prompt_name])
                prompt_name = None
            continue
        if mode == 'logic:':
            if ':' in line:
                if prompt_name:
                    prompts[prompt_name] = join_prompt(prompts[prompt_name])
                prompt_name = line.strip(':')
                prompts[prompt_name] = []
                continue
            prompts[prompt_name].append(line)
        elif mode == 'setup:':
            eval(f'{line}')
        elif mode == 'loop:':
            loop.append(line)
    return prompts, loop
