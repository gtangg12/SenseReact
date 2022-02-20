"""
    N Programming Language Parser
"""

import time
from kernel_util import drivers


class Program:
    def __init__(self, filename):
        self.filename = filename
        self.compiled = False


running_program = None
clock_frequency = 0.1
clock_iteration = 0


def load(var):
    return running_program.states[var]


def store(var, value):
    running_program.states[var] = value


def set_clock_frequency(frequency):
    global clock_frequency
    clock_frequency = frequency


def execute(program):
    assert program.compiled, "Program is not compiled."
    global running_program, clock_frequency, clock_iteration
    program.states = {}
    running_program = program
    timer = time.time()
    while True:
        for line in running_program.loop:
            eval(line)
        while (time.time() - timer < 1.0 / clock_frequency):
            pass
        timer = time.time()
        clock_iteration += 1


def exec_driver(name, method, *args):
    if len(args):
        command = f'drivers[\'{name}\'].{method}({*args,})'
    else:
        command = f'drivers[\'{name}\'].{method}()'
    return eval(command)


def exec_logic(name, inp):
    prompt = running_program.prompts[name]
    prompt += f'\n\nHuman: {inp}:'
    '''
    response = models['text_completion'].create(engine='text-davinci-001',
                                                prompt=prompt,
                                                temperature=0.3,
                                                max_tokens=512)
    text = get_response_text(response)
    '''


    text = """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.
Given the scenario description, the assistant will do as the instructions say.

Instructions:
1) If people are fighting, say "If you don't stop, I will call the police!".
2) If one side is hurt, say "I am calling 911!".
3) If one side is dying, say "Someone get an AED. Someone start CPR! Someone pray healthcare is cheap!".
4) You can say possibly many things at once.

Human: One man is dying.
AI: Someone get an AED. Someone start CPR! Someone pray healthcare is cheap!
Human:
"""


    start_sequence = "\nAI: "
    restart_sequence = "\nHuman:"
    start = text.rfind(start_sequence) + len(start_sequence)
    end = text.rfind(restart_sequence)
    output = text[start:end]
    return output


def compile(program):
    """
    """
    def join_prompt(prompt):
        return '\n'.join(prompt).rstrip('\n')

    program_path = f'programs/{program.filename}.n'
    with open(program_path, 'r') as fin:
        lines = [line.strip(' \n\t') for line in fin.readlines()]
    mode = None
    loop = []
    prompts, prompt_name = {}, None

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
                if line[line.find('.') - 1] != '\\':
                    if prompt_name:
                        prompts[prompt_name] = join_prompt(prompts[prompt_name])
                    prompt_name = line.strip(':')
                    prompts[prompt_name] = []
                    continue
                else:
                    line = line.replace('\:', ':')
            prompts[prompt_name].append(line)
        elif mode == 'setup:':
            eval(f'{line}')
        elif mode == 'loop:':
            loop.append(line)

    program.prompts, program.loop = prompts, loop
    program.compiled = True
