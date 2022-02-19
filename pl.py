import time
from kernel_util import drivers

"""
N Programming Language
"""
class Program:
    def __init__(self, filename):
        self.frequency = 0.1
        self.prompts, self.loop = self.compile(f'programs/{filename}.n')

    def compile(self, filename):
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
                eval(f'self.{line}')
            elif mode == 'loop:':
                loop.append(line)
        return prompts, loop

    def execute(self):
        timer = time.time()
        while True:
            states = {}
            for line in self.loop:
                print(line)
                if '=' in line:
                    var, value = [x.strip(' ') for x in line.split('=')]
                    if value in states:
                        value = states[value]
                    else:
                        value = eval(f'self.{value}')
                    states[var] = value
                else:
                    eval(f'self.{value}')
            print("AKJSDF")
            print(states)
            while (time.time() - timer < 1.0 / self.frequency):
                pass
            timer = time.time()
            print(states)

    def exec_driver(self, name, method):
        return eval(f'drivers[\'{name}\'].{method}()')

    def exec_logic(self, name, inp):
        prompt = eval(f'self.prompts[\'{name}\']\n\nHuman: {inp}:')
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

    def load(addr):
        pass

    def store(addr, value):
        pass

    def set_frequency(self, frequency):
        self.frequency = frequency
