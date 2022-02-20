import sys
import atexit
from program import Program, compile, execute
from kernel_state import cleanup_threads


atexit.register(cleanup_threads)


def main():
    program = Program('detect_fall')
    compile(program)
    execute(program)


if __name__ == '__main__':
    main()
