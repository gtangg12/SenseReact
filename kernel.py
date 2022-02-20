import argparse
from process import Program, compile, execute


def main():
    program = Program('detect_fight')
    compile(program)
    execute(program)


if __name__ == '__main__':
    main()
