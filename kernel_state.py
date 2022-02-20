import sys

clock_frequency = 10
clock_iteration = 0

def increment_clock_iteration():
    global clock_iteration
    clock_iteration += 1


running_program = None
threads = []


def set_running_program(program):
    global running_program
    running_program = program


def get_running_program():
    global running_program
    return running_program


def cleanup_threads():
    pass
