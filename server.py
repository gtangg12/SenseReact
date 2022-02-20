import sys
sys.path.append('./')
import threading
from drivers.perception.perception_server import perception_run

running_drivers = {
    'perception': perception_run,
}

threads = []

def main():
    for name, func in running_drivers.items():
        p = threading.Thread(target=func, args=(), daemon=True)
        threads.append(p)
        p.start()

    print(threads)

    while True:
        pass


if __name__ == '__main__':
    main()
