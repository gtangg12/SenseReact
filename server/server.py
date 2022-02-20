import sys
sys.path.append('./')
import threading
from drivers.perception.perception_server import perception_run


SERVER_DOCK = 'dock'

running_drivers = {
    'perception': perception_run,
}

threads = []

def main():
    for name, func in running_drivers.items():
        p = threading.Thread(target=func, args=(cls,), daemon=True)
        threads.append(p)
        p.start()

if __name__ == '__main__':
    main()
