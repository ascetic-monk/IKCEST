import time
import sys


class LogWriter():
    def __init__(self, root):
        self.file_name = root + time.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        with open(self.file_name,'w') as f:
            f.write(message)

    def flush(self):
        pass
