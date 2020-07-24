import sys,os
import yaml

class LogWriter():
    def __init__(self, root:str, loading_log_root:str=None, start:bool=True):
        self.root = root
        self.status = []

        if loading_log_root is not None:
            self.load_log(loading_log_root)

    def load_log(self, loading_log_root:str):
        with open(loading_log_root, mode="r") as f:
            self.status = yaml.load(f)

    def update(self, log):
        self.status.append(log)
        with open(self.root, mode="w") as f:
            yaml.dump(self.status, f, indent=2)

