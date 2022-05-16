import time
import numpy as np
from pathlib import Path

def create_directory_structure(savepath):
    Path(savepath).mkdir(parents=True, exist_ok=True)

# class that stores stats similarly to an experience replay
class Logger():
    def __init__(self, headers):
        self.headers = headers
        self.stats = []

        # Add the extra headers that the logger calculates
        self.headers += ["Delta time", "FPS"]
        self.time = time.time()

    def add_stats(self, stats):
        time_ = time.time()
        time_diffs = time_ - self.time
        self.time = time_

        # add delta time and FPS to stats
        stats += (time_diffs, stats[1] // time_diffs, )
        self.stats.append(stats)

    def set_headers(self, headers):
        self.headers = headers

    def get_headers(self):
        return self.headers

    def get_stats(self):
        return self.headers, np.array(self.stats)

    def save_stats(self, filename):
        stats = np.array(self.stats)
        np.savetxt(f"{filename}",
                   stats, delimiter=",",
                   header=",".join(self.headers), fmt='%1.3f')
        print('done')

