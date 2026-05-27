import os
import compress_pickle as pickle_cmp
import pickle as pickle_raw
import sys
import numpy as np

COMPRESS = 1
COMPRESSION = "gzip"

# NOTE: This script was created by Indra Heckenbach, but is used in several other scripts.


class SampleManager:
    def __init__(self, filename=None, compress=COMPRESS):
        self.filename = filename
        self.clear()
        self.compress = compress

    def clear(self):
        self.sample_xs = []
        self.sample_ys = []
        self.sample_keys = []

    def dump(self, data, file):
        if self.compress:
            return pickle_cmp.dump(
                data, file, compression=COMPRESSION, set_default_extension=False
            )
        else:
            return pickle_raw.dump(data, file, pickle_raw.HIGHEST_PROTOCOL)

    def load(self, file):
        if self.compress:
            return pickle_cmp.load(
                file, compression=COMPRESSION, set_default_extension=False
            )
        else:
            return pickle_raw.load(file)

    def load_samples(self):
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as file:
                print(self.filename)
                self.sample_keys, self.sample_xs, self.sample_ys = self.load(file)
                # xs = np.asarray(self.sample_xs)
                # print("Loaded samples:", xs.shape, xs.dtype)
                print("Loaded samples:", len(self.sample_xs))
        else:
            print("*** No samples exist!", self.filename)

    def save_samples(self):
        with open(self.filename, "wb") as file:
            self.dump((self.sample_keys, self.sample_xs, self.sample_ys), file)
            print("Saved samples:", len(self.sample_xs))

    def get(self):
        return self.sample_xs, self.sample_ys, self.sample_keys

    def count(self):
        return len(self.sample_keys)

    def add(self, x, y, key=None):
        self.sample_xs.append(x)
        self.sample_ys.append(y)
        if key is None:
            key = "auto:" + str(len(self.sample_xs))
        self.sample_keys.append(key)
        return key

    def remove(self, key):
        # print(self.sample_keys)
        # print(key)
        pos = self.sample_keys.index(key)
        self.remove_at(pos)

    def remove_at(self, pos):
        del self.sample_xs[pos]
        del self.sample_ys[pos]
        del self.sample_keys[pos]

    def get_at(self, pos):
        return self.sample_keys[pos], self.sample_xs[pos], self.sample_ys[pos]

    def find_sample(self, key):
        try:
            pos = self.sample_keys.index(key)
            return pos, self.sample_xs[pos], self.sample_ys[pos]
        except ValueError:
            return None, None, None

    def map_x_by_y(self):
        y_groups = {}
        for pos in range(len(self.sample_ys)):
            yv = self.sample_ys[pos]
            if yv not in y_groups:
                y_groups[yv] = []
            y_groups[yv].append(self.sample_xs[pos])
        return y_groups

    def get_by_y(self, yval):
        xx, kk = [], []
        for idx in range(len(self.sample_xs)):
            if self.sample_ys[idx] == yval:
                xx.append(self.sample_xs[idx])
                kk.append(self.sample_keys[idx])
        return xx, kk

    def shuffle(self):
        xyz = list(zip(self.sample_xs, self.sample_ys, self.sample_keys))
        np.random.shuffle(xyz)
        xs, ys, keys = zip(*xyz)
        return list(xs), list(ys), list(keys)
