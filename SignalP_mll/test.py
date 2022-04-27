import numpy as np
from sklearn import preprocessing
import torch

max_len = 70
def padding(list):
    str = list[0]
    while(len(str)<max_len):
        str = str + "X"
    return [str]
if __name__ == '__main__':
    seq = ["AETCZAO", "SKTZP"]
    list = [" ".join("".join(padding(sample.split()))) for sample in seq]
    print(list)