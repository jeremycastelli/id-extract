from fastai.vision.all import *

def get_y(line):
    f = ColReader(['a1','a2','a3','a4'], label_delim=' ')
    points = [[float(p[0]), float(p[1])] for p in f(line)]
    return points

