import torch
from torch.nn import functional as F

def jaccard(str1, str2):

    try:
        if type(str1) is str:
            str1 = str1.lower().split()
            str2 = str2.lower().split()
        a = set(str1) 
        b = set(str2)
        if len(a) == 0 and len(b) == 0:
            return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except ArithmeticError:
        return 0

def one_hot(vec, num_classes = 2):
    return F.one_hot(torch.LongTensor(vec), num_classes)