import torch
from torch.nn import functional as F

def jaccard(str1, str2):
    
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return min(max(float(len(c)) / (len(a) + len(b) - len(c) + 1e-7), 0.0), 1.0)

def one_hot(vec, num_classes = 2):
    return F.one_hot(torch.LongTensor(vec), num_classes)