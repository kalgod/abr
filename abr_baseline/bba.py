import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from const import *

class BBA():
    def __init__(self):
        pass

    def predict(self, state):
        # select bit_rate according to decision tree
        bit_rate=state[0,0,-1]*float(np.max(VIDEO_BIT_RATE)) #kbps
        buffer_size=state[0,1,-1]*BUFFER_NORM_FACTOR #s
        download_speed=state[0,2,-1] #MB/s
        delay=state[0,3,-1]*M_IN_K / BUFFER_NORM_FACTOR #ms
        next_chunk_size=state[0,4, :A_DIM] #MB
        remain_chunks=state[0,5,-1]*float(CHUNK_TIL_VIDEO_END_CAP)#number

        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)
        res=np.zeros((state.shape[0],A_DIM))
        res[:,bit_rate]=1
        return res