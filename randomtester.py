import random
import ujson as json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

with open('qtable_navigation.json') as table_navi:
    q_table_navigation = json.load(table_navi)

prob = F.softmax(torch.tensor(q_table_navigation["8411200"]), dim=0).detach().numpy()
print(prob)
print(np.random.choice(np.flatnonzero(prob == prob.max())))