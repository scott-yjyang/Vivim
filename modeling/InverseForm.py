# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

    
class InverseNet(nn.Module):
    def __init__(self):
        super(InverseNet, self).__init__()
        # Regressor for the 3 * 2 affine matrix
        self.fc = nn.Sequential(
            nn.Linear(224*224*2, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 32),
            nn.ReLU(True),
            nn.Linear(32, 4)
        )

    def forward(self, x1, x2):
        # Perform the usual forward pass
        x = torch.cat((x1.view(-1, 224*224),x2.view(-1, 224*224)), dim=1)
        out=self.fc(x)
        return x1, x2, out
