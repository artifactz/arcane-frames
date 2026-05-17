import torch
import numpy as np
from . import model


model_ = model.pretrained()
model_.eval()


def from_resnet(embed: np.ndarray, prob: np.ndarray):
    array = np.concat((embed, prob), axis=0)
    tensor = torch.from_numpy(array[np.newaxis, :])
    with torch.no_grad():
        output = model_(tensor)
    return output.cpu().numpy()[0][0]
