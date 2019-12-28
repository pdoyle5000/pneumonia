import torch
from typing import List
from PIL.JpegImagePlugin import JpegImageFile
from PIL.ImageStat import Stat
import numpy as np
from simple_net import SimpleNet
from dataset import inference_composer


class PneumoniaClassifier:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> SimpleNet:
        model = SimpleNet(1)
        model.load_state_dict(torch.load(model_path))
        return model

    def predict(self, inputs: JpegImageFile) -> List[float]:
        stats = Stat(inputs)
        composer = inference_composer(
            (256, 256), np.divide(stats.stddev, 255), np.divide(stats.mean, 255)
        )
        with torch.no_grad():
            image = composer(inputs)
            output = self.model(image.float().unsqueeze(0))
            sigmoid = torch.nn.Sigmoid()
            # most CNNs return lists, wrapping this in a list to conform
            # to analysis APIs later on.
        return [float(sigmoid(output.detach().squeeze(1)))]
