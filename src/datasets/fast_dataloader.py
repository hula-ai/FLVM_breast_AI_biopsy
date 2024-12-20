import cv2
import torchvision.transforms.functional as TF
import numpy as np

from PIL import Image


class InMemoryFastDataloader:
    def __init__(self, img_size=512):
        self._img_size = img_size

    def __call__(self, mamm_file_path):
        image = Image.open(mamm_file_path)

        norm_image = cv2.normalize(
            np.array(image), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        image = Image.fromarray(norm_image).convert("RGB")

        image = TF.resize(image, self._img_size)
        return image
