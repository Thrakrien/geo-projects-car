import os
import numpy as np
import torch
import pytest
from PIL import Image
from torchvision import transforms
from tempfile import TemporaryDirectory

from src.dataset import PatchifySegmentationDataset, PatchifyInference


@pytest.fixture
def dummy_data():
    with TemporaryDirectory() as tmp:
        images_dir = os.path.join(tmp, "images")
        masks_dir = os.path.join(tmp, "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        txt_path = os.path.join(tmp, "files.txt")

        names = []
        for i in range(2):
            name = f"img_{i}.png"
            names.append(name)

            img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)

            Image.fromarray(img).save(os.path.join(images_dir, name))
            Image.fromarray(mask).save(os.path.join(masks_dir, name))

        with open(txt_path, "w") as f:
            f.write("\n".join(names))

        yield txt_path, images_dir, masks_dir
