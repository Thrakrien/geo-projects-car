import numpy as np
from PIL import Image

from src.dataset import PatchGrid, extract_patch_pair


def test_extract_patch_pair_preserves_spatial_grid_index():
    """Patch index must map to the same aligned image/mask crop."""
    image_np = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask_np = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    image = Image.fromarray(image_np)
    mask = Image.fromarray(mask_np)
    grid = PatchGrid(image_size=8, patch_size=4, step=2)

    image_patch, mask_patch = extract_patch_pair(
        image=image,
        mask=mask,
        grid=grid,
        patch_idx=5,
    )

    assert np.array_equal(np.array(image_patch), image_np[2:6, 4:8, :])
    assert np.array_equal(np.array(mask_patch), mask_np[2:6, 4:8])
