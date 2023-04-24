import numpy as np
from PIL import Image as PILImage

# masks = np.load("masks.npy", allow_pickle=True)
# input_image = PILImage.open("dog.png")
# base_image = input_image.convert('RGBA')
# base_arr = np.asarray(base_image, dtype=np.uint8)
# for mask in masks:
#     segmentations = mask['segmentation']
#     layer = np.ones((segmentations.shape[0], segmentations.shape[1], 3))
#     color_mask = np.random.random((1, 3)).tolist()[0]
#     for i in range(3):
#         layer[:,:,i] = color_mask[i]
#     layer = np.dstack((layer, segmentations*0.35))
#     current_image = PILImage.fromarray((layer * 225).astype(np.uint8))
#     current_image.save("layer.png")
#     base_image.alpha_composite(current_image)

# base_image.save("dog_segmented.png")

import uuid

masks = np.load("masks.npy", allow_pickle=True)

# for i in range(len(masks)):
segmentation = masks[5]["segmentation"]
base = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
layer = np.repeat(segmentation[:, :, np.newaxis], 3, axis=-1)
layer = base + layer
PILImage.fromarray((layer * 255).astype(np.uint8)).save(f"masks/layer-{uuid.uuid1()}.png")
