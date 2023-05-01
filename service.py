import bentoml
import gradio as gr
import numpy as np
import torch

from bentoml.io import Image, JSON, Multipart, Text
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI
from interface import create_block
from PIL import Image as PILImage
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


DEVICE = "cuda"


class SegmentAnythingRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu","nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        self.model_type = "default"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    @bentoml.Runnable.method(batchable=False)
    def segment(self, input_image: PILImage):
        arr = np.asarray(input_image, dtype=np.uint8)
        return self.mask_generator.generate(arr)

class InpaintRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = True
 
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        self.pipe.to(DEVICE)

    @bentoml.Runnable.method(batchable=False)
    def inpaint(self, prompt, image, mask_image):
        return self.pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]


segment_anything_runner = bentoml.Runner(SegmentAnythingRunner, name="segment_anything_runner")
inpaint_runner = bentoml.Runner(InpaintRunner, name="inpaint_runner")
svc = bentoml.Service(name="replace-anything", runners=[segment_anything_runner, inpaint_runner])


@svc.api(
    input=Image(),
    output=Multipart(image=Image(), colors=JSON(), masks=JSON()),
)
def segment(input_image: PILImage):
    masks = segment_anything_runner.segment.run(input_image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    base_image = input_image.convert('RGBA')
    color_masks = []
    for mask in masks:
        segmentations = mask['segmentation']
        layer = np.ones((segmentations.shape[0], segmentations.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            layer[:,:,i] = color_mask[i]
        layer = np.dstack((layer, segmentations*0.35))
        current_masks = PILImage.fromarray((layer * 225).astype(np.uint8))
        color_masks.append(color_mask)
        base_image.alpha_composite(current_masks)
        
    return base_image, color_masks, masks


INPAINT_DIMENSIONS = (512, 512)


@svc.api(
    input=Multipart(prompt=Text(), image=Image(), mask=Image()),
    output=Image(),
)
def inpaint(prompt: str, image: PILImage, mask: PILImage):
    image = image.resize(INPAINT_DIMENSIONS)
    mask = mask.resize(INPAINT_DIMENSIONS)
    return inpaint_runner.inpaint.run(prompt, image, mask)


app = FastAPI()
app = gr.mount_gradio_app(app, create_block(segment, inpaint), path="/interface")
svc.mount_asgi_app(app, "/")
