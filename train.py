import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image as PILImage

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)

pipe.to("mps")

init_image=PILImage.open("dog.png").resize((512, 512))
mask_image=PILImage.open("mask.png").resize((512, 512))
prompt = "face of a cute cream lynx point cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

image.save("inpainted.png")
