import gradio as gr
import numpy as np

from PIL import Image as PILImage


def create_block(segment, inpaint):

    def process_image(image):
        image, color_masks, masks = segment(image)
        mask_images = []
        for i in range(min(len(masks), 10)):
            segmentation = masks[i]["segmentation"]
            layer = np.ones((segmentation.shape[0], segmentation.shape[1], 3))
            for j in range(3):
                layer[:,:,j] = color_masks[i][j]
            layer = np.dstack((layer, segmentation))
            mask_images.append(PILImage.fromarray((layer * 255).astype(np.uint8)))

        return image, mask_images, mask_images, masks
    
    def select_mask(dropdown, masks):
        index = int(dropdown)
        if not masks:
            return None
        segmentation = masks[index]["segmentation"]
        base = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))
        layer = np.repeat(segmentation[:, :, np.newaxis], 3, axis=-1)
        layer = base + layer
        mask_image = PILImage.fromarray((layer * 255).astype(np.uint8))
        return mask_image

    def inpaint_image(prompt, image, mask_image):
        return inpaint(prompt, image, mask_image)

    block = gr.Blocks(css=".gradio-container")

    with block:
        with gr.Row():
            gr.Markdown("<h1><center>Replace Anything</center></h1>")

        with gr.Row():
            gr.Markdown("<h3><center>With Segment Anything, Stable Diffusion from Hugging Face 🤗, and BentoML 🍱</center></h3>")

        with gr.Row():
            image_input = gr.inputs.Image(shape=(224, 224), label="Upload an Image", type="pil")

        with gr.Row():
            segment_button = gr.Button(label="Submit", value="Segment")
        
        with gr.Row():
            segmented_image = gr.outputs.Image(label="Segmented Image", type="pil")

        with gr.Row():
            mask_gallery = gr.Gallery(label="Masks", type="pil")
        
        with gr.Row():
            mask_dropdown = gr.Dropdown([str(index) for index in range(10)], label="Select a Mask", multiselect=False)
            mask_image = gr.outputs.Image(label="Selected Mask", type="pil")

        with gr.Row():
            prompt_text = gr.Text(label="Describe the Replacement", type="text")
        
        with gr.Row():
            replace_button = gr.Button(label="Submit", value="Replace")

        with gr.Row():
            replaced_image = gr.outputs.Image(label="Replaced Image", type="pil")
        
        masks_data_state = gr.State()
        masked_images_state = gr.State()

        segment_button.click(process_image, inputs=[image_input], outputs=[segmented_image, mask_gallery, masked_images_state, masks_data_state])
        mask_dropdown.change(select_mask, inputs=[mask_dropdown, masks_data_state], outputs=[mask_image])
        replace_button.click(inpaint_image, inputs=[prompt_text, image_input, mask_image], outputs=[replaced_image])
    
    return block
