from io import BytesIO
from PIL import Image
from typing import Union, List
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

ImageLike = Union[Image.Image]  

# set up BLIP and load the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def resize_image(image: ImageLike, width: int, height: int) -> ImageLike:
    """Resize an image to the given width and height.

    Args:
        image (ImageLike): Input image (PIL Image).
        width (int): Desired width.
        height (int): Desired height.

    Returns:
        ImageLike: A resized PIL Image object.
    """
    return image.resize((width, height))


def get_width_height(size: str) -> List:
    """get width and height of the image from the given size as a string, for example - 
        size = '512x512' 

    Args:
        size (str): size described as '_width_x_height_' example '512x512'

    Returns:
        List: returns a list of interger as [width, height] extracted from the 
        given size
    """
    return [int(val) for val in size.split("x")] 

def download_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    # generate caption
    output = model.generate(
        **inputs,
        max_length=300,      # llonger generation
        num_beams=7,         # better quality
        do_sample=True,      # randomness for richness
        top_k=100,            # randomness control
        top_p=0.95,          # nucleus sampling for variety
        temperature=1.2,     # higher temperature for creativity
        early_stopping=False # full length generation
)
    
    # decode
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    return caption
