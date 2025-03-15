from io import BytesIO
from PIL import Image
from typing import Union, List

ImageLike = Union[Image.Image]  

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
