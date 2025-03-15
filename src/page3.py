import streamlit as st
import openai
from PIL import Image
from io import BytesIO
from src.utils import get_width_height, resize_image
from rembg import remove
import numpy as np

def ensure_bytes(image_data):
    """Ensure the image data is converted to bytes for processing."""
    if isinstance(image_data, Image.Image):  # If it's already a PIL image
        img_byte_array = BytesIO()
        image_data.save(img_byte_array, format="PNG")
        return img_byte_array.getvalue()
    elif isinstance(image_data, bytes):  # If it's already bytes
        return image_data
    elif isinstance(image_data, np.ndarray):  # If it's a NumPy array
        img = Image.fromarray(image_data)
        img_byte_array = BytesIO()
        img.save(img_byte_array, format="PNG")
        return img_byte_array.getvalue()
    else:
        raise TypeError("Unsupported image format returned by rembg.remove()")

def page3():
    st.title("OpenAI DALL·E Image Editing")
    st.info("#### NOTE: You can download an image by right-clicking on it and selecting 'Save image as'.")

    with st.form(key="form"):
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])
        mask_file = st.file_uploader("Choose a mask file", type=["png", "jpg"])
        prompt = st.text_input("Enter a text prompt")
        size = st.selectbox("Select image size", ("256x256", "512x512", "1024x1024"))
        num_images = st.selectbox("Select number of images to generate", (1, 2, 3, 4))
        submit_button = st.form_submit_button("Generate")

    if submit_button and uploaded_file and mask_file and prompt:
        try:
            # Open images
            our_image = Image.open(uploaded_file)
            mask_image = Image.open(mask_file)

            # Resize images to match the selected size
            width, height = get_width_height(size)
            our_image = resize_image(our_image, width, height)
            mask_image = resize_image(mask_image, width, height)

            # Display uploaded images
            st.image(our_image, caption="Uploaded Image", use_container_width=True)
            st.image(mask_image, caption="Uploaded Mask", use_container_width=True)

            # Remove background from mask using rembg
            background_removed_mask_data = remove(mask_image)  # Can return bytes, PIL Image, or ndarray
            background_removed_mask_bytes = ensure_bytes(background_removed_mask_data)  # Ensure it's bytes
            background_removed_mask = Image.open(BytesIO(background_removed_mask_bytes))  # Convert to PIL Image

            st.image(background_removed_mask, caption="Processed Mask", use_container_width=True)

            # Convert both images to byte format
            img_byte_array = BytesIO()
            our_image.save(img_byte_array, format="PNG")
            img_byte_array = img_byte_array.getvalue()

            mask_byte_array = BytesIO()
            background_removed_mask.save(mask_byte_array, format="PNG")
            mask_byte_array = mask_byte_array.getvalue()

            # Initialize OpenAI client
            client = openai.OpenAI()

            # Generate edited images
            response = client.images.edit(
                image=img_byte_array,
                mask=mask_byte_array,
                prompt=prompt,
                model="dall-e-2",  # Use "dall-e-3" for better quality
                n=num_images,
                size=size
            )

            # Display generated images
            for idx, img in enumerate(response.data):
                image_url = img.url
                if image_url:
                    st.image(image_url, caption=f"Generated Image {idx+1}", use_container_width=True)
                else:
                    st.error(f"Error retrieving image {idx+1}.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    page3()
