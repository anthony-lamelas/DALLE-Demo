import streamlit as st
import openai
from PIL import Image
from io import BytesIO
from src.utils import get_width_height, resize_image

def page2():
    st.title("OpenAI DALL·E Image Variation")
    st.info("#### NOTE: You can download an image by right-clicking on it and selecting 'Save image as'.")

    with st.form(key="form"):
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg"])
        num_images = st.selectbox("Select number of images to generate", (1, 2, 3, 4))
        submit_button = st.form_submit_button("Generate")

    if submit_button and uploaded_file:
        try:
            # Open and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

            # Convert resized image to bytes
            img_byte_array = BytesIO()
            image.save(img_byte_array, format="PNG")  
            img_byte_array = img_byte_array.getvalue()

            # Initialize OpenAI client
            client = openai.OpenAI()

            # Generate variations
            response = client.images.create_variation(
                image=img_byte_array,
                model="dall-e-3",  
                n=num_images,
                size="1024x1024"
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
    page2()
