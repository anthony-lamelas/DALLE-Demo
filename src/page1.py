import streamlit as st
import openai
import requests

def page1():
    st.title("OpenAI DALL·E Image Generation")
    st.info("#### NOTE: You can download an image by right-clicking on it and selecting 'Save image as'.")

    # Create a form to input the prompt, image size, and number of images
    with st.form(key="image_form"):
        prompt = st.text_input("Enter text prompt for image generation", placeholder="photorealistic image of Richard Feynman")
        size = st.selectbox("Select image size", ("256x256", "512x512", "1024x1024"))
        num_images = st.selectbox("Select number of images to generate", (1, 2, 3, 4))
        submit_button = st.form_submit_button("Generate")

    if submit_button and prompt:
        # Initialize OpenAI client (ensure your credentials are configured)
        client = openai.OpenAI()
        
        try:
            response = client.images.generate(
                model="dall-e-2",  # or "dall-e-3" for higher quality
                prompt=prompt,
                n=num_images,
                size=size
            )
            
            # Display each generated image
            for i, image in enumerate(response.data):
                image_url = image.url
                if image_url is not None:
                    st.image(image_url, caption=f"Generated Image {i+1}", use_container_width=True)
                else:
                    st.error(f"No URL returned for image {i+1}.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    page1()
