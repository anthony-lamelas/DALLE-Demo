import streamlit as st
import openai
import requests

def page4():
    st.title("OpenAI DALL·E Image Generation")
    st.info("#### NOTE: You can download an image by right-clicking on it and selecting 'Save image as'.")   
    chatprompt = "TEMPORARY"

    # Create a form to input the prompt, image size, and number of images
    with st.form(key="image_form"):

        concept = st.text_input("Describe what you want to see in deatail.", placeholder="A futuristic floating city in the clouds.")
        submit_button = st.form_submit_button("Generate")

    if submit_button and concept:
        # Initialize OpenAI client 
        client = openai.OpenAI()

        try:

            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a prompt engineer for DALLE, generate clear and descriptive image prompts that will make for the best manga and comic books. Generate vivid and detailed prompts."},
                    {"role": "user", "content": f"Rewrite the following manga or comic book idea into a highly detailed prompt for DALLE: '{concept}'. It must be less than 1000 characters."}
                ]
            )

            content = gpt_response.choices[0].message.content
            if content is None:
                st.error("Error, gpt resopnse returned None type.")
                return

            prompt = content.strip()
            st.success("Refined prompt from gpt-4o.")
            st.write(prompt)

            image_response = client.images.generate(
                model="dall-e-3",  
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            # Display each image
            for i, image in enumerate(image_response.data):
                image_url = image.url
                if image_url is not None:
                    st.image(image_url, caption=f"Generated Image {i+1}", use_container_width=True)
                else:
                    st.error(f"No URL returned for image {i+1}.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    page4()
