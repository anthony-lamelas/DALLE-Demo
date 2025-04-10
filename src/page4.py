import streamlit as st
import openai
import os
import requests
from dotenv import load_dotenv

from src.utils import *

def page4():
    st.title("OpenAI DALL·E Image Generation (Manga Panels)")
    st.info("#### Enter a full story description. GPT-4o will break it into multiple manga panel prompts, each under 1000 characters.")

    # form to input the story description and number of panels
    with st.form(key="manga_form"):
        story_description = st.text_area(
            "Describe your manga story in detail:",
            placeholder="A cyberpunk detective on the run in a neon-lit city...",
            height=200
        )
        num_panels = st.number_input("Enter the amount of panels you want to generate below:", min_value=1, step=1)
        submit_button = st.form_submit_button("Generate Panels")

    if submit_button and story_description:
        try:
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI()

            # break the story into panel prompts
            system_msg = (
                "You are a prompt engineer for DALL·E. "
                "Take the user's story and split it into multiple panel prompts for a manga or comic, "
                "each describing a distinct scene in detail, under 1000 characters each. Make sure to include 'In colored manga theme generate the following:'" \
                "in the beggining of each prompt. "
                "Return them as a numbered list, for example:\n"
                "Panel 1: In colored manga theme generate the following: <prompt text>\n"
                "Panel 2: In colored manga theme generate the following: <prompt text>\n"
                "..."
            )
            user_msg = f"Break this story into {num_panels} separate prompts:\n\n{story_description}"

            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
            )

            content = gpt_response.choices[0].message.content
            if content is None:
                st.error("Error: GPT-4o returned None for the content.")
                return

            st.subheader("GPT-4o Breakdown of the Story into Manga Panels:")
            st.write(content)

            # parse the GPT output to extract each panel prompt
            panel_prompts = []
            lines = content.strip().split("\n")
            current_prompt = []

            for line in lines:
                if "Panel" in line and ":" in line:
                    if current_prompt:
                        panel_prompts.append("\n".join(current_prompt).strip())
                        current_prompt = []
                    current_prompt.append(line.split(":", 1)[1].strip())
                else:
                    current_prompt.append(line.strip())

            if current_prompt:
                panel_prompts.append("\n".join(current_prompt).strip())

            # truncate to the user-requested number of panels
            panel_prompts = panel_prompts[:num_panels]

            if len(panel_prompts) < num_panels:
                st.warning(
                    f"GPT-4o returned only {len(panel_prompts)} panel(s). "
                    "Try adjusting your story or the number of panels."
                )

            # generate each panel with DALL·E, then caption it, then feed that caption into the next prompt

            previous_caption = ""  # store the caption of the last image

            for idx, prompt_text in enumerate(panel_prompts, start=1):
                st.subheader(f"Panel {idx} Prompt:")

                # 3A: Combine the previous caption into the next prompt so it follows the story
                if previous_caption:
                    # Add story continuity: "make this next image based on the description of the previous image"
                    combined_prompt = (
                        "Generate this next image based on the following description "
                        f"of the previous image as you are telling a story: {previous_caption}. {prompt_text}"
                    )
                else:
                    # For the first panel, no previous caption
                    combined_prompt = prompt_text
                
                st.write(combined_prompt)

                # 3B: Generate the image from DALL·E
                try:
                    image_response = client.images.generate(
                        model="dall-e-3",
                        prompt=combined_prompt,
                        n=1,
                        size="1024x1024"
                    )

                    # 3C: Display the returned image & generate a caption for it
                    for img_data in image_response.data:
                        if img_data.url is None:
                            st.error(f"Error: No URL returned for Panel {idx}.")
                        else:
                            st.image(img_data.url, caption=f"Panel {idx} (1024x1024)", use_container_width=True)

                            # Download and caption the image
                            image_pil = download_image(img_data.url)   # returns a PIL image
                            new_caption = generate_caption(image_pil)  # returns a string

                            # Show the auto-generated caption
                            st.write(f"**Caption for Panel {idx}:** {new_caption}")

                            # Store this caption to feed into the NEXT prompt
                            previous_caption = new_caption

                except Exception as dalle_error:
                    st.error(f"Error generating image for Panel {idx}: {dalle_error}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    page4()
