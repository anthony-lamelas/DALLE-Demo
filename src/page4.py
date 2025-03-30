import streamlit as st
import openai
import requests

def page4():
    st.title("OpenAI DALL·E Image Generation (Manga Panels)")
    st.info("#### Enter a full story description. GPT-4o will break it into multiple manga panel prompts, each under 1000 characters.")

    # Create a form to input the story description and number of panels
    with st.form(key="manga_form"):
        story_description = st.text_area(
            "Describe your manga story in detail:",
            placeholder="A cyberpunk detective on the run in a neon-lit city...",
            height=200
        )
        num_panels = st.selectbox("How many panels (images) do you want?", (2, 3, 4, 5))
        submit_button = st.form_submit_button("Generate Panels")

    if submit_button and story_description:
        try:
            # 1) Initialize OpenAI client
            client = openai.OpenAI()

            # 2) Create a system and user message to instruct GPT-4o
            system_msg = (
                "You are a prompt engineer for DALL·E. "
                "Take the user's story and split it into multiple panel prompts for a manga or comic, "
                "each describing a distinct scene in detail, under 1000 characters each. "
                "Return them as a numbered list, for example:\n"
                "Panel 1: <prompt text>\n"
                "Panel 2: <prompt text>\n"
                "..."
            )
            user_msg = f"Break this story into {num_panels} separate prompts:\n\n{story_description}"

            # 3) Request completion from GPT-4o
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

            # 4) Parse the GPT output to extract each panel’s prompt
            panel_prompts = []
            lines = content.strip().split("\n")
            current_prompt = []

            for line in lines:
                # Look for lines in the format "Panel X: something"
                if "Panel" in line and ":" in line:
                    # If we were building a prompt, store it before starting a new one
                    if current_prompt:
                        panel_prompts.append("\n".join(current_prompt).strip())
                        current_prompt = []
                    # Start a new prompt with the part after the colon
                    current_prompt.append(line.split(":", 1)[1].strip())
                else:
                    current_prompt.append(line.strip())

            # Don't forget the last prompt if it exists
            if current_prompt:
                panel_prompts.append("\n".join(current_prompt).strip())

            # Truncate to the user-requested number of panels
            panel_prompts = panel_prompts[:num_panels]

            if len(panel_prompts) < num_panels:
                st.warning(
                    f"GPT-4o returned only {len(panel_prompts)} panel(s). "
                    "Try adjusting your story or the number of panels."
                )

            # 5) For each panel prompt, generate a DALL·E image
            for idx, prompt_text in enumerate(panel_prompts, start=1):
                st.subheader(f"Panel {idx} Prompt:")
                st.write(prompt_text)

                try:
                    # Always use dall-e-3, single image at 1024x1024
                    image_response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt_text,
                        n=1,
                        size="1024x1024"
                    )

                    # Correctly iterate over the returned data
                    for img_data in image_response.data:
                        if img_data.url is None:
                            st.error(f"Error: No URL returned for Panel {idx}.")
                        else:
                            st.image(
                                img_data.url,
                                caption=f"Panel {idx} (1024x1024)",
                                use_container_width=True
                            )

                except Exception as dalle_error:
                    st.error(f"Error generating image for Panel {idx}: {dalle_error}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    page4()
