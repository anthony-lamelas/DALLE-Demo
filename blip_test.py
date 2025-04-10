from src.utils import download_image, generate_caption
import openai
import os
from dotenv import load_dotenv


openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

client = openai.OpenAI()
image_response = client.images.generate(
                        model="dall-e-3",
                        prompt="in a colored ghibli manga theme generate a dragon",
                        n=1,
                        size="1024x1024"
                    )

image_url = image_response.data[0].url
image_pil = download_image(image_url)
caption = generate_caption(image_pil)
print(caption)



