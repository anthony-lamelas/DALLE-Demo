import streamlit as st
from src.page1 import page1
from src.page2 import page2
from src.page3 import page3
from src.page4 import page4
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


pages = {
    "Text to image": page1,
    "Image variation": page2,
    "Image edit": page3,
    "Chatbot pipeline": page4
}

# Create the selectbox in the sidebar
page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Display the selected page
pages[page]()
