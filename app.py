#!pip install langchain !pip install youtube-transcript-api
# Install the necessary libraries

# Import the required module
from langchain.document_loaders import YoutubeLoader #class langchain.document_loaders.youtube.YoutubeLoader(video_id: str, add_video_info: bool = False, language: Union[str, Sequence[str]] = 'en', translation: str = 'en', continue_on_failure: bool = False)[source] #Methods :from_youtube_url(youtube_url, **kwargs)
import streamlit as st

# Set the URL for the YouTube video whose transcript you want to load
youtube_url = "https://www.youtube.com/shorts/CXT3iCzuODQ"

# Create a YoutubeLoader object to load the transcript from the provided URL in Japanese language
loader = YoutubeLoader.from_youtube_url(youtube_url, language="ja") #loader = YoutubeLoader.from_youtube_url(youtube_url, language="ja", add_video_info=True)

# Load the transcript content of the YouTube video
docs = loader.load()

# Display the loaded transcript content
print(docs) #docs

st.write("Docs :",docs)
