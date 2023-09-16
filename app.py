# Import necessary libraries
import streamlit as st
from langchain import HuggingFacePipeline

# Function to load the language model
@st.cache_resource
def load_language_model():
    return HuggingFacePipeline.from_model_id(model_id="rinna/japanese-gpt2-small", task="text-generation")

# Display a loading spinner
with st.spinner('Please wait...'):
    # Load the language model
    language_model = load_language_model()
    st.write("Done!")

# Generate and display text
st.write(language_model("生命、宇宙、そして万物についての究極の疑問の答えは, ", num_return_sequences=6))
