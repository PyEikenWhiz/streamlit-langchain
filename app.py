# Import necessary libraries
import streamlit as st
from langchain import HuggingFacePipeline

# Function to load the language model
#@st.cache_resource 
def load_language_model():
    return HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")

# Display a loading spinner
with st.spinner('Please wait...'):
    # Load the language model
    language_model = load_language_model()
    st.write("Done!")

# Generate and display text
st.write(language_model("Once upon a time, "))
