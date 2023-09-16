# Import necessary libraries
import streamlit as st
from langchain import HuggingFacePipeline

# Create a Streamlit app
st.title("Text Generation with GPT-2")

# Function to load the language model
@st.cache_resource
def load_language_model():
    language_model = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
    return language_model

# Display a loading spinner
with st.spinner('Please wait...'):
    # Load the language model
    language_model = load_language_model()
    st.write("Done!")

# User input for text generation
user_input = st.text_input("Enter a starting phrase:", "Once upon a time,")

# Generate and display text based on user input
if user_input:
    generated_text = language_model(user_input)
    st.write("Generated text:")
    st.write(generated_text)
