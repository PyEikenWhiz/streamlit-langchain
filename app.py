# Import necessary libraries
import streamlit as st  # Import the Streamlit library for building web apps
from langchain import HuggingFacePipeline  # Import the language model library
import time  # Import the time module to measure execution time

# Record the start time
start_time = time.time()

# Create a Streamlit app with a title
st.title("Text Generation with GPT-2")

# Function to load the language model
@st.cache_resource  # This decorator caches the result for better performance
def load_language_model():
    # Load the language model using HuggingFacePipeline
    language_model = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
    return language_model

# Display a loading spinner while the language model is loaded
with st.spinner('Please wait...'):
    # Load the language model
    language_model = load_language_model()
    st.write("Done!")  # Indicate that the loading is complete

# User input for text generation
user_input = st.text_input("Enter a starting phrase:", "Once upon a time,")

# Generate and display text based on user input
if user_input:
    generated_text = language_model(user_input)  # Generate text
    st.write("Generated text:")  # Display a label for the generated text
    st.write(generated_text)  # Display the generated text

# Record the end time
end_time = time.time()

# Calculate and display the execution time
execution_time = end_time - start_time
st.write(f"Execution time: {execution_time:.2f} seconds")
