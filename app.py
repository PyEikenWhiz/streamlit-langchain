# Import necessary libraries
import streamlit as st
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Function to load the language model
@st.cache_resource
def load_language_model():
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)
    return HuggingFacePipeline(pipeline=pipe)

# Display a loading spinner
with st.spinner('Please wait...'):
    # Load the language model
    language_model = load_language_model()
    st.write("Done!")

# Generate and display text
st.write(language_model("Once upon a time, "))
