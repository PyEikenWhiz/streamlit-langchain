import streamlit as st

# Import the required class #!pip install transformers
from langchain import PromptTemplate, LLMChain

# Define the template with a variable named "subject"
template = "Tell me about {subject}"
# Create an instance of PromptTemplate and specify the variable names using input_variables
prompt = PromptTemplate(template=template, input_variables=["subject"])######################
# Use the format method to replace the variable "subject" with a specific value
prompt_text = prompt.format(subject="IT enginner")
# Print the resulting text
#print(prompt_text)
st.write("prompt: ",prompt_text)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

#from LLM (HuggingFacePipeline)
@st.cache_resource
def load_language_model():
    model_id = "rinna/japanese-gpt2-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

#print(llm(prompt_text))
llm = load_language_model()
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Osaka"
st.write(llm_chain.run(question))
