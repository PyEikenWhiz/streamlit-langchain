import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

#from LLM (HuggingFacePipeline)
@st.cache_resource
def load_language_model():
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)
    return HuggingFacePipeline(pipeline=pipe)

#print(llm(prompt_text))
llm = load_language_model()

# Import the required class #!pip install transformers
from langchain import PromptTemplate, LLMChain 
from langchain.chains import SimpleSequentialChain

#1 Define the template with a variable named "subject"
template = "Tell me about {subject}"
# Create an instance of PromptTemplate and specify the variable names using input_variables
prompt = PromptTemplate(template=template, input_variables=["subject"])
st.write(prompt)

llm_chain1 = LLMChain(prompt=prompt, llm=llm)

#2 Define the template with a variable named "input"
template = "Summarize: {input}"
# Create an instance of PromptTemplate and specify the variable names using input_variables
prompt = PromptTemplate(template=template, input_variables=["input"])
st.write(prompt)

llm_chain2 = LLMChain(prompt=prompt, llm=llm)

overall_chain = SimpleSequentialChain(chains=[llm_chain1, llm_chain2], verbose=True)
st.write(overall_chain.run("Osaka"))
#print(overall_chain)
