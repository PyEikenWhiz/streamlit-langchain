# Install the necessary libraries !pip install langchain pypdf
# Import the required module
from langchain.document_loaders import PyPDFLoader #class langchain.document_loaders.pdf.PyPDFLoader(file_path: str, password: Optional[Union[str, bytes]] = None)
import streamlit as st

# Set the URL for the PDF document to load
pdf_url = "https://di-acc2.com/wp-content/uploads/2023/06/tokyo_travel.pdf" # Set the query for the information you want to find in the document #query = "Tell me about Tokyo's famous sightseeing spot, 'Tokyo Skytree'."

# Create a PyPDFLoader object to load the PDF document from the provided URL
pdf_loader = PyPDFLoader(pdf_url)

# Load the content of the PDF document
results = pdf_loader.load()

# Display the results, which contain information from the PDF document related to the query #print(results)
#print("Pages :",len(results))
#print("results[2] :",results[2].page_content) #results

st.write("Pages :",len(results))
st.write("results[2] :",results[2].page_content)
