import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from multiple txt files
def extract_text_from_txt_files(files):
    documents = []
    for file in files:
        text = file.read().decode("utf-8")  # Assuming the files are in UTF-8 format
        documents.append(text)
    return documents

# Vector Space Model implementation
def vector_space_model(documents, query):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    return similarities

# BM25 model implementation
def bm25_model(documents, query):
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    return scores

# TF-IDF model implementation
def tfidf_model(documents, query):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    return similarities

# Main app
st.sidebar.title("Information Retrieval Models")
model_choice = st.sidebar.radio("Choose a model", ["Vector Space", "BM25", "TF-IDF"])

st.title("Information Retrieval using different techniques")

# File uploader for multiple txt files
uploaded_files = st.file_uploader("Upload multiple .txt files", type=["txt"], accept_multiple_files=True)
query = st.text_input("Enter your query")
submit = st.button("Submit")

if uploaded_files and query and submit:
    documents = extract_text_from_txt_files(uploaded_files)
    
    if model_choice == "Vector Space":
        st.write("Vector Space Model Results")
        similarities = vector_space_model(documents, query)
    elif model_choice == "BM25":
        st.write("BM25 Model Results")
        similarities = bm25_model(documents, query)
    else:
        st.write("TF-IDF Model Results")
        similarities = tfidf_model(documents, query)
    
    # Display the top 5 results
    results = pd.DataFrame({"Document": [file.name for file in uploaded_files], "Score": similarities})
    results = results.sort_values(by="Score", ascending=False).head(5)
    st.write(results)
