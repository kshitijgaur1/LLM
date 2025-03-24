import os
import streamlit as st
import pickle
import time
import nltk
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (especially for OpenAI API key)
load_dotenv()

# Download required NLTK data
try:
    nltk.download('punkt')
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="Enter news article URL")
    if url.strip():  # Only add non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one valid URL!")
        st.stop()
        
    # Step 1: Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Step 1: Loading data from URLs... ✅")
    try:
        data = loader.load()
        if not data:
            st.error("No content could be extracted from the provided URLs. Please check the URLs and try again.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Step 2: Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200,  # Optional overlap between chunks
    )
    main_placeholder.text("Step 2: Splitting data into chunks... ✅")
    docs = text_splitter.split_documents(data)
    
    if not docs:
        st.error("No text content could be extracted from the articles. Please try different URLs.")
        st.stop()

    # Step 3: Generate embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Step 3: Creating embeddings and building FAISS index... ✅")
    try:
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        # Save the FAISS index locally
        vectorstore_openai.save_local("faiss_index")
        # Save the documents separately
        with open("documents.pkl", "wb") as f:
            pickle.dump(docs, f)
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        st.stop()

    main_placeholder.success("FAISS index successfully built and saved!")

# Step 4: Query Handling
query = main_placeholder.text_input("Ask a question about the articles:")

if query:
    if os.path.exists("faiss_index"):
        main_placeholder.text("Loading FAISS index for retrieval...")
        try:
            # Load the embeddings
            embeddings = OpenAIEmbeddings()
            
            # Load the FAISS index
            vectorstore_openai = FAISS.load_local("faiss_index", embeddings)
            
            # Load the documents
            with open("documents.pkl", "rb") as f:
                docs = pickle.load(f)

            # Create retriever from loaded FAISS object
            retriever = vectorstore_openai.as_retriever()

            # Initialize the retrieval chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

            # Perform the query
            result = chain({"question": query}, return_only_outputs=True)

            # Display results
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split sources by newline
                for source in sources_list:
                    st.write(source)
        except Exception as e:
            st.error(f"Error loading FAISS index or processing query: {e}")
    else:
        st.error("FAISS index not found. Please process URLs first!")
