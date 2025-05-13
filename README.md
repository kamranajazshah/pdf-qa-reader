# PDF Q&A Reader

A **PDF Question Answering** tool built using **LangChain**, **OpenAI GPT-3.5**, and **FAISS** for document processing and interactive querying. The app allows users to upload a PDF document, process it, and ask questions based on the content of the document. The answers are generated using retrieval-augmented generation (RAG) with the OpenAI language model.

## Features

- **PDF Upload**: Upload any PDF document to interact with.
- **Q&A Interaction**: Ask questions and get AI-generated answers based on the PDF content.
- **Retrieval-Augmented Generation (RAG)**: The app uses FAISS for efficient text retrieval and OpenAI's GPT-3.5 for answering questions.
- **Text Preprocessing**: Automatically processes and cleans the text to remove unnecessary details (e.g., DOIs, page numbers).

## Tech Stack

- **LangChain**: For chaining together language models and document processing.
- **OpenAI GPT-3.5**: For generating answers to user queries.
- **FAISS**: For efficient similarity search and document indexing.
- **Streamlit**: For building the interactive web interface.
- **python-dotenv**: To manage environment variables like the OpenAI API key.
- **langchain-community**: For community-based document loaders and vector stores.
