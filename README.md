# IBM_RAG_Specialization
Labs, notes, and code from Coursera's Generative AI Applications (Courses).
# RAG Fundamentals – IBM RAG Specialization

This module covers the fundamentals of **Retrieval-Augmented Generation (RAG)**, including
document ingestion, text chunking, embeddings, vector databases, retrievers, and
LLM-powered question answering using **LangChain** and **IBM watsonx.ai**.

The repository contains **hands-on labs** and **mini-projects** demonstrating real-world
RAG applications.

---

## 📌 Background
Large Language Models (LLMs) are powerful but limited by their training data.
Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant information
from external documents and injecting it into the generation process.

---

## 🧠 What is RAG?
RAG is a technique that combines:
- **Information Retrieval** (Vector Databases)
- **Embeddings**
- **Large Language Models**

This enables accurate, grounded, and up-to-date responses based on private or custom data.

---

## 🏗️ RAG Architecture
1. Load documents
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in a vector database
5. Retrieve relevant chunks
6. Pass retrieved context to the LLM
7. Generate final response

---

## 🎯 Objectives
- Understand RAG concepts and architecture
- Build an end-to-end RAG pipeline
- Work with IBM watsonx foundation models
- Create interactive RAG applications using Gradio

---

## ⚙️ Setup

### Installing Required Libraries
```bash
pip install langchain chromadb gradio pypdf ibm-watsonx-ai
Importing Required Libraries

Key libraries used:

langchain

langchain_ibm

chromadb

gradio

ibm_watsonx_ai

🔬 Lab: RAG Pipeline with IBM watsonx
Lab Contents

Load PDF documents

Split documents into chunks

Generate embeddings using IBM Slate

Store vectors using ChromaDB

Build a retriever

Construct a RetrievalQA chain

Create a Gradio-based RAG chatbot

Lab Topics Covered

Preprocessing

Document loading

Chunking strategy

Embedding and storage

LLM construction

LangChain integration

Prompt usage

Conversational memory

Agent-style wrapping

📂 Location: labs/

🧪 Exercises

Work with your own document

Return source documents with answers

Use an alternative LLM model

🚀 Projects
📄 Project 1: Watsonx RAG PDF Chatbot

An interactive RAG-based chatbot that allows users to upload PDF files
and ask questions using IBM Granite and Slate embeddings.

Features:

PDF ingestion

Semantic search

Context-aware answers

Gradio web interface

Tech Stack:

IBM watsonx.ai

LangChain

ChromaDB

Gradio

📂 Location: Projects/

💬 Project 2: LinkedIn Icebreaker Bot

A conversational RAG application that analyzes LinkedIn profiles and
generates personalized icebreakers and responses.

Features:

LinkedIn profile extraction

Vector-based semantic understanding

Multi-model LLM support

Session-based conversational memory

Interactive Gradio UI

Tech Stack:

LangChain

Vector Databases

LLMs (IBM Granite / LLaMA)

Gradio

📂 Location: Projects/

🛠️ Tools & Technologies

Python

IBM watsonx.ai

LangChain

ChromaDB

Gradio

Foundation Models (Granite, LLaMA)

🎓 Learning Outcome

By completing this module, you will:

Understand RAG end-to-end workflows

Build production-style RAG pipelines

Integrate LLMs with private documents

Develop interactive GenAI applications

📌 Notes

API keys are not included for security reasons

Large datasets are excluded from this repository
