# Course 01: Prompt Engineering & LangChain Fundamentals

This repository contains hands-on labs and exercises from **Course 02** of my **Retrieval-Augmented Generation (RAG) Specialization** journey.

The focus of this course is on:
- Prompt engineering techniques
- Understanding Large Language Models (LLMs)
- Building applications using **LangChain**
- Working with IBM watsonx foundation models

---

## 📌 Objectives

By completing this course, I learned how to:

- Design effective prompts for LLMs
- Apply zero-shot, one-shot, and few-shot prompting
- Use Chain-of-Thought (CoT) prompting
- Understand self-consistency in reasoning
- Build real-world applications using LangChain
- Work with prompt templates, output parsers, memory, chains, and agents

---

## 🛠️ Setup

### Install Required Libraries

```bash
pip install langchain langchain-ibm flask pydantic

Import Required Libraries

The labs use:

langchain

langchain-ibm

ibm_watsonx_ai

pydantic

flask

IBM watsonx credentials are handled automatically in the Skills Network environment.

📂 Labs Overview
🔹 Lab 01: Prompt Engineering

📁 Lab-01-Prompt-Engineering/

Topics covered:

Basic prompts

Zero-shot prompting

One-shot prompting

Few-shot prompting

Chain-of-Thought (CoT) prompting

Self-consistency

Real-world applications of prompting

Exercises include:

Comparing LLM responses

Improving reasoning quality using prompt design

Understanding how prompt structure impacts output

🔹 Lab 02: LangChain Fundamentals

📁 Lab-02-LangChain-Fundamentals/

Topics covered:

Introduction to LangChain

Models and chat models

Prompt templates

Output parsers (JSON with Pydantic)

Document loaders and text splitters

Retrieval-based systems

Memory-enabled chatbots

Chains and multi-step workflows

Tools and agents

Exercises include:

Comparing multiple foundation models (LLaMA, Granite, Mistral)

Building structured JSON outputs

Creating a retrieval system

Implementing chat memory

Creating a LangChain agent with tools

🚀 Technologies Used

Python 3.12

LangChain

IBM watsonx.ai

Foundation Models

LLaMA

Granite

Mistral

Flask

Pydantic

📈 Learning Outcomes

This course strengthened my understanding of:

How LLMs respond to different prompt strategies

Building modular and scalable GenAI pipelines

Using LangChain abstractions effectively

Designing structured and reliable AI outputs

These concepts form the foundation for RAG-based applications and advanced GenAI systems.

✍️ Author

Muhammad Saqib
Machine Learning & Generative AI Enthusiast
RAG Specialization Learner



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
