<!-- README.md -->
# Simple Langgraph RAG Agent for my MATH1072 Pre-Readings

A simple Retrieval-Augmented Generation (RAG) agent that answers questions about the MATH1072 pre-readings PDF using LangChain, LangGraph, and Chroma.

## Features
- Loads and chunks a PDF of course pre-readings  
- Builds a Chroma vector store for similarity search  
- Exposes a `retriever_tool` to fetch relevant chunks  
- Orchestrates a LangGraph state machine to call the tool as needed  

## Prerequisites
- Python 3.8 or higher 
- An OpenAI API key
