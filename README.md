Overview

This project implements a GPU-accelerated Retrieval-Augmented Generation (RAG) system built on the Ramayan corpus.
It enables contextual question answering by combining semantic search with a locally hosted Mistral LLM from Hugging Face.
The system retrieves relevant passages from the Ramayan and generates grounded responses using a large language model.

📚 Complete RAG System for Ramayana - Beginner Friendly Guide

🎯 What You'll Learn
- What RAG (Retrieval Augmented Generation) is and why it's powerful
- How to build a complete RAG system from scratch using FREE tools
- How to evaluate your RAG system
- How to fine-tune models for better performance
- Before/After comparisons to see the improvement

---

🤔 What is RAG?

**RAG (Retrieval Augmented Generation)** combines two powerful capabilities:

1. **Retrieval**: Finding relevant information from a knowledge base (Ramayana PDF)
2. **Generation**: Using an LLM to generate accurate answers based on that retrieved information

**Why is this better than just asking an LLM directly?**
- ✅ **Up-to-date info**: LLMs have knowledge cutoffs; RAG uses YOUR documents
- ✅ **Accuracy**: Answers are grounded in actual source material. 
- ✅ **Citations**: You can show which part of the document the answer came from
- ✅ **Domain-specific**: Works with specialized knowledge (like Ramayana)

**The RAG Pipeline:**
User Question → Retrieve Relevant Passages → LLM Generates Answer → Response

What we're installing and why:

PyMuPDF: Extracts text from PDF files
sentence-transformers: Creates embeddings (vector representations of text)
chromadb: Vector database to store and search embeddings
transformers: Hugging Face library for using LLMs
langchain: Framework for building RAG applications - FREE!
