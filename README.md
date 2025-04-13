# AI-Powered YouTube Summarizer & RAG Chatbot with Gemini

# YouTube Summarizer + RAG Prompts

Welcome! This notebook demonstrates a **GenAI-powered system** that can:
-  **Summarize YouTube videos** using [gemini-1.5-flash-latest]
-  **Ask questions** about the video using **Retrieval-Augmented Generation (RAG)**
-  Store and retrieve content using **Chroma Vector DB** and **SentenceTransformer**

---

## Problem statement

**Problem:** YouTube videos are rich in information, but it's time-consuming to extract insights or answer questions.

**Goal:** Build an AI assistant that can:
- Automatically summarize any video
- Let users ask focused questions using retrieved context

**GenAI Solution:** We use Google Gemini for summarization and Q&A, enhanced with embedding-based retrieval.

---

## GenAI Capabilities Used

| Capability            | Description                                      |
|----------------------|--------------------------------------------------|
| Few-shot prompting   | Gemini is guided by examples for better output   |
| Structured output    | Summaries returned in structured JSON format     |
| Long context window  | Handles long video transcripts with chunking     |
| RAG (Retrieval-Augmented Generation) | Retrieves relevant transcript context for Q&A |

---

## Notebook Guide

1. **Setup & Installation** – Install libraries, configure Gemini, ChromaDB, SBERT
2. **Transcript Extraction** – Fetch and process YouTube transcripts
3. **Summarization (Gemini)** – Few-shot + chunked summaries in JSON
4. **Embedding & Storage** – Use SBERT + Chroma to store transcripts
5. **RAG Q&A** – Ask questions and get contextual answers using Gemini

---

## Install Required Libraries

### Code Explanation
Install library for calling youtube transcipts api, chromadb and transformers

---

## Problem Statement

Watching long videos to find key information is time-consuming. What if you could **instantly summarize any video** and **ask questions** like: _"What was said about climate change?"_ — just like chatting with the content?

This blog post walks through an interactive GenAI-powered notebook that:
- Summarizes YouTube transcripts with **Gemini 1.5**
- Uses **RAG (Retrieval-Augmented Generation)** to answer questions from video context
- Stores transcripts as embeddings with **ChromaDB**

---

## Tools & Technologies

- Google Gemini 1.5 (`generativeai`)
- ChromaDB (vector database)
- Sentence Transformers for semantic embeddings
- YouTube Transcript API
- Kaggle notebook runtime

---

## Key Capabilities Demonstrated

- **Few-shot prompting**: Gemini guided with examples for high-quality summaries
- **Structured output**: JSON summaries with `title`, `summary`, and `key_points`
- **Long context support**: Handles full transcripts by chunking
- **RAG (Retrieval-Augmented Generation)**: User queries are answered based on similar transcript chunks

---

## Notebook Structure

### 1. Setup

```python
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
```

### 2. Transcript Extraction + Summarization

The notebook uses `youtube_transcript_api` to fetch captions, breaks the transcript into manageable chunks, and feeds it into Gemini for summarization with structured JSON output.

```python
def summarize_text(transcript):
    # chunk + prompt Gemini + collect partial summaries
```

### 3. Vector Embedding + ChromaDB Storage

Each full transcript is embedded using `all-MiniLM-L6-v2`, and stored in a Chroma collection for later similarity retrieval.

```python
embedding = embedder.encode([transcript])[0]
collection.add(..., embeddings=[embedding.tolist()])
```

### 4. Ask Questions with RAG

```python
def answer_question_with_rag(query):
    # retrieve context using embedding
    # prompt Gemini to answer based on that context
    ...
```

---

## 🤖 Live Use Case

Paste a YouTube link like:

```text
https://www.youtube.com/watch?v=5MgBikgcWnY
```

Ask:
> “What does the speaker say about curiosity?”

And get a context-grounded Gemini answer in seconds!

---


