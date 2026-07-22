# Harvard MLSys Study Assistant ("Zero to Hero")
An end-to-end production RAG application built for the **LLM Zoomcamp Final Project**. This conversational AI study assistant ingests the open-source Quarto and Markdown source chapters from the **Harvard CS249r: Machine Learning Systems** textbook (`harvard-edge/cs249r_book`). 

It provides an intuitive interface for students to learn complex hardware-aware machine learning, compilation graphs, edge AI constraints, and tinyML systems from absolute zero to hero.

> **Disclaimer**
>
> This project was developed as an educational project for the **LLM Zoomcamp Final Project**. It is **not affiliated with, endorsed by, or sponsored by Harvard University or the authors of the *CS249r: Machine Learning Systems* textbook**.
>
> The RAG knowledge base is built from the publicly available Quarto and Markdown source files from the **harvard-edge/cs249r_book**.
>
> **Repository**: <https://github.com/harvard-edge/cs249r_book>
>
> All rights to the original content belong to their respective authors and copyright holders.
>
> This application is intended solely as a study aid. AI-generated responses may occasionally contain inaccuracies or incomplete information. For authoritative and up-to-date content, please refer to the original textbook and repository.

## Problem
Machine Learning Systems (MLSys) is a multidisciplinary subject covering deep learning optimization, distributed training, quantization, compilation, TinyTorch, edge AI, and hardware acceleration. Students often struggle to locate explanations across multiple chapters and understand how concepts connect. Searching through an entire textbook interrupts learning.

## What our assistant does
Our AI assistant helps students:

Explain Machine Learning Systems concepts.
Retrieve relevant textbook sections.
Explain TinyTorch examples.
Compare techniques (e.g., quantization vs pruning).
Generate quizzes for revision.
Cite textbook chapters.

## Target Users

Undergraduate and graduate students studying Harvard CS249r: Machine Learning Systems, engineers learning TinyML, Edge AI, model optimization, and compiler systems.

## LLM Zoomcamp Project Pillars Matrix
| Course Requirement | Implementation Architecture Detail |
| :--- | :--- |
| **1. Unique Dataset & Ingestion** | Custom scraper downloads `.qmd`/`.md` files from Harvard's repo; parses text keeping code blocks intact. |
| **2. Vector & Retrieval Database** | Local **Qdrant Vector Database** container running local `all-MiniLM-L6-v2` dense embedding arrays. |
| **3. Interchangeable LLM Engine** | Unified RAG layer with zero-downtime routing toggling between **Groq (Llama 3)** and **OpenAI (GPT-4o)**. |
| **4. Evaluation Strategy** | Built-in **Concept Quiz Tester** utilizing an active LLM-as-a-Judge generation and validation mechanism. |
| **5. Monitoring & Analytics** | Thread-safe SQLite logger tracking timestamps, questions, contexts, response latencies, and user UI feedback loops. |

## System Architecture Diagram 
```text
                     ┌──────────────────┐
                     │ Harvard GitHub   │ (Fetches .qmd / .md files)
                     └─────────┬────────┘
                               │
                        [ 1. Ingest.py ]
                               │ (all-MiniLM-L6-v2 Embeddings)
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │                   DOCKER ENVIRONMENT                   │
  │                                                        │
  │  ┌──────────────┐       ┌─────────────┐                │
  │  │  Qdrant DB   │◄─────►│ RAG Engine  │                │
  │  │ (Port 6333)  │       │ (Groq/OAI)  │                │
  │  └──────────────┘       └──────┬──────┘                │
  │                                │                       │
  │                         [ 2. App.py ]                  │
  │                                │                       │
  │  ┌──────────────┐              ▼                       │
  │  │ SQLite Log   │◄────── [Streamlit UI]                │
  │  │ (Monitoring) │          (Port 8501)                 │
  │  └──────────────┘                                      │
  └────────────────────────────────────────────────────────┘
```
## Execution & Reproducibility Guide
Follow these exact steps to spin up the containerized ecosystem on your machine.
### 1. Prerequisites
Ensure you have the following installed:

- **Docker** and **Docker Compose**
- A **Groq API Key** or **OpenAI API Key**

### 2. Environment Configuration
Create a `.env` file in the root directory of your cloned folder and fill in your credentials:
```env
# Choose: groq or openai
LLM_PROVIDER=groq

GROQ_API_KEY=gsk_your_actual_groq_key_here
OPENAI_API_KEY=sk_your_actual_openai_key_here
```
### 3. Spin Up the Docker Microservices
Build and start the application container and the vector database backend in detached mode:
```bash
docker-compose up --build -d
```
### 4. Execute Knowledge Ingestion & Vectorization
Run the ingestion worker inside the container environment to pull the Harvard dataset, split the text blocks, compute local embeddings, and index them into Qdrant:
```bash
docker-compose exec app python ingest.py
```
### 5. Access the Platform UI
Once ingestion concludes successfully, open your web browser and navigate to:
```text
Open ***http://localhost:8501*** in your browser.
```
## Evaluation & Production Monitoring

- **User Feedback Tracking:** Every response includes **👍 Good Answer** and **👎 Missing Info** buttons. Feedback is logged in the local SQLite database.

- **Telemetry Dashboard:** Open **Analytics Dashboard** from the sidebar to inspect response latency, request distributions, and user feedback.

