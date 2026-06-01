# Digital Hospital Assistant using FAISS CPU RAG

An intelligent, document-grounded hospital helpdesk system built using 
Retrieval-Augmented Generation (RAG), FAISS CPU vector search, NVIDIA 
Llama-3 language models, and a Gradio-based web interface. The system 
processes hospital PDF documents and delivers real-time, citation-backed 
answers to patient and staff queries.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Reference](#reference)
- [License](#license)

---

## Overview

Traditional hospital helpdesk systems rely on manual support or 
rule-based chatbots, which are slow, inconsistent, and unable to 
process large unstructured documents. This project addresses those 
limitations by building an AI-powered assistant that:

- Ingests hospital PDFs (policies, FAQs, guidelines)
- Converts them into searchable vector embeddings using NVIDIA NV-Embed
- Retrieves the most relevant context via FAISS similarity search
- Generates accurate, grounded responses using NVIDIA Llama-3
- Delivers answers through a clean, streaming Gradio interface

The system is fully CPU-optimized, making it accessible to institutions 
without dedicated GPU hardware.

---

## Features

- PDF knowledge base ingestion with automatic chunking
- FAISS CPU-based vector similarity search
- RAG pipeline with context-grounded response generation
- Streaming responses with inline source citations
- Voice input support via Facebook Wav2Vec2 speech-to-text
- Adjustable top-K retrieval parameter via UI slider
- Raw retrieval output tab for transparency and debugging
- Lightweight deployment requiring no GPU

---

## System Architecture

The system follows a four-layer architecture:

1. Input Layer
   - Accepts PDF uploads and text/voice queries
   - Captures session metadata and chat history

2. Processing Layer
   - Extracts and cleans text using PyPDFLoader
   - Splits documents into chunks using RecursiveCharacterTextSplitter
   - Generates dense vector embeddings via NVIDIA NV-Embed

3. Retrieval and Generation Layer
   - Performs top-K similarity search using FAISS
   - Builds a structured context window from retrieved chunks
   - Generates grounded responses via NVIDIA Llama-3
   - Validates output to minimize hallucinations

4. Storage Layer
   - FAISS vector index for document embeddings
   - Cache store for frequent queries and LLM responses
   - Future scope: persistent database for logs and query history

---

## Tech Stack

| Component              | Technology                          |
|------------------------|-------------------------------------|
| Language               | Python 3.10+                        |
| LLM                    | NVIDIA Llama-3 (meta/llama3-8b-instruct) |
| Embeddings             | NVIDIA NV-Embed (nvidia/nv-embed-v1)|
| Vector Store           | FAISS CPU                           |
| Orchestration          | LangChain                           |
| PDF Processing         | PyPDFLoader (LangChain Community)   |
| Speech-to-Text         | Facebook Wav2Vec2 (Hugging Face)    |
| UI Framework           | Gradio                              |
| Deep Learning Runtime  | PyTorch                             |

---

## Project Structure

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A valid NVIDIA API key with access to Llama-3 and NV-Embed models
- pip package manager

### Steps

1. Clone the repository

```bash
git clone https://github.com/your-username/digital-hospital-assistant-rag.git
cd digital-hospital-assistant-rag
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure your API key

```bash
cp .env.example .env
# Open .env and replace the placeholder with your actual NVIDIA API key
```

5. Run the application

```bash
python app/app.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`.

---

## Usage

1. Open the application in your browser.
2. Upload a hospital knowledge base PDF using the file upload section.
3. Click "Build Knowledge Base" and wait for the confirmation message.
4. Type your question in the chat input or record a voice query and 
   click "Transcribe Audio".
5. Press "Send" or hit Enter to receive a streamed, citation-backed answer.
6. Use the "Raw Retrieval Output" tab to inspect the context retrieved 
   from your document.
7. Adjust the K slider to control how many document chunks are retrieved 
   per query.

---

## Performance Metrics

digital-hospital-assistant-rag/
│
├── app/
│   └── app.py
│
├── assets/
│   └── architecture.png
│
├── docs/
│   └── reference_paper.pdf
│
├── sample_data/
│   └── hospital_knowledge_base_sample.pdf
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md


### System Performance

| Metric                        | Value            |
|-------------------------------|------------------|
| Average Query Processing Time | 180 to 320 ms    |
| FAISS Retrieval Latency       | 25 to 40 ms      |
| LLM Inference Time            | 120 to 250 ms    |
| CPU Usage                     | 18 to 30 percent |
| Memory Usage                  | 350 to 520 MB    |
| Vector Index Size             | 18 to 25 MB      |
| Document Processing Throughput| 200 to 350 pages/min |

### Detection Accuracy

| Metric                  | Value   |
|-------------------------|---------|
| True Positive Rate      | 94.6%   |
| False Positive Rate     | 3.2%    |
| True Negative Rate      | 97.9%   |
| False Negative Rate     | 5.4%    |
| Overall Accuracy        | 96.6%   |

### Response Time Analysis

| Parameter                    | Time        |
|------------------------------|-------------|
| Average Query Response Time  | 2.1 seconds |
| Context Retrieval (FAISS)    | 0.3 seconds |
| LLM Generation Time          | 0.5 seconds |
| System Recovery Time         | 1.2 seconds |

---

## Reference

R. Naveen, P. Deepan, P. Manish, M. Sri Moukthika, M. Uday Kumar,
"Digital Hospital Assistant using FAISS CPU RAG",
Department of Artificial Intelligence and Machine Learning,
St. Peter's Engineering College, Hyderabad, India.

---

## License

This project is licensed under the MIT License.
See the LICENSE file for full details.
