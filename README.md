# CallNito - AI System for Detecting Scam Calls

📌 **Project Overview**

This project focuses on designing and building an application for real-time scam call detection. It utilizes **ASR (Automatic Speech Recognition)** to transcribe Thai voice conversations and **RAG (Retrieval-Augmented Generation)** to leverage a knowledge base of Thairath news articles about various scams. The entire system is accessible via a **Line Bot** integrated with a **[Streamlit](https://streamlit.io/)** web app, powered by LangChain and the **[Google Gemini LLM](https://aistudio.google.com/)**, with modern environment management using `uv`.

---

🧾 **Introduction**

In recent years, "Call Center Gangs" have become a critical issue in Thailand 🇹🇭, causing massive financial loss and distress. Having tools for timely verification and real-time warnings is essential.

This project builds a **scam detection pipeline** that:
1.  **Listens**: Uses **[Typhoon ASR (Realtime)](https://opentyphoon.ai/)** to convert Thai speech to text instantly.
2.  **Analyzes**: Utilizes a **RAG system** with a database of real news articles from **Thairath** to serve as the "brain" for providing context to the LLM.
3.  **Detects**: Integrates **[Google Gemini LLM](https://aistudio.google.com/)** to analyze the conversation patterns against the retrieved scam context and blacklist data.
4.  **Alerts**: Sends immediate notifications to a Line Group if a high-risk conversation is detected.

To ensure the system is easy to develop and deploy, this project uses `uv` for rapid dependency and virtual environment management, and `Streamlit` to create a dashboard that is accessible to general users via a Line Bot link.

---

🚀 **Getting Started**

You can run this project on your local machine by following these steps:

### 1. Clone the Repository

```bash
$ git clone https://github.com/your-username/callnito.git
$ cd callnito
```

### 2. Create & Activate Virtual Environment

This project uses `uv` (which is much faster than pip and venv). You can learn more about it **[here](https://github.com/astral-sh/uv)**.

```bash
# 1. Create the virtual environment
$ uv venv

# 2. Activate the virtual environment
# For Windows (PowerShell):
$ .\.venv\Scripts\Activate

# For macOS/Linux:
$ source .venv/bin/activate
```

### 3. Install Dependencies

`uv` will read the `pyproject.toml` file and install all required dependencies.

```bash
$ uv sync
```

### ⚠️ Important: API Keys Setup

You must set up your environment variables first!

Create a new file named `.env` in the project's root directory and add the following keys:

```env
# [Gemini API Key](https://aistudio.google.com/app/apikey) (Required for LLM analysis)
GOOGLE_API_KEY="YOUR_OWN_GOOGLE_API_KEY"

# Line Bot Configuration
LINE_CHANNEL_ACCESS_TOKEN="your_channel_access_token"
LINE_CHANNEL_SECRET="your_channel_secret"
LINE_USER_ID="your_user_id_for_testing"

# Web URL (For Line Bot to link to Streamlit)
# You will get this after setting up ngrok/pinggy (see below)
WEB_URL="https://xxxx-xxxx.ngrok-free.app"
```

---

🗄️ **Data Pipeline: Generating the Knowledge Base**

The RAG system requires a knowledge base to function. This project includes scripts to build the vector store from Thairath news data.

**`rag_langchain.py`**
*   **Purpose**: This script processes the scam-related news content (from `thairath_articles_with_content.csv`), creates text embeddings using `BAAI/bge-m3`, and indexes them into a **FAISS/Chroma** vector store (`chroma_db_thairath`).
*   **Usage**: Run this script once to generate or update the knowledge base.
    ```bash
    $ uv run rag_langchain.py
    ```

---

📂 **Test Data for Evaluation**

To evaluate the system or test the detection capabilities, we provide a curated dataset of simulated scam calls. You can use these files to test the **ASR transcription** and **Scam Detection** accuracy.

📥 **[Download Test Data (Google Drive)](https://drive.google.com/drive/folders/11C9riAkgEDBxAFWAGzMVkiuUSfscoRAg?usp=sharing)**

**Dataset Includes:**
*   **Audio Files (`.wav`/`.mp3`)**: Simulated scam conversations and normal conversations.
*   **Transcripts**: Correct text for evaluating ASR performance.
*   **Evaluation Logs**: Example results to compare with your run.

---

💬 **Usage: Scam Detection System**

You need to run two services simultaneously:

1.  **Backend (FastAPI)**: Handles Line Bot Webhooks.
    ```bash
    $ uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Frontend (Streamlit)**: Using for Voice/Text Analysis.
    ```bash
    $ uv run streamlit run app.py
    ```

> **Note**: You must keep these two terminals running. If you close them, the application will stop working.

⚙️ **Deployment & Public URL**

To integrate with the **Line Bot** and allow external access to the **Streamlit App**, you need to expose your local ports (8000 and 8501) to the internet.

> **Why do this step last?**
> The services (FastAPI & Streamlit) must be running on your local machine *before* you create a tunnel. Tools like ngrok/pinggy need an active service to forward traffic to.

**Option A: Using [ngrok](https://ngrok.com/) (Recommended)**
1.  Run `ngrok http 8000` -> Use this URL for the **Line Webhook** (e.g., `.../callback`).
2.  Run `ngrok http 8501` -> Use this URL for the `WEB_URL` in `.env`.

**Option B: Using [pinggy.io](https://pinggy.io/)**
Run the following command to get a quick public URL:
```bash
$ ssh -p 443 -R0:localhost:8000 qr@a.pinggy.io
```

---

**Key Interface Features:**
*   **Voice Input**: Upload audio files or record directly via microphone.
*   **Real-time Transcription**: Uses **[Typhoon ASR](https://opentyphoon.ai/)** to transcribe Thai speech instantly.
*   **LLM-Powered Analysis**: The user's conversation is sent to the **RAG-powered Gemini LLM**.
*   **Risk Assessment**: The system analyzes the text against known scam patterns (from Thairath news) and provides a "Verdict" (Scam/Safe) with confidence levels.
*   **Line Alert**: If a high risk is detected, a notification is automatically FLAGGED to your Line Group.

---

🤖 **System Architecture: Google Gemini & RAG Integration**

To enhance the scam detection capabilities, we integrated RAG with the **Google Gemini LLM**.

*   **Why Google Gemini & RAG?**
    *   **Reduces Hallucination**: RAG grounds the LLM, forcing it to base its answers on "real news data" rather than making things up.
    *   **Contextual Awareness**: The system understands the latest scam trends in Thailand.
    *   **Typhoon ASR**: Specialized in Thai speech recognition, ensuring accurate transcription before analysis.

*   **Techniques Used**
    *   **RAG Pipeline (`rag_langchain.py`)**: Uses LangChain to build the pipeline, load Thairath news, create embeddings (`bge-m3`), and store them in ChromaDB.
    *   **Dynamic Prompting**: The prompt sent to Gemini is dynamically constructed, combining the "transcribed text" with the "retrieved context from RAG."
    *   **Vector Search**: Uses semantic search to find similar past scam cases to the current conversation.

---

> **Warning**: **Educational Purpose Only**
> This project and its data are intended for educational and experimental purposes. The accuracy of scam detection depends on the ASR quality and the provided knowledge base.