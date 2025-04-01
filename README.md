# 🗞️ Personalized Daily News Summarizer (RAG + LLM)

This project is a personalized news summarization app that uses **Retrieval-Augmented Generation (RAG)** to fetch, filter, and summarize daily news articles according to a user's interests and preferred summary style.

### 🔍 Overview

The app:
- Pulls recent news articles using public APIs or RSS feeds
- Retrieves only the **most relevant articles** based on your preferences
    - **DOES NOT** chunk articles due to context issues; news articles tend to be short
- Summarizes them using a **customizable LLM-based summarizer**
- Delivers clean, readable digests in your preferred tone and format
- Collects user feedback (👍/👎 or rewrite requests) to adapt over time

---
### 💻 Example Usage
#### News Summariser
<sub><i>*Note: all prompts were selected from top headlines when creating this project.</i></sub>

Example 1: Politics
![Example Query 1](images/Example%20Query%201.png)
Example 2: Economy
![Example Query 2](images/Example%20Query%202.png)

#### Custom Tone Adaption
[... In Progress...]

---
### 🔨 Roadblocks + Solutions
| Roadblock      | Solution                                  |
|----------------|-------------------------------------------|
|No inherent 'politics' category in NewsAPI|Created custom category using keyword search via the everything endpoint |
|LLM Output irrelevant to user query| Retrieved articles via semantic similarity (FAISS), applied a similarity threshold, and returned a fallback "unsure" answer if no context was strong enough|
|Bias transparency|**Potential addition(s)**: Keep database of sources with bias scores, train new agent to scan for bias and generate score.|
|Dated articles|**Potential addition(s)**: Implement system to weight articles via date, or omit after certain timeframe.|

<sub><i>*Note: all **potential additions** have not been added yet, and are stated to address gaps in project application.</i></sub>

### ⚙️ Features

- 🔎 **Semantic Article Retrieval** using vector similarity (FAISS / Chroma)
- 🤖 **Custom Summarization Styles** (bullet points, casual, academic, etc.)
- 📰 **News API or RSS Integration** (e.g., NewsAPI, The Guardian)
- 📥 **User Preference Profiles** (topics, style, summary length)
- 🧠 **LLM Integration** using OpenAI, Mistral, or LLaMA models
- 📊 **Feedback Logging** for future RLHF or fine-tuning

---

### 🛠️ Tech Stack

| Component      | Tool                                      |
|----------------|-------------------------------------------|
| Language       | Python                                    |
| Retrieval      | FAISS / ChromaDB                          |
| Embeddings     | sentence-transformers / OpenAI Embeddings |
| LLMs           | OpenAI GPT-4 / HuggingFace models         |
| Summarization  | LangChain / Custom Prompt Templates       |
| UI (Optional)  | Streamlit or Flask                        |
| Data Storage   | JSON / SQLite / CSV                       |

---

### 📥 Example Usage

```bash
python main.py --user_profile config/user_1.json

```
### Roadblocks and Solutions
- Retrieving 3 articles, but only one relevant --> extract distance score and find threshold

