# Research Agent — AI-Powered Research Platform

An intelligent research platform built with Python that combines web search, 
vector memory, PDF analysis, and multi-agent AI to automate research workflows.
Built in 48 hours. Zero cost. 8 powerful modes.

---

## What it does

Most AI tools just answer questions. This platform **researches, remembers, 
verifies, compares, and teaches** — autonomously.

---

## 8 Modes

### 💬 Research Chat
Chat with an AI that searches the web in real time and remembers 
everything it learns. Ask follow-up questions — it maintains full 
conversation context like ChatGPT. Every answer is saved to a vector 
database so related questions are answered from memory instantly.

### ⚖️ Topic Comparison  
Enter any two topics and get a structured side-by-side research 
comparison with key findings, challenges, use cases — and a final 
verdict. Example: Python vs C++, React vs Vue, IIT vs NIT.

### 📄 PDF Chat
Upload any PDF document and have a full conversation with it. 
Ask for summaries, extract specific information, or ask questions 
about the content. Works with textbooks, research papers, reports, 
contracts — any PDF.

### 📊 Research Report Generator
Type any topic → agent autonomously researches it from multiple 
angles → generates a fully formatted, downloadable PDF report with:
- Introduction
- Key Findings  
- Applications & Use Cases
- Challenges & Limitations
- Future Outlook
- Conclusion

A complete research report in 30 seconds.

### ✅ Fact Checker
Paste any claim, news headline, or WhatsApp forward. The agent 
searches multiple sources and returns:
- **Verdict**: TRUE / FALSE / MISLEADING / UNVERIFIED
- **Confidence**: HIGH / MEDIUM / LOW
- **Evidence For and Against**
- **Cited Sources with URLs**

### 📚 Multi-Document Analysis
Upload 2-3 PDFs simultaneously and ask questions across all of them:
- "What do all these documents agree on?"
- "What contradicts between them?"
- "Which document covers X topic best?"

Perfect for comparing research papers, legal documents, or study materials.

### 🎓 Study Buddy
Upload any syllabus or notes PDF. The agent:
- Generates MCQ quiz questions
- Adjustable difficulty (Easy / Medium / Hard)
- Tracks your score
- Shows correct answers with detailed explanations
- Identifies topics needing revision

Your personal AI exam preparation tool.

### 🤖 Multi-Agent Pipeline
Three specialized AI agents collaborate autonomously:

| Agent | Role |
|-------|------|
| 🔍 Researcher | Searches web, collects and organizes facts |
| 🔎 Critic | Challenges findings, identifies gaps and bias |
| ✍️ Writer | Synthesizes everything into a final verified answer |

Watch three AIs debate and collaborate in real time to produce 
the most accurate, balanced answer possible.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.3 70B via Groq API |
| Web Search | Tavily API |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| UI | Streamlit |
| PDF Processing | PyPDF2 |
| Report Generation | ReportLab |
| Language | Python |

---

## Key Technical Concepts

**Agentic AI** — The system autonomously decides what steps to take. 
It doesn't just answer — it plans, searches, remembers, and produces 
structured outputs.

**RAG (Retrieval Augmented Generation)** — Before answering, the agent 
retrieves relevant information from its vector memory. This grounds 
responses in real data and eliminates repeated searches.

**Vector Embeddings** — Every answer is converted into a mathematical 
vector and stored in ChromaDB. Similar questions are matched using 
cosine similarity — enabling semantic memory, not just keyword matching.

**Persistent Memory** — The agent remembers everything across sessions. 
Ask about quantum computing today, ask a related question tomorrow — 
it retrieves from memory and skips the web search.

**Multi-Agent Collaboration** — Three specialized LLMs work as a team, 
each with a different role and perspective, producing more accurate and 
balanced outputs than any single agent.

---

## Why this is different

| Feature | Google | ChatGPT | This App |
|---------|--------|---------|----------|
| Real-time web search | ✅ | ❌ | ✅ |
| Remembers past research | ❌ | ❌ | ✅ |
| Reads your PDFs | ❌ | ✅ | ✅ |
| Generates downloadable reports | ❌ | ❌ | ✅ |
| Fact checks with sources | ❌ | ❌ | ✅ |
| Compares multiple documents | ❌ | ❌ | ✅ |
| Quizzes you on your notes | ❌ | ❌ | ✅ |
| Multi-agent pipeline | ❌ | ❌ | ✅ |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Kishor1114/research-agent.git
cd research-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API keys
Create a `config.py` file in the project root:
```python
GROQ_API_KEY = "your-groq-key"       # Free at console.groq.com
TAVILY_API_KEY = "your-tavily-key"   # Free at tavily.com
```

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Getting API Keys (Both Free)

**Groq API** — https://console.groq.com
- Sign up free
- Click API Keys → Create Key
- 14,400 free requests per day

**Tavily API** — https://tavily.com
- Sign up free  
- API key on dashboard
- 1000 free searches per month

---

## Project Structure
```
research-agent/
├── app.py              # Main Streamlit app (8 modes)
├── agent.py            # Terminal version
├── requirements.txt    # Dependencies
├── config.py           # API keys (not pushed to GitHub)
└── .gitignore          # Keeps secrets safe
```

---

## Built With

- **Groq** — Fastest LLM inference API available
- **Llama 3.3 70B** — Meta's open source LLM
- **ChromaDB** — Local vector database
- **Tavily** — Search API built for AI agents
- **Streamlit** — Python web app framework

---

## Project Stats

- Time to build: 48 hours
- Lines of code: ~400
- API cost: ₹0 (all free tier)
- Modes: 8
- Technologies: 7

---

## What's Next

- [ ] Voice input
- [ ] YouTube video summarizer  
- [ ] Web page chat (paste any URL)
- [ ] Deploy to Streamlit Cloud
- [ ] Auto-generated follow-up questions
- [ ] Research history dashboard