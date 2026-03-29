import json
from logger import log_decision, log_llm

MODES = {
    "chat": "General research questions, current events, factual questions",
    "compare": "Comparing two things, X vs Y, differences between",
    "fact_check": "Verifying claims, is it true that, fact checking, WhatsApp forwards",
    "report": "Generate a report, write a detailed report, research report on",
    "multi_agent": "Deep research, complex analysis, thorough investigation, multiple perspectives",
    "pdf_chat": "Questions about an uploaded document, summarize my PDF, read this file",
    "multi_doc": "Compare multiple documents, analyze these files, what do these docs say",
    "study_buddy": "Quiz me, test me, create flashcards, study questions from my notes"
}

def decide_mode(client, question, model="llama-3.3-70b-versatile"):
    """
    LLM decides which mode to use based on the question.
    """
    log_llm("Detecting mode...")

    modes_desc = "\n".join([f"- {k}: {v}" for k, v in MODES.items()])

    prompt = f"""You are an AI assistant router. Based on the user question, decide which mode to use.

Available modes:
{modes_desc}

User question: {question}

Respond with ONLY a JSON object:
{{"mode": "chat", "reason": "This is a general question", "needs_file": false}}

Rules:
- needs_file is true ONLY for pdf_chat, multi_doc, study_buddy
- For compare mode, extract both topics if possible
- Be decisive — pick the best single mode
- Default to "chat" if unsure"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        decision = json.loads(raw)
        mode = decision.get("mode", "chat")
        reason = decision.get("reason", "")
        needs_file = decision.get("needs_file", False)

        if mode not in MODES:
            mode = "chat"

        log_decision(f"MODE: {mode} — {reason}")
        return mode, needs_file, reason

    except:
        log_decision("chat (fallback)")
        return "chat", False, "fallback"


def decide_tool(client, question, has_pdf=False, memory_available=False, model="llama-3.3-70b-versatile"):
    """
    LLM decides which tool to use based on the question.
    """
    tools_available = ["search", "academic_search"]
    if memory_available:
        tools_available.append("memory")
    if has_pdf:
        tools_available.append("pdf")

    log_llm("Planning which tool to use...")

    prompt = f"""You are an AI agent planner. Based on the user question, decide which tool to use.

Available tools:
- search: Search the web for general questions, news, current events, general knowledge
- academic_search: Search academic databases for questions about researchers, PhDs, papers, professors, universities
- memory: Use previously stored research (only if memory_available is True)
- pdf: Answer from uploaded PDF document (only if has_pdf is True)

User question: {question}
Memory available: {memory_available}
PDF available: {has_pdf}

Respond with ONLY a JSON object:
{{"tool": "search", "reason": "This is a general knowledge question"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        decision = json.loads(raw)
        tool = decision.get("tool", "search")
        reason = decision.get("reason", "")

        if tool not in tools_available:
            tool = "search"

        log_decision(f"TOOL: {tool} — {reason}")
        return tool

    except:
        log_decision("search (fallback)")
        return "search"