import json
from logger import log_decision, log_llm

MODES = {
    "fact_check": "VERIFY A SPECIFIC CLAIM. Use this when the user asks whether a claim, statement, belief, rumor, assertion, or specific piece of information is true, false, misleading, or accurate. Strong signals: 'is it true', 'fact check', 'fact-check', 'is this claim correct', 'does evidence support', 'is it a myth', 'is this accurate'.",
    "compare": "Comparing exactly two things, X vs Y, differences between them",
    
    "report": "Generate a structured research report, write a detailed report, research report on a topic",
    
    "multi_agent": "Deep research or complex investigation requiring comprehensive analysis, multiple perspectives, pros and cons, advantages and disadvantages",
    
    "pdf_chat": "Questions about an uploaded document, summarize my PDF, answer questions from this file",
    
    "multi_doc": "Compare or analyze multiple uploaded documents",
    
    "study_buddy": "Quiz me, test me, create flashcards, generate study questions from my notes",
    
    "chat": "General research questions that do NOT ask to verify a specific claim"
}

def decide_mode(client, question, model="openai/gpt-oss-20b"):
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

IMPORTANT ROUTING RULES:

1. FACT_CHECK HAS PRIORITY when the user is asking whether a SPECIFIC CLAIM is true, false, accurate, misleading, or supported by evidence.
   Examples:
   - "Is it true that X?"
   - "Fact check this claim: X"
   - "Does evidence support the claim that X?"
   - "Is X a myth?"
   - "Is this statement accurate: X?"

2. COMPARE is for explicitly comparing TWO subjects.
   Examples:
   - "Python vs Java"
   - "Compare React and Vue"

3. PDF_CHAT is for questions about an uploaded PDF.

4. MULTI_DOC is for analyzing or comparing multiple uploaded documents.

5. STUDY_BUDDY is for quizzes, flashcards, and studying from uploaded notes.

6. REPORT is for explicitly requesting a structured research report.

7. MULTI_AGENT is for complex, comprehensive investigations requiring deep analysis or multiple perspectives.

8. CHAT is the fallback for normal questions that do not clearly match the specialized modes.

IMPORTANT:
If the user asks "Is it true that..." or asks to verify a specific claim, choose "fact_check" rather than "chat".

Respond with ONLY valid JSON:
{{"mode": "fact_check", "reason": "The user is asking to verify a specific claim.", "needs_file": false}}
"""

    try:
        print("DEBUG MODEL:", model)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
    except Exception as e:
        print("DEBUG MODEL:", model)
        print("DEBUG ERROR TYPE:", type(e).__name__)
        print("DEBUG ERROR:", str(e))
        raise

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


def decide_tool(client, question, has_pdf=False, memory_available=False, model="openai/gpt-oss-20b"):
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