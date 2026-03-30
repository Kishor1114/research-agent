import json
from logger import log_llm

def verify_answer(client, question, answer, context, model="llama-3.3-70b-versatile"):
    """
    Asks the LLM to verify its own answer against the source context.
    Returns confidence level and any corrections needed.
    """
    log_llm("Verifying answer against sources...")

    prompt = f"""You are a fact verification system. Your job is to check if an answer is properly supported by the source context.

Question: {question}

Answer given: {answer}

Source context used: {context[:2000]}

Evaluate the answer and respond with ONLY this JSON:
{{
    "confidence": "high",
    "supported": true,
    "issues": "",
    "correction": ""
}}

Rules:
- confidence: "high" if answer is clearly supported, "medium" if partially supported, "low" if not well supported
- supported: true if the answer matches the sources, false if it contradicts or goes beyond sources
- issues: describe any problems with the answer (empty string if none)
- correction: suggest a correction if needed (empty string if none needed)"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        result = json.loads(raw)
        return result
    except:
        return {
            "confidence": "medium",
            "supported": True,
            "issues": "",
            "correction": ""
        }


def extract_sources(context):
    """Extract URLs from search context"""
    sources = []
    for line in context.split("\n"):
        if line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
            if url and url not in sources:
                sources.append(url)
    return sources[:3]