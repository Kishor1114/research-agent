import streamlit as st
import time
from logger import log_llm, log_search, log_memory, log_result

def run_chat(client, question, context, source, chat_history, elapsed_start):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful research assistant. Follow these rules:
1. Answer clearly based on context provided
2. If uncertain about specific facts say so clearly
3. Never guess dates, years or personal details
4. End your answer with: Sources: [list relevant URLs if available]"""
        }
    ]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nContext ({source}):\n{context}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

def run_compare(client, web_search_fn, question):
    import re
    # Extract two topics from question
    log_llm("Extracting topics to compare...")
    extract = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Extract exactly two topics being compared from this question. Respond with JSON only: {{\"topic1\": \"...\", \"topic2\": \"...\"}}\n\nQuestion: {question}"
        }],
        max_tokens=80
    )
    raw = extract.choices[0].message.content.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json","").strip()
        topics = __import__('json').loads(raw)
        topic1 = topics.get("topic1", "Topic 1")
        topic2 = topics.get("topic2", "Topic 2")
    except:
        topic1, topic2 = "Topic 1", "Topic 2"

    log_search(f"Comparing: {topic1} vs {topic2}")

    res1 = web_search_fn(topic1)
    res2 = web_search_fn(topic2)

    sum1 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Summarize in bullet points: what it is, key benefits, key challenges, best use cases."},
            {"role": "user", "content": f"Topic: {topic1}\n\nResearch:\n{res1}"}
        ]
    ).choices[0].message.content

    sum2 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Summarize in bullet points: what it is, key benefits, key challenges, best use cases."},
            {"role": "user", "content": f"Topic: {topic2}\n\nResearch:\n{res2}"}
        ]
    ).choices[0].message.content

    verdict = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Give a concise 3-4 sentence verdict comparing both. Be direct."},
            {"role": "user", "content": f"Compare {topic1} vs {topic2}.\n\n{topic1}:\n{sum1}\n\n{topic2}:\n{sum2}"}
        ]
    ).choices[0].message.content

    return topic1, topic2, sum1, sum2, verdict

def run_fact_check(client, web_search_fn, claim):
    log_search(f"Fact checking: {claim}")
    s1 = web_search_fn(claim)
    s2 = web_search_fn(f"is it true that {claim}")
    s3 = web_search_fn(f"{claim} fact check")
    combined = f"Search 1:\n{s1}\n\nSearch 2:\n{s2}\n\nSearch 3:\n{s3}"

    result = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an expert fact checker. Respond in EXACT format:
VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIED]
CONFIDENCE: [HIGH / MEDIUM / LOW]
SUMMARY: [2-3 sentences]
EVIDENCE FOR: [supporting evidence]
EVIDENCE AGAINST: [contradicting evidence]
SOURCES: [relevant URLs]
EXPLANATION: [detailed explanation]"""
            },
            {"role": "user", "content": f"Claim: {claim}\n\nResearch:\n{combined}"}
        ]
    ).choices[0].message.content

    return result

def run_report(client, web_search_fn, topic):
    log_search(f"Generating report on: {topic}")
    sections = {
        "Introduction": f"Give a clear introduction to {topic}.",
        "Key Findings": f"What are the most important facts about {topic}?",
        "Applications & Use Cases": f"What are real world applications of {topic}?",
        "Challenges & Limitations": f"What are the main challenges of {topic}?",
        "Future Outlook": f"What is the future of {topic}?",
        "Conclusion": f"Summarize key takeaways about {topic}."
    }
    web_context = web_search_fn(topic)
    report = {}
    for section, prompt in sections.items():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert report writer. Write clearly and professionally in paragraphs."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{web_context}"}
            ]
        )
        report[section] = response.choices[0].message.content
    return report