def generate_answer(client, question, context, source, chat_history=[], model="llama-3.3-70b-versatile"):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful research assistant. Follow these rules strictly:
1. Answer clearly based on the context provided
2. If the context does not clearly confirm a specific fact — especially dates, years, or personal details about real people — say 'I could not find reliable information about this. Please verify from official sources.'
3. Never guess or assume specific facts
4. If uncertain say so clearly
5. ALWAYS end your answer with:
Sources:
- [Title or description]: URL"""
        }
    ]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Question: {question}\n\nContext ({source}):\n{context}"})

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def generate_comparison(client, topic, research, model="llama-3.3-70b-versatile"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Summarize in bullet points: what it is, key benefits, key challenges, best use cases."},
            {"role": "user", "content": f"Topic: {topic}\n\nResearch:\n{research}"}
        ]
    ).choices[0].message.content


def generate_verdict(client, topic1, topic2, summary1, summary2, model="llama-3.3-70b-versatile"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Give a concise 3-4 sentence verdict comparing both. Be direct."},
            {"role": "user", "content": f"Compare {topic1} vs {topic2}.\n\n{topic1}:\n{summary1}\n\n{topic2}:\n{summary2}"}
        ]
    ).choices[0].message.content