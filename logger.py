from datetime import datetime
import os

LOG_ENABLED = True

def log(step, message, emoji="🔹"):
    if not LOG_ENABLED:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {emoji} [{step}] {message}")

def log_decision(tool):
    log("AGENT", f"Decided to use: {tool.upper()}", "🧠")

def log_search(query):
    log("SEARCH", f"Searching: {query}", "🌐")

def log_memory(hit=True):
    if hit:
        log("MEMORY", "Found relevant memory", "💾")
    else:
        log("MEMORY", "No relevant memory found", "❌")

def log_llm(action):
    log("LLM", action, "⚡")

def log_result(source, confidence=""):
    log("RESULT", f"Source: {source} {confidence}", "✅")