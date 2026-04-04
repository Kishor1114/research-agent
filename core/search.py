import requests

# =========================
# WEB SEARCH (Tavily)
# =========================
def web_search(api_key, query, max_results=3):
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results
            }
        )

        results = response.json().get("results", [])

        output = ""
        for r in results:
            output += f"Title: {r.get('title')}\n"
            output += f"URL: {r.get('url')}\n"
            output += f"Summary: {r.get('content')}\n\n"

        return output if output else "No results found."

    except Exception as e:
        return f"Search error: {str(e)}"


# =========================
# ACADEMIC SEARCH
# =========================
def academic_search(query):
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": 3,
                "fields": "title,authors,year"
            }
        )

        data = response.json()
        papers = data.get("data", [])

        if not papers:
            return None

        output = "Academic Results:\n\n"
        for p in papers:
            output += f"Title: {p.get('title')}\n"
            output += f"Year: {p.get('year')}\n"
            authors = p.get("authors", [])
            output += "Authors: " + ", ".join([a["name"] for a in authors]) + "\n\n"

        return output

    except:
        return None


# =========================
# EXTRACT SOURCES
# =========================
def extract_sources(text):
    sources = []
    for line in text.split("\n"):
        if line.startswith("URL:"):
            sources.append(line.replace("URL:", "").strip())
    return sources