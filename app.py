import streamlit as st
from groq import Groq
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from planner import decide_tool
from logger import log_search, log_memory, log_llm, log_result
from planner import decide_mode, decide_tool
from agent_modes import run_compare, run_fact_check, run_report
import time
import io

import os
try:
    from config import GROQ_API_KEY, TAVILY_API_KEY
except ImportError:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    except:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")




if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file!")
    st.stop()
if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY not found in .env file!")
    st.stop()


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide"
)

# --- Custom CSS for polish ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-web { color: #0F6E56; font-size: 12px; }
    .source-mem { color: #854F0B; font-size: 12px; }
    .source-pdf { color: #534AB7; font-size: 12px; }
    .compare-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    .compare-title {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #534AB7;
    }
    .stMetric { background: white; padding: 12px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Setup ---
@st.cache_resource
def load_clients():
    client = Groq(api_key=GROQ_API_KEY)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma = chromadb.PersistentClient(path="./memory")
    collection = chroma.get_or_create_collection(name="research_memory")
    return client, embedder, chroma, collection

client, embedder, chroma, collection = load_clients()

# --- Web Search ---
def web_search(query):
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3
        }
    )
    results = response.json().get("results", [])
    output = ""
    for r in results:
        output += f"Title: {r['title']}\n"
        output += f"URL: {r['url']}\n"
        output += f"Summary: {r['content']}\n\n"
    return output if output else "No results found."

# --- PDF Reader ---
def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text[:6000]

# --- Memory ---
def save_to_memory(question, answer):
    text = f"Q: {question}\nA: {answer}"
    embedding = embedder.encode(text).tolist()
    existing = collection.count()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[f"mem_{existing + 1}"]
    )

def check_memory(question):
    if collection.count() == 0:
        return None
    embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    distance = results["distances"][0][0]
    if distance < 0.8:
        return results["documents"][0][0]
    return None

def academic_search(query):
    # Search Semantic Scholar - free academic API
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": 3,
                "fields": "title,authors,year,externalIds"
            }
        )
        data = response.json()
        papers = data.get("data", [])
        if not papers:
            return None
        output = "Academic search results:\n"
        for p in papers:
            output += f"Title: {p.get('title')}\n"
            output += f"Year: {p.get('year')}\n"
            authors = p.get('authors', [])
            output += f"Authors: {', '.join([a['name'] for a in authors])}\n\n"
        return output
    except:
        return None

# --- Ask ---
def ask(question, chat_history=[], pdf_context=None):
    start_time = time.time()

    # --- Step 1: Agent decides which tool to use ---
    memory_available = collection.count() > 0
    has_pdf = pdf_context is not None

    tool = decide_tool(
        client=client,
        question=question,
        has_pdf=has_pdf,
        memory_available=memory_available
    )

    # --- Step 2: Execute the chosen tool ---
    if tool == "pdf" and pdf_context:
        log_llm("Reading PDF...")
        context = f"PDF Document Content:\n{pdf_context}"
        source = "pdf"

    elif tool == "memory":
        memory = check_memory(question)
        if memory:
            log_memory(hit=True)
            context = f"From previous research:\n{memory}"
            source = "memory"
        else:
            log_memory(hit=False)
            log_search(question)
            context = web_search(question)
            source = "web"
            tool = "search"

    elif tool == "academic_search":
        log_search(f"[ACADEMIC] {question}")
        academic_context = academic_search(question)
        web_context = web_search(question)
        if academic_context:
            context = f"Academic sources:\n{academic_context}\n\nWeb sources:\n{web_context}"
        else:
            context = web_context
        source = "web"

    else:
        log_search(question)
        context = web_search(question)
        source = "web"

    # --- Step 3: Generate answer ---
    log_llm("Generating answer...")

    messages = [
        {
            "role": "system",
            "content": """You are a helpful research assistant. Follow these rules strictly:
1. Answer clearly based on the context provided
2. If the context does not clearly confirm a specific fact — especially dates, years, or personal details about real people — say 'I could not find reliable information about this. Please verify from official sources.'
3. Never guess or assume specific facts like graduation years, birth dates, or achievements of real people
4. If you are uncertain say so clearly — do not make up an answer
5. For factual questions about specific people always mention where the info came from
6. End your answer with: Sources: [list the relevant URLs from context if available]"""
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

    answer = response.choices[0].message.content

    # --- Step 4: Save to memory if confident ---
    uncertain_phrases = [
        "could not find reliable",
        "please verify",
        "i am not sure",
        "unclear",
        "i don't know"
    ]
    is_uncertain = any(phrase in answer.lower() for phrase in uncertain_phrases)

    if source == "web" and not is_uncertain:
        save_to_memory(question, answer)

    # --- Step 5: Track metrics ---
    elapsed = round(time.time() - start_time, 2)
    log_result(source, f"({elapsed}s)")

    return answer, source, elapsed, tool

# --- Compare two topics ---
def compare_topics(topic1, topic2):
    with st.spinner(f"Researching {topic1}..."):
        results1 = web_search(topic1)
        response1 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Summarize this research clearly in bullet points covering: what it is, key benefits, key challenges, and best use cases."},
                {"role": "user", "content": f"Topic: {topic1}\n\nResearch:\n{results1}"}
            ]
        )
        summary1 = response1.choices[0].message.content

    with st.spinner(f"Researching {topic2}..."):
        results2 = web_search(topic2)
        response2 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Summarize this research clearly in bullet points covering: what it is, key benefits, key challenges, and best use cases."},
                {"role": "user", "content": f"Topic: {topic2}\n\nResearch:\n{results2}"}
            ]
        )
        summary2 = response2.choices[0].message.content

    # Generate final verdict
    with st.spinner("Generating comparison verdict..."):
        verdict_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a research analyst. Give a concise verdict comparing both topics in 3-4 sentences. Be direct and opinionated."},
                {"role": "user", "content": f"Compare {topic1} vs {topic2}.\n\n{topic1} summary:\n{summary1}\n\n{topic2} summary:\n{summary2}"}
            ]
        )
        verdict = verdict_response.choices[0].message.content

    return summary1, summary2, verdict

# --- Report Generator ---
def generate_report(topic):
    sections = {
        "Introduction": f"Give a clear introduction to {topic}. What is it and why does it matter?",
        "Key Findings": f"What are the most important facts, developments, and insights about {topic}?",
        "Applications & Use Cases": f"What are the real world applications and use cases of {topic}?",
        "Challenges & Limitations": f"What are the main challenges, risks, and limitations of {topic}?",
        "Future Outlook": f"What is the future of {topic}? What trends and developments are expected?",
        "Conclusion": f"Summarize the key takeaways about {topic} in a concise conclusion."
    }

    report_content = {}

    # Search web once for context
    with st.spinner("Searching the web..."):
        web_context = web_search(topic)

    # Generate each section
    for section, prompt in sections.items():
        with st.spinner(f"Writing {section}..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research report writer. Write clearly, professionally and in depth. Use paragraphs, not bullet points."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nUse this research context:\n{web_context}"
                    }
                ]
            )
            report_content[section] = response.choices[0].message.content

    return report_content

def create_pdf(topic, report_content):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )

    # ── COLOR PALETTE ──────────────────────────────────────
    PRIMARY     = colors.HexColor('#2D2B6B')   # deep purple
    ACCENT      = colors.HexColor('#5B5BD6')   # medium purple
    LIGHT_BG    = colors.HexColor('#F4F3FF')   # lavender tint
    HIGHLIGHT   = colors.HexColor('#FFF8E7')   # warm yellow tint
    BORDER      = colors.HexColor('#D0CFFF')   # light purple border
    TEXT_DARK   = colors.HexColor('#1A1A2E')   # near black
    TEXT_GREY   = colors.HexColor('#555577')   # muted purple-grey
    WHITE       = colors.white
    GREEN_BG    = colors.HexColor('#F0FFF4')
    GREEN_BORDER= colors.HexColor('#68D391')

    # ── STYLES ─────────────────────────────────────────────
    body = ParagraphStyle('body', fontSize=10.5, fontName='Helvetica',
        textColor=TEXT_DARK, leading=17, spaceAfter=8, alignment=TA_JUSTIFY)

    bullet = ParagraphStyle('bullet', fontSize=10.5, fontName='Helvetica',
        textColor=TEXT_DARK, leading=16, spaceAfter=5,
        leftIndent=18, bulletIndent=6)

    caption = ParagraphStyle('caption', fontSize=9, fontName='Helvetica-Oblique',
        textColor=TEXT_GREY, alignment=TA_CENTER, spaceAfter=4)

    kf_style = ParagraphStyle('kf', fontSize=10, fontName='Helvetica',
        textColor=PRIMARY, leading=15, spaceAfter=4, leftIndent=8)

    W = 6.8 * inch   # full usable width
    HALF = 3.3 * inch

    story = []

    # ═══════════════════════════════════════════════════════
    # COVER
    # ═══════════════════════════════════════════════════════
    cover = Table([[
        Paragraph(f'<font color="#2D2B6B" size="24"><b>Research Report</b></font>', ParagraphStyle('ct', fontSize=24, fontName='Helvetica-Bold', textColor=PRIMARY, alignment=TA_CENTER, leading=30)),
    ]], colWidths=[W])
    cover.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), LIGHT_BG),
        ('TOPPADDING', (0,0),(-1,-1), 24),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('LEFTPADDING', (0,0),(-1,-1), 20),
        ('RIGHTPADDING', (0,0),(-1,-1), 20),
    ]))
    story.append(cover)

    # Topic title
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph(topic, ParagraphStyle('topic',
        fontSize=17, fontName='Helvetica-Bold', textColor=ACCENT,
        alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph(
        'Generated by <b>Research Agent</b>  •  kishor-research-agent.streamlit.app',
        ParagraphStyle('sub', fontSize=9, fontName='Helvetica',
            textColor=TEXT_GREY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1*inch))
    story.append(HRFlowable(width='100%', thickness=2.5, color=ACCENT, spaceAfter=16))

    # ── Quick-stats row ─────────────────────────────────────
    def stat_cell(number, label):
        return [
            Paragraph(f'<font color="#5B5BD6" size="18"><b>{number}</b></font>',
                ParagraphStyle('sn', fontSize=18, fontName='Helvetica-Bold',
                    textColor=ACCENT, alignment=TA_CENTER)),
            Paragraph(f'<font color="#555577" size="9">{label}</font>',
                ParagraphStyle('sl', fontSize=9, fontName='Helvetica',
                    textColor=TEXT_GREY, alignment=TA_CENTER)),
        ]

    stats = Table(
        [[
            stat_cell('6','Sections')[0], stat_cell('6','Sections')[1],
            Paragraph('', body),
            stat_cell('AI','Powered')[0], stat_cell('AI','Powered')[1],
            Paragraph('', body),
            stat_cell('Live','Web Data')[0], stat_cell('Live','Web Data')[1],
        ]],
        colWidths=[0.9*inch, 1.1*inch, 0.3*inch, 0.9*inch, 1.1*inch, 0.3*inch, 0.9*inch, 1.1*inch]
    )
    stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), LIGHT_BG),
        ('TOPPADDING', (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story.append(stats)
    story.append(Spacer(1, 0.18*inch))

    # ═══════════════════════════════════════════════════════
    # SECTION HELPERS
    # ═══════════════════════════════════════════════════════
    section_nums = {
        "Introduction": "01", "Key Findings": "02",
        "Applications & Use Cases": "03", "Challenges & Limitations": "04",
        "Future Outlook": "05", "Conclusion": "06"
    }

    def section_header(num, title):
        t = Table([[
            Paragraph(f'<font color="white" size="12"><b>{num}</b></font>',
                ParagraphStyle('n', fontSize=12, fontName='Helvetica-Bold',
                    textColor=WHITE, alignment=TA_CENTER)),
            Paragraph(f'<font color="white" size="13"><b>{title}</b></font>',
                ParagraphStyle('h', fontSize=13, fontName='Helvetica-Bold',
                    textColor=WHITE)),
        ]], colWidths=[0.5*inch, 6.3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), PRIMARY),
            ('TOPPADDING', (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('LEFTPADDING', (0,0),(0,0), 14),
            ('LEFTPADDING', (1,0),(1,0), 10),
            ('VALIGN', (0,0),(-1,-1), 'MIDDLE'),
        ]))
        return t

    def highlight_box(text, bg=HIGHLIGHT, border=colors.HexColor('#F6AD55')):
        t = Table([[Paragraph(text, ParagraphStyle('hb', fontSize=10,
            fontName='Helvetica', textColor=TEXT_DARK, leading=15))]],
            colWidths=[W - 0.3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), bg),
            ('TOPPADDING', (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('LEFTPADDING', (0,0),(-1,-1), 14),
            ('RIGHTPADDING', (0,0),(-1,-1), 14),
            ('LINEAFTER', (0,0),(0,-1), 4, border),
        ]))
        return t

    def two_col_bullets(items):
        """Split a list of bullets into two columns"""
        mid = (len(items) + 1) // 2
        left  = items[:mid]
        right = items[mid:]

        def col(lst):
            paras = [Paragraph(f'• {i}', ParagraphStyle('cb',
                fontSize=10, fontName='Helvetica', textColor=TEXT_DARK,
                leading=14, spaceAfter=4, leftIndent=10))
                for i in lst]
            return paras

        rows = []
        for i in range(max(len(left), len(right))):
            l = left[i]  if i < len(left)  else ''
            r = right[i] if i < len(right) else ''
            lp = Paragraph(f'• {l}', ParagraphStyle('cb2', fontSize=10,
                fontName='Helvetica', textColor=TEXT_DARK, leading=14)) if l else Paragraph('', body)
            rp = Paragraph(f'• {r}', ParagraphStyle('cb2', fontSize=10,
                fontName='Helvetica', textColor=TEXT_DARK, leading=14)) if r else Paragraph('', body)
            rows.append([lp, rp])

        t = Table(rows, colWidths=[HALF, HALF])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), LIGHT_BG),
            ('TOPPADDING', (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
            ('LEFTPADDING', (0,0),(-1,-1), 10),
            ('RIGHTPADDING', (0,0),(-1,-1), 10),
            ('VALIGN', (0,0),(-1,-1), 'TOP'),
        ]))
        return t

    def callout_box(label, text, bg=GREEN_BG, border=GREEN_BORDER):
        content = f'<b>{label}</b><br/>{text}'
        t = Table([[Paragraph(content, ParagraphStyle('cb3',
            fontSize=10, fontName='Helvetica', textColor=TEXT_DARK,
            leading=15))]],
            colWidths=[W - 0.3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), bg),
            ('TOPPADDING', (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('LEFTPADDING', (0,0),(-1,-1), 14),
            ('RIGHTPADDING', (0,0),(-1,-1), 14),
            ('BOX', (0,0),(-1,-1), 1, border),
        ]))
        return t

    # ═══════════════════════════════════════════════════════
    # RENDER EACH SECTION
    # ═══════════════════════════════════════════════════════
    for sec_idx, (section, content) in enumerate(report_content.items()):
        num = section_nums.get(section, f"0{sec_idx+1}")

        story.append(section_header(num, section))
        story.append(Spacer(1, 0.12*inch))

        # Split raw text into paragraphs
        raw_paras = [p.strip() for p in content.split('\n') if p.strip()]

        # Separate bullets from body text
        bullets_collected = []
        body_paras = []

        for p in raw_paras:
            is_bullet = (
                p.startswith('•') or p.startswith('-') or
                p.startswith('*') or
                (len(p) > 2 and p[0].isdigit() and p[1] in '.)')
            )
            if is_bullet:
                clean = p.lstrip('•-*0123456789.) ').strip()
                bullets_collected.append(clean)
            else:
                body_paras.append(p)

        # ── Introduction: opening highlight + body ──
        if section == "Introduction":
            if body_paras:
                story.append(highlight_box(
                    f'<b>Overview:</b> {body_paras[0]}',
                    bg=LIGHT_BG, border=ACCENT
                ))
                story.append(Spacer(1, 0.1*inch))
                for p in body_paras[1:]:
                    story.append(Paragraph(p, body))

        # ── Key Findings: 2-col bullet grid + body ──
        elif section == "Key Findings":
            if body_paras:
                story.append(Paragraph(body_paras[0], body))
            if bullets_collected:
                story.append(Spacer(1, 0.08*inch))
                story.append(two_col_bullets(bullets_collected))
                story.append(Spacer(1, 0.1*inch))
            for p in body_paras[1:]:
                story.append(Paragraph(p, body))

        # ── Applications: alternating highlight boxes ──
        elif section == "Applications & Use Cases":
            app_keywords = ['imaging', 'analytics', 'drug', 'monitoring',
                            'chatbot', 'virtual', 'robot', 'diagnosis']
            highlight_paras = []
            regular_paras = []
            for p in body_paras:
                if any(kw in p.lower() for kw in app_keywords) and len(p) < 300:
                    highlight_paras.append(p)
                else:
                    regular_paras.append(p)

            for i, p in enumerate(body_paras[:2]):
                story.append(Paragraph(p, body))

            if len(body_paras) >= 3:
                story.append(Spacer(1, 0.08*inch))
                mid = len(body_paras[2:]) // 2 + 1
                apps_left  = body_paras[2:2+mid]
                apps_right = body_paras[2+mid:]

                rows = []
                for l, r in zip(apps_left, apps_right):
                    rows.append([
                        Paragraph(f'<b>→</b> {l}', ParagraphStyle('al',
                            fontSize=9.5, fontName='Helvetica',
                            textColor=TEXT_DARK, leading=14)),
                        Paragraph(f'<b>→</b> {r}', ParagraphStyle('ar',
                            fontSize=9.5, fontName='Helvetica',
                            textColor=TEXT_DARK, leading=14)),
                    ])
                if rows:
                    app_table = Table(rows, colWidths=[HALF, HALF])
                    app_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0),(-1,-1), LIGHT_BG),
                        ('TOPPADDING', (0,0),(-1,-1), 7),
                        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
                        ('LEFTPADDING', (0,0),(-1,-1), 10),
                        ('RIGHTPADDING', (0,0),(-1,-1), 10),
                        ('VALIGN', (0,0),(-1,-1), 'TOP'),
                        ('INNERGRID', (0,0),(-1,-1), 0.5, BORDER),
                    ]))
                    story.append(app_table)

        # ── Challenges: warning-style boxes ──
        elif section == "Challenges & Limitations":
            challenge_icons = ['⚠', '⚠', '⚠', '⚠', '⚠', '⚠']
            for i, p in enumerate(body_paras):
                if i < 4:
                    story.append(highlight_box(
                        f'<b>{challenge_icons[i%len(challenge_icons)]} Challenge {i+1}:</b> {p}',
                        bg=colors.HexColor('#FFF5F5'),
                        border=colors.HexColor('#FC8181')
                    ))
                    story.append(Spacer(1, 0.06*inch))
                else:
                    story.append(Paragraph(p, body))

        # ── Future Outlook: callout + body ──
        elif section == "Future Outlook":
            for i, p in enumerate(body_paras):
                if i == 0:
                    story.append(callout_box('🔮 Future Outlook:', p))
                    story.append(Spacer(1, 0.1*inch))
                elif i % 3 == 0 and i > 0:
                    story.append(highlight_box(p, bg=LIGHT_BG, border=ACCENT))
                    story.append(Spacer(1, 0.06*inch))
                else:
                    story.append(Paragraph(p, body))

        # ── Conclusion: summary box + bullets if any ──
        elif section == "Conclusion":
            if body_paras:
                story.append(callout_box('📌 Key Takeaway:', body_paras[0],
                    bg=LIGHT_BG, border=ACCENT))
                story.append(Spacer(1, 0.1*inch))
            if bullets_collected:
                story.append(two_col_bullets(bullets_collected))
                story.append(Spacer(1, 0.08*inch))
            for p in body_paras[1:]:
                story.append(Paragraph(p, body))

        # ── Default: body + bullets ──
        else:
            for p in body_paras:
                story.append(Paragraph(p, body))
            if bullets_collected:
                story.append(Spacer(1, 0.06*inch))
                for b in bullets_collected:
                    story.append(Paragraph(f'• {b}', bullet))

        story.append(Spacer(1, 0.1*inch))
        story.append(HRFlowable(width='100%', thickness=0.5,
            color=BORDER, spaceAfter=12))

    # ═══════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════
    story.append(Spacer(1, 0.15*inch))
    footer = Table([[
        Paragraph(
            'This report was autonomously generated by <b>Research Agent</b> '
            'using real-time web search and AI synthesis.',
            ParagraphStyle('ft', fontSize=9, fontName='Helvetica-Oblique',
                textColor=TEXT_GREY, alignment=TA_CENTER))
    ]], colWidths=[W])
    footer.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), LIGHT_BG),
        ('TOPPADDING', (0,0),(-1,-1), 10),
        ('BOTTOMPADDING', (0,0),(-1,-1), 10),
        ('LEFTPADDING', (0,0),(-1,-1), 16),
        ('RIGHTPADDING', (0,0),(-1,-1), 16),
    ]))
    story.append(footer)

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Fact Checker ---
def fact_check(claim):
    # Search from multiple angles
    with st.spinner("Searching for evidence..."):
        search1 = web_search(claim)
        search2 = web_search(f"is it true that {claim}")
        search3 = web_search(f"{claim} fact check")

    combined_research = f"Search 1:\n{search1}\n\nSearch 2:\n{search2}\n\nSearch 3:\n{search3}"

    with st.spinner("Analyzing evidence..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert fact checker. Analyze the claim against web evidence and respond in this EXACT format:

VERDICT: [TRUE / FALSE / MISLEADING / UNVERIFIED]

CONFIDENCE: [HIGH / MEDIUM / LOW]

SUMMARY: [2-3 sentence summary of your finding]

EVIDENCE FOR: [What evidence supports this claim]

EVIDENCE AGAINST: [What evidence contradicts this claim]

SOURCES: [List the relevant URLs from the search results]

EXPLANATION: [Detailed explanation of your verdict in 2-3 paragraphs]"""
                },
                {
                    "role": "user",
                    "content": f"Claim to fact-check: {claim}\n\nWeb Research:\n{combined_research}"
                }
            ]
        )

    return response.choices[0].message.content

# --- Multi Document Comparison ---
def analyze_multiple_docs(docs_content, question):
    # Build context with all documents labeled
    combined = ""
    for i, (name, content) in enumerate(docs_content.items()):
        combined += f"--- Document {i+1}: {name} ---\n{content}\n\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an expert document analyst. When comparing documents:
- Be specific about which document says what
- Highlight agreements and contradictions clearly
- Use document names when referring to sources
- Structure your response clearly with headers"""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocuments:\n{combined}"
            }
        ]
    )
    return response.choices[0].message.content

# --- Study Buddy ---
def generate_quiz(content, num_questions=5):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an expert teacher. Generate MCQ quiz questions from the content.
Respond in this EXACT format for each question, nothing else:

Q1: [Question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [Correct letter]
EXPLANATION: [Why this is correct]

Q2: [Question here]
...and so on"""
            },
            {
                "role": "user",
                "content": f"Generate {num_questions} MCQ questions from this content:\n\n{content}"
            }
        ]
    )
    return response.choices[0].message.content

# --- Multi Agent Pipeline ---
def run_multi_agent(question):
    thoughts = []

    # --- Agent 1: Researcher ---
    with st.status("Researcher Agent working...", expanded=True) as status:
        st.write("Searching the web for information...")
        research = web_search(question)
        st.write("Searching for additional context...")
        research2 = web_search(f"{question} latest developments")

        researcher_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a Researcher Agent. Your job is to:
1. Analyze the web search results
2. Extract the most important facts
3. Organize findings clearly
4. Note any gaps or uncertainties in the research
Be thorough and factual."""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nWeb Research:\n{research}\n\nAdditional Research:\n{research2}\n\nProvide a thorough research summary."
                }
            ]
        )
        researcher_output = researcher_response.choices[0].message.content
        thoughts.append(("Researcher", researcher_output))
        status.update(label="Researcher Agent done!", state="complete")

    # --- Agent 2: Critic ---
    with st.status("Critic Agent reviewing...", expanded=True) as status:
        st.write("Analyzing researcher findings...")
        st.write("Checking for gaps and bias...")

        critic_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a Critic Agent. Your job is to:
1. Review the researcher's findings critically
2. Identify what was missed or overlooked
3. Challenge any weak or biased claims
4. Add important counterpoints or nuances
5. Suggest what additional context is needed
Be constructively critical and intellectually honest."""
                },
                {
                    "role": "user",
                    "content": f"Original Question: {question}\n\nResearcher's Findings:\n{researcher_output}\n\nCritically review these findings. What was missed? What needs more nuance?"
                }
            ]
        )
        critic_output = critic_response.choices[0].message.content
        thoughts.append(("Critic", critic_output))
        status.update(label="Critic Agent done!", state="complete")

    # --- Agent 3: Writer ---
    with st.status("Writer Agent composing...", expanded=True) as status:
        st.write("Synthesizing all findings...")
        st.write("Writing final answer...")

        writer_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a Writer Agent. Your job is to:
1. Take the researcher's findings AND the critic's feedback
2. Synthesize everything into one clear, balanced, well-structured answer
3. Address the gaps the critic identified
4. Write in a clear, engaging style
5. End with key takeaways
Produce the best possible final answer."""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nResearcher found:\n{researcher_output}\n\nCritic noted:\n{critic_output}\n\nWrite the final, comprehensive answer."
                }
            ]
        )
        writer_output = writer_response.choices[0].message.content
        thoughts.append(("Writer", writer_output))
        status.update(label="Writer Agent done!", state="complete")

    return thoughts

def parse_quiz(quiz_text):
    questions = []
    blocks = quiz_text.strip().split("\n\n")

    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 6:
            continue
        try:
            question = lines[0].split(":", 1)[1].strip() if ":" in lines[0] else lines[0]
            options = {}
            answer = ""
            explanation = ""

            for line in lines[1:]:
                if line.startswith("A)"):
                    options["A"] = line[2:].strip()
                elif line.startswith("B)"):
                    options["B"] = line[2:].strip()
                elif line.startswith("C)"):
                    options["C"] = line[2:].strip()
                elif line.startswith("D)"):
                    options["D"] = line[2:].strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()

            if question and len(options) == 4 and answer:
                questions.append({
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "explanation": explanation
                })
        except:
            continue

    return questions



# --- Sidebar ---
with st.sidebar:
    st.title("🔬 Research Agent")
    st.caption("Powered by Llama 3 + Groq")
    st.divider()

    # Mode selector
    mode = st.radio(
        "Mode",
        ["🤖 Auto (Smart Mode)", "💬 Chat", "⚖️ Compare Topics", "📄 PDF Chat", "📊 Report Generator", "✅ Fact Checker", "📚 Multi-Doc", "🎓 Study Buddy", "🤖 Multi-Agent"],
        label_visibility="collapsed"
    )

    st.divider()

    if mode == "📄 PDF Chat":
        st.subheader("📄 Upload a PDF")
        uploaded_pdf = st.file_uploader(
            "Upload any PDF",
            type="pdf"
        )
        pdf_context = None
        if uploaded_pdf:
            with st.spinner("Reading PDF..."):
                pdf_context = read_pdf(uploaded_pdf)
            st.success(f"PDF loaded!")
    else:
        pdf_context = None
        uploaded_pdf = None

    st.divider()
    st.subheader("🧠 Memory")
    st.metric("Topics stored", collection.count())
    if st.button("Clear memory"):
        try:
            chroma.delete_collection("research_memory")
        except:
            pass
        st.cache_resource.clear()
        st.success("Cleared! Restarting...")
        st.rerun()

    st.divider()
    st.caption("Built with Groq + Llama 3 + ChromaDB + Tavily")

# --- Main area ---

# =====================
# AUTO SMART MODE
# =====================
if mode == "🤖 Auto (Smart Mode)":
    st.title("🤖 Auto Research Agent")
    st.caption("Just ask anything — I decide what to do automatically")

    auto_file = st.file_uploader(
        "Upload a file (optional — for PDF/Study/Multi-doc questions)",
        type="pdf", key="auto_file"
    )

    for key, default in [
        ("auto_messages", []), ("auto_quiz_questions", []),
        ("auto_quiz_answers", {}), ("auto_quiz_submitted", False), ("auto_quiz_score", 0)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Show chat history
    for msg in st.session_state.auto_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "meta" in msg:
                st.caption(msg["meta"])

    # ── Quiz display — OUTSIDE chat input block ──
    if st.session_state.auto_quiz_questions and not st.session_state.auto_quiz_submitted:
        st.divider()
        st.subheader("📝 Quiz Time!")
        for i, q in enumerate(st.session_state.auto_quiz_questions):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            selected = st.radio(
                f"Q{i+1}",
                [f"{k}) {v}" for k, v in q["options"].items()],
                key=f"auto_q_{i}",
                label_visibility="collapsed"
            )
            st.session_state.auto_quiz_answers[i] = selected[0] if selected else ""
            st.divider()

        if st.button("Submit Quiz", key="auto_submit"):
            score = sum(1 for i, q in enumerate(st.session_state.auto_quiz_questions)
                        if st.session_state.auto_quiz_answers.get(i) == q["answer"])
            st.session_state.auto_quiz_score = score
            st.session_state.auto_quiz_submitted = True
            st.rerun()

    # ── Quiz results — OUTSIDE chat input block ──
    if st.session_state.auto_quiz_submitted and st.session_state.auto_quiz_questions:
        total = len(st.session_state.auto_quiz_questions)
        score = st.session_state.auto_quiz_score
        percentage = int((score / total) * 100)

        if percentage >= 80:
            st.success(f"Excellent! {score}/{total} ({percentage}%)")
        elif percentage >= 60:
            st.warning(f"Good effort! {score}/{total} ({percentage}%)")
        else:
            st.error(f"Needs revision! {score}/{total} ({percentage}%)")

        st.progress(percentage / 100)
        st.divider()

        for i, q in enumerate(st.session_state.auto_quiz_questions):
            your_ans = st.session_state.auto_quiz_answers.get(i, "")
            correct_ans = q["answer"]
            if your_ans == correct_ans:
                st.success(f"Q{i+1}: {q['question']}")
                st.write(f"✅ {your_ans}) {q['options'].get(your_ans, '')}")
            else:
                st.error(f"Q{i+1}: {q['question']}")
                st.write(f"❌ Your: {your_ans}) {q['options'].get(your_ans, '')}")
                st.write(f"✅ Correct: {correct_ans}) {q['options'].get(correct_ans, '')}")
            if q["explanation"]:
                st.caption(f"Explanation: {q['explanation']}")
            st.divider()

        if st.button("Retake", key="auto_retake"):
            st.session_state.auto_quiz_submitted = False
            st.session_state.auto_quiz_answers = {}
            st.session_state.auto_quiz_questions = []
            st.rerun()

    # ── Chat input — AFTER quiz display ──
    if question := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.auto_messages.append({"role": "user", "content": question})

        start_time = time.time()
        detected_mode, needs_file, reason = decide_mode(client, question)

        if needs_file and not auto_file:
            response_text = f"I detected you want to use **{detected_mode.replace('_', ' ').title()}** mode for this. Please upload a PDF file to continue!"
            with st.chat_message("assistant"):
                st.write(response_text)
                st.caption(f"🧠 Detected mode: {detected_mode} | Waiting for file upload")
            st.session_state.auto_messages.append({
                "role": "assistant", "content": response_text,
                "meta": f"🧠 Mode: {detected_mode}"
            })

        else:
            with st.chat_message("assistant"):

                if detected_mode == "chat":
                    with st.spinner("Thinking..."):
                        answer, source, elapsed, tool = ask(question, st.session_state.auto_messages)
                    st.write(answer)
                    icons = {"memory": "🧠 from memory", "web": "🌐 searched web", "pdf": "📄 from PDF"}
                    col1, col2, col3 = st.columns(3)
                    with col1: st.caption(icons.get(source, ""))
                    with col2: st.caption(f"⚡ Tool: {tool}")
                    with col3: st.caption(f"⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": answer, "meta": f"🧠 chat | ⚡ {tool} | ⏱️ {elapsed}s"})

                elif detected_mode == "compare":
                    import json as _json
                    with st.spinner("Extracting topics..."):
                        raw = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": f"Extract two topics being compared. JSON only: {{\"topic1\": \"...\", \"topic2\": \"...\"}}\n\nQuestion: {question}"}],
                            max_tokens=80
                        ).choices[0].message.content.strip()
                        try:
                            if "```" in raw: raw = raw.split("```")[1].replace("json", "").strip()
                            topics = _json.loads(raw)
                            topic1, topic2 = topics.get("topic1", "Topic 1"), topics.get("topic2", "Topic 2")
                        except:
                            topic1, topic2 = "Topic 1", "Topic 2"
                    s1, s2, verdict = compare_topics(topic1, topic2)
                    col1, col2 = st.columns(2)
                    with col1: st.subheader(topic1); st.write(s1)
                    with col2: st.subheader(topic2); st.write(s2)
                    st.divider(); st.subheader("⚡ Verdict"); st.info(verdict)
                    elapsed = round(time.time() - start_time, 2)
                    st.caption(f"🧠 compare | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": f"Compared {topic1} vs {topic2}. Verdict: {verdict}", "meta": f"🧠 compare | ⏱️ {elapsed}s"})

                elif detected_mode == "fact_check":
                        result = run_fact_check(client, web_search, question)
                        detected_verdict = "UNVERIFIED"
                        for v in ["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]:
                            if f"VERDICT: {v}" in result:
                                detected_verdict = v
                                break
                        emoji_map = {"TRUE": "✅", "FALSE": "❌", "MISLEADING": "⚠️", "UNVERIFIED": "❓"}
                        emoji = emoji_map.get(detected_verdict, "❓")
                        if detected_verdict == "TRUE":
                            st.success(f"{emoji} VERDICT: {detected_verdict}")
                        elif detected_verdict == "FALSE":
                            st.error(f"{emoji} VERDICT: {detected_verdict}")
                        elif detected_verdict == "MISLEADING":
                            st.warning(f"{emoji} VERDICT: {detected_verdict}")
                        else:
                            st.info(f"{emoji} VERDICT: {detected_verdict}")
                        formatted_result = result.replace(" CONFIDENCE:", "\nCONFIDENCE:") \
                                                 .replace(" SUMMARY:", "\nSUMMARY:")
                        st.text(formatted_result)
                        elapsed = round(time.time() - start_time, 2)
                        st.caption(f"🧠 Mode: fact_check | ⏱️ {elapsed}s")
                        st.session_state.auto_messages.append({
                            "role": "assistant",
                            "content": result,
                            "meta": f"🧠 fact_check | ⏱️ {elapsed}s"
                        })

                elif detected_mode == "report":
                    topic = question
                    for phrase in ["generate a report on", "write a report about", "create a report on", "report on"]:
                        topic = topic.lower().replace(phrase, "").strip()
                    report_content = generate_report(topic)
                    for section, content in report_content.items():
                        st.subheader(section); st.write(content); st.divider()
                    st.download_button("⬇️ Download as PDF", create_pdf(topic, report_content), f"{topic}_report.pdf", "application/pdf")
                    elapsed = round(time.time() - start_time, 2)
                    st.caption(f"🧠 report | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": f"Generated report on: {topic}", "meta": f"🧠 report | ⏱️ {elapsed}s"})

                elif detected_mode == "study_buddy" and auto_file:
                    pdf_text = read_pdf(auto_file)
                    with st.spinner("Generating quiz from your notes..."):
                        quiz_text = generate_quiz(f"Difficulty level: Medium\n\n{pdf_text}", 5)
                        questions = parse_quiz(quiz_text)
                    if questions:
                        st.session_state.auto_quiz_questions = questions
                        st.session_state.auto_quiz_answers = {}
                        st.session_state.auto_quiz_submitted = False
                        st.session_state.auto_quiz_score = 0
                        st.success(f"Quiz ready — {len(questions)} questions from your notes!")
                    else:
                        st.error("Could not generate quiz. Try again!")
                    elapsed = round(time.time() - start_time, 2)
                    st.caption(f"🧠 study_buddy | 📄 {auto_file.name} | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": f"Generated quiz from {auto_file.name}", "meta": f"🧠 study_buddy | ⏱️ {elapsed}s"})
                    st.rerun()

                elif detected_mode == "pdf_chat" and auto_file:
                    pdf_text = read_pdf(auto_file)
                    answer, source, elapsed, tool = ask(question, [], pdf_text)
                    st.write(answer)
                    st.caption(f"🧠 pdf_chat | 📄 {auto_file.name} | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": answer, "meta": f"🧠 pdf_chat | ⏱️ {elapsed}s"})

                elif detected_mode == "multi_doc" and auto_file:
                    pdf_text = read_pdf(auto_file)
                    with st.spinner("Analyzing document..."):
                        result = analyze_multiple_docs({auto_file.name: pdf_text}, question)
                    st.write(result)
                    elapsed = round(time.time() - start_time, 2)
                    st.caption(f"🧠 multi_doc | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": result, "meta": f"🧠 multi_doc | ⏱️ {elapsed}s"})

                elif detected_mode == "multi_agent":
                    thoughts = run_multi_agent(question)
                    for agent_name, output in thoughts:
                        if agent_name == "Researcher":
                            with st.expander("🔍 Researcher findings", expanded=False): st.write(output)
                        elif agent_name == "Critic":
                            with st.expander("🔎 Critic review", expanded=False): st.write(output)
                        elif agent_name == "Writer":
                            st.subheader("✍️ Final Answer"); st.write(output)
                    elapsed = round(time.time() - start_time, 2)
                    st.caption(f"🧠 multi_agent | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": "Multi-agent research complete", "meta": f"🧠 multi_agent | ⏱️ {elapsed}s"})

                else:
                    with st.spinner("Thinking..."):
                        answer, source, elapsed, tool = ask(question, st.session_state.auto_messages)
                    st.write(answer)
                    st.caption(f"🧠 {detected_mode} | ⚡ {tool} | ⏱️ {elapsed}s")
                    st.session_state.auto_messages.append({"role": "assistant", "content": answer, "meta": f"🧠 {detected_mode} | ⏱️ {elapsed}s"})


# CHAT MODE
if mode == "💬 Chat":
    st.title("💬 Research Chat")
    st.caption("Ask anything — searches the web and remembers what it learns")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "source" in msg:
                icons = {"memory": "🧠 answered from memory", "web": "🌐 searched the web", "pdf": "📄 from PDF"}
                st.caption(icons.get(msg["source"], ""))

    if question := st.chat_input("Ask a research question..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, source, elapsed, tool = ask(question, st.session_state.messages)
            st.write(answer)
            icons = {"memory": "🧠 answered from memory", "web": "🌐 searched the web", "pdf": "📄 from PDF", "academic_search": "🎓 searched academic sources"}
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{icons.get(source, '')} ")
            with col2:
                st.caption(f"⚡ Tool: {tool}")
            with col3:
                st.caption(f"⏱️ {elapsed}s")
            

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "source": source
        })

# COMPARE MODE
elif mode == "⚖️ Compare Topics":
    st.title("⚖️ Compare Two Topics")
    st.caption("Research and compare any two topics side by side")

    col1, col2 = st.columns(2)
    with col1:
        topic1 = st.text_input("Topic 1", placeholder="e.g. Python")
    with col2:
        topic2 = st.text_input("Topic 2", placeholder="e.g. JavaScript")

    if st.button("Compare Now", type="primary"):
        if topic1 and topic2:
            summary1, summary2, verdict = compare_topics(topic1, topic2)

            # Side by side display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="compare-box">
                    <div class="compare-title">{topic1}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(summary1)

            with col2:
                st.markdown(f"""
                <div class="compare-box">
                    <div class="compare-title">{topic2}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(summary2)

            # Verdict
            st.divider()
            st.subheader("Verdict")
            st.info(verdict)
        else:
            st.warning("Please enter both topics!")

# PDF MODE
elif mode == "📄 PDF Chat":
    st.title("📄 Chat with your PDF")

    if not uploaded_pdf:
        st.info("Upload a PDF from the sidebar to get started")
    else:
        st.success(f"Chatting with: {uploaded_pdf.name}")

        if "pdf_messages" not in st.session_state:
            st.session_state.pdf_messages = []

        for msg in st.session_state.pdf_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question := st.chat_input("Ask about your PDF..."):
            with st.chat_message("user"):
                st.write(question)
            st.session_state.pdf_messages.append({"role": "user", "content": question})

            with st.chat_message("assistant"):
                with st.spinner("Reading PDF..."):
                    answer, source, elapsed, tool = ask(question, st.session_state.pdf_messages, pdf_context)
                st.write(answer)
                st.caption("📄 answered from PDF")

            st.session_state.pdf_messages.append({"role": "assistant", "content": answer})

# REPORT GENERATOR MODE
elif mode == "📊 Report Generator":
    st.title("📊 Research Report Generator")
    st.caption("Generate a full research report on any topic and download it as PDF")

    topic = st.text_input(
        "Enter a topic",
        placeholder="e.g. Artificial Intelligence in Healthcare"
    )

    if st.button("Generate Report", type="primary"):
        if topic.strip():

            # Generate the report
            report_content = generate_report(topic)

            # Show preview on screen
            st.success("Report generated!")
            st.divider()

            for section, content in report_content.items():
                st.subheader(section)
                st.write(content)
                st.divider()

            # Download button
            pdf_buffer = create_pdf(topic, report_content)
            st.download_button(
                label="Download as PDF",
                data=pdf_buffer,
                file_name=f"{topic.replace(' ', '_')}_report.pdf",
                mime="application/pdf",
                type="primary"
            )

        else:
            st.warning("Please enter a topic!")


# FACT CHECKER MODE
elif mode == "✅ Fact Checker":
    st.title("✅ Fact Checker")
    st.caption("Paste any claim, news headline, or WhatsApp forward — I'll verify it with web sources")

    # Example claims to try
    with st.expander("Try these examples"):
        examples = [
            "India is the most populous country in the world",
            "Drinking warm water cures cancer",
            "5G towers spread COVID-19",
            "Electric vehicles are worse for environment than petrol cars",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.fact_claim = ex

    claim = st.text_area(
        "Enter a claim to verify",
        value=st.session_state.get("fact_claim", ""),
        placeholder="e.g. India has the largest army in the world",
        height=100
    )

    if st.button("Verify Claim", type="primary"):
        if claim.strip():
            result = fact_check(claim)

            # Parse verdict for color coding
            verdict_color = {
                "TRUE": "success",
                "FALSE": "error",
                "MISLEADING": "warning",
                "UNVERIFIED": "info"
            }
            verdict_emoji = {
                "TRUE": "✅",
                "FALSE": "❌",
                "MISLEADING": "⚠️",
                "UNVERIFIED": "❓"
            }

            # Detect verdict from result
            detected = "UNVERIFIED"
            for v in ["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"]:
                if f"VERDICT: {v}" in result:
                    detected = v
                    break

            # Show verdict banner
            emoji = verdict_emoji.get(detected, "❓")
            if detected == "TRUE":
                st.success(f"{emoji} VERDICT: {detected}")
            elif detected == "FALSE":
                st.error(f"{emoji} VERDICT: {detected}")
            elif detected == "MISLEADING":
                st.warning(f"{emoji} VERDICT: {detected}")
            else:
                st.info(f"{emoji} VERDICT: {detected}")

            st.divider()

            # Show full analysis
            st.subheader("Full Analysis")
            st.write(result)

            # Save to memory
            save_to_memory(f"Fact check: {claim}", result)

        else:
            st.warning("Please enter a claim to verify!")

# MULTI-DOC MODE
elif mode == "📚 Multi-Doc":
    st.title("📚 Multi-Document Analysis")
    st.caption("Upload 2-3 PDFs and ask questions across all of them")

    # Upload multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload 2-3 PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 documents to compare")
        else:
            # Read all PDFs
            docs_content = {}
            for file in uploaded_files:
                with st.spinner(f"Reading {file.name}..."):
                    content = read_pdf(file)
                    docs_content[file.name] = content

            # Show loaded docs
            st.success(f"Loaded {len(uploaded_files)} documents!")
            cols = st.columns(len(uploaded_files))
            for i, file in enumerate(uploaded_files):
                with cols[i]:
                    st.info(f"📄 {file.name}")

            st.divider()

            # Quick action buttons
            st.subheader("Quick analysis")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("What do they agree on?"):
                    st.session_state.multidoc_q = "What are the main points all documents agree on?"
            with col2:
                if st.button("What contradicts?"):
                    st.session_state.multidoc_q = "What points or information contradict between these documents?"
            with col3:
                if st.button("Summarize all"):
                    st.session_state.multidoc_q = "Give me a concise summary of each document and then an overall synthesis"

            # Custom question
            question = st.text_input(
                "Or ask your own question",
                value=st.session_state.get("multidoc_q", ""),
                placeholder="e.g. Which document is most relevant to AI in healthcare?"
            )

            if st.button("Analyze", type="primary"):
                if question.strip():
                    with st.spinner("Analyzing all documents..."):
                        result = analyze_multiple_docs(docs_content, question)

                    st.subheader("Analysis")
                    st.write(result)

                    # Save to memory
                    save_to_memory(question, result)
                else:
                    st.warning("Please enter a question!")
    else:
        # Show instructions when no files uploaded
        st.info("Upload PDFs from the file uploader above to get started")

        st.subheader("What you can do")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Compare research papers**
            Upload 2-3 papers on the same topic and find what they agree and disagree on

            **Analyze legal documents**
            Upload contracts or policies and find differences between them
            """)
        with col2:
            st.markdown("""
            **Study multiple textbook chapters**
            Upload chapters and get a unified summary

            **Compare company reports**
            Upload annual reports and analyze differences year over year
            """)

# STUDY BUDDY MODE
elif mode == "🎓 Study Buddy":
    st.title("🎓 Study Buddy")
    st.caption("Upload your notes — get quizzed instantly")

    # Initialize session state
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    # Upload notes
    study_pdf = st.file_uploader("Upload your notes or syllabus PDF", type="pdf")

    if study_pdf:
        with st.spinner("Reading your notes..."):
            study_content = read_pdf(study_pdf)
        st.success(f"Notes loaded — {len(study_content)} characters read!")

        col1, col2 = st.columns(2)
        with col1:
            num_q = st.slider("Number of questions", 3, 10, 5)
        with col2:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Creating your quiz..."):
                quiz_text = generate_quiz(
                    f"Difficulty level: {difficulty}\n\n{study_content}",
                    num_q
                )
                questions = parse_quiz(quiz_text)

            if questions:
                st.session_state.quiz_questions = questions
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.success(f"Quiz ready — {len(questions)} questions!")
            else:
                st.error("Could not generate quiz. Try again!")

    # Show quiz
    if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
        st.divider()
        st.subheader("Answer all questions")

        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            options_list = [f"{k}) {v}" for k, v in q["options"].items()]
            selected = st.radio(
                f"Select answer for Q{i+1}",
                options_list,
                key=f"q_{i}",
                label_visibility="collapsed"
            )
            st.session_state.quiz_answers[i] = selected[0] if selected else ""
            st.divider()

        if st.button("Submit Quiz", type="primary"):
            score = 0
            for i, q in enumerate(st.session_state.quiz_questions):
                if st.session_state.quiz_answers.get(i) == q["answer"]:
                    score += 1
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            st.rerun()

    # Show results
    if st.session_state.quiz_submitted and st.session_state.quiz_questions:
        total = len(st.session_state.quiz_questions)
        score = st.session_state.quiz_score
        percentage = int((score / total) * 100)

        st.divider()

        # Score banner
        if percentage >= 80:
            st.success(f"Excellent! You scored {score}/{total} ({percentage}%)")
        elif percentage >= 60:
            st.warning(f"Good effort! You scored {score}/{total} ({percentage}%)")
        else:
            st.error(f"Needs revision! You scored {score}/{total} ({percentage}%)")

        # Progress bar
        st.progress(percentage / 100)

        st.divider()

        # Show answers with explanations
        st.subheader("Review your answers")
        for i, q in enumerate(st.session_state.quiz_questions):
            your_ans = st.session_state.quiz_answers.get(i, "")
            correct_ans = q["answer"]
            is_correct = your_ans == correct_ans

            if is_correct:
                st.success(f"Q{i+1}: {q['question']}")
                st.write(f"Your answer: {your_ans}) {q['options'].get(your_ans, '')} ✅")
            else:
                st.error(f"Q{i+1}: {q['question']}")
                st.write(f"Your answer: {your_ans}) {q['options'].get(your_ans, '')} ❌")
                st.write(f"Correct answer: {correct_ans}) {q['options'].get(correct_ans, '')} ✅")

            if q["explanation"]:
                st.caption(f"Explanation: {q['explanation']}")
            st.divider()

        # Retry button
        if st.button("Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.rerun()

# MULTI-AGENT MODE
elif mode == "🤖 Multi-Agent":
    st.title("🤖 Multi-Agent Research Pipeline")
    st.caption("3 AI agents work together — Researcher finds, Critic challenges, Writer composes")

    # How it works explanation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔍 **Researcher**\nSearches web, collects and organizes facts")
    with col2:
        st.warning("🔎 **Critic**\nChallenges findings, identifies gaps and bias")
    with col3:
        st.success("✍️ **Writer**\nSynthesizes everything into a final answer")

    st.divider()

    question = st.text_input(
        "Enter your research question",
        placeholder="e.g. What is the future of electric vehicles in India?"
    )

    if st.button("Run Multi-Agent Pipeline", type="primary"):
        if question.strip():
            st.divider()
            thoughts = run_multi_agent(question)
            st.divider()

            # Show each agent's thinking
            for agent_name, output in thoughts:
                if agent_name == "Researcher":
                    with st.expander(f"🔍 Researcher Agent findings", expanded=False):
                        st.write(output)
                elif agent_name == "Critic":
                    with st.expander(f"🔎 Critic Agent review", expanded=False):
                        st.write(output)
                elif agent_name == "Writer":
                    st.subheader("Final Answer")
                    st.write(output)

                    # Save to memory
                    save_to_memory(question, output)

        else:
            st.warning("Please enter a question!")