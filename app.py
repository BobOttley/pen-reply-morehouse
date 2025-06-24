import os, json, pickle, re, numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# ──────────────────────────────
# ✅  SET-UP
# ──────────────────────────────
load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")

EMBED_MODEL               = "text-embedding-3-small"
SIMILARITY_THRESHOLD      = 0.30
RESPONSE_LIMIT            = 3
STANDARD_MATCH_THRESHOLD  = 0.85

# ──────────────────────────────
# 🔒  PII REDACTION
# ──────────────────────────────
PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",       # emails
    r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b",                      # US-style phone
    r"\+?\d[\d\s\-]{7,}\d",                                      # int’l phone
    r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b",                    # UK postcode
]
def remove_personal_info(text:str)->str:
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[redacted]", text, flags=re.I)
    # crude “My name is …” removal
    # Redact full names after "My name is"
    text = re.sub(r"\bmy name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", "my name is [redacted]", text, flags=re.I)

    # Redact full names in greetings like "Dear Clare Bennett" → "Dear Parent"
    text = re.sub(r"\bDear\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", "Dear Parent", text)

    # Optionally redact any capitalised name patterns elsewhere (more aggressive)
    text = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[redacted name]", text)

    return text

# ──────────────────────────────
# 📦  HELPERS
# ──────────────────────────────
def embed_text(text: str) -> np.ndarray:
    text = text.replace("\n", " ")
    res  = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(res.data[0].embedding)

def markdown_to_html(text: str) -> str:
    """Convert markdown links to clickable HTML (keeps anchor text)."""
    return re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        lambda m: f'<a href="{m.group(2)}" target="_blank">{m.group(1)}</a>',
        text
    )

# ──────────────────────────────
# 📚  LOAD SCHOOL KB
# ──────────────────────────────
with open("data/metadata.pkl", "rb") as f:
    kb = pickle.load(f)
    doc_embeddings = np.array(kb["embeddings"])
    metadata       = kb["metadata"]
print(f"✅ Loaded {len(metadata)} knowledge chunks.")

# ──────────────────────────────
# 📁  LOAD SAVED STANDARD RESPONSES
# ──────────────────────────────
standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path="standard_responses.json"
    if not os.path.exists(path):
        print("⚠️ No standard_responses.json found.")
        return
    try:
        with open(path,"r") as f:
            saved=json.load(f)
        for entry in saved:
            msg = remove_personal_info(entry["message"])
            rep = entry["reply"]                   # reply already HTML-ised
            standard_messages.append(msg)
            standard_embeddings.append(embed_text(msg))
            standard_replies.append(rep)
        print(f"✅ Loaded {len(standard_messages)} template replies.")
    except Exception as e:
        print(f"❌ Failed loading templates: {e}")

_load_standard_library()

def check_standard_match(q_vec: np.ndarray) -> str:
    if not standard_embeddings: return ""
    sims = [1 - cosine(q_vec, emb) for emb in standard_embeddings]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= STANDARD_MATCH_THRESHOLD:
        print(f"🔁 Using template (similarity {sims[best_idx]:.2f})")
        return standard_replies[best_idx]
    return ""

# ──────────────────────────────
# 📨  POST /reply
# ──────────────────────────────
@app.route("/reply", methods=["POST"])
def reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()

        # 🔒 sanitise
        question    = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)

        if not question:
            return jsonify({"error":"No message received."}), 400

        q_vec = embed_text(question)

        # 1) pre-approved template?
        matched = check_standard_match(q_vec)
        if matched:
            return jsonify({
                "reply": matched,
                "sentiment_score": 10,
                "strategy_explanation": "Used approved template."
            })

        # 2) sentiment (mini model, cheap)
        sent_prompt = f"""
        You are a sentiment analysis assistant for a UK prep school.

        Your task is to return a JSON object with:
        - a sentiment score (1 to 10), where 10 is highly positive and 1 is negative
        - a brief admissions response strategy

        Always use British spelling. The output must be clean JSON with no backticks, no code blocks, no markdown.

        Format strictly:
        {{
          "score": 1-10,
          "strategy": "Short strategic recommendation (max 35 words)"
        }}

        Enquiry:
        \"\"\"{question}\"\"\"
        """.strip()

        # Run completion
        sent_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": sent_prompt}],
            temperature=0.2
        ).choices[0].message.content.strip()

        # Clean and enforce parsing
        try:
            sent_response = re.sub(r"^```(?:json)?\n?", "", sent_response).rstrip("`").strip()
            sent = json.loads(sent_response)
            score = int(sent.get("score", 5))
            strat = sent.get("strategy", "").strip()
            if not strat:
                raise ValueError("Missing strategy")
        except Exception as e:
            print(f"❌ Sentiment parse failed: {e}")
            score = 5
            strat = "Handle personally — sentiment and tone unclear."

        # 3) KB retrieval
        sims = [(1 - cosine(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = [m for m in sims if m[0] >= SIMILARITY_THRESHOLD]
        top = sorted(top, key=lambda x:x[0], reverse=True)[:RESPONSE_LIMIT]

        if not top:
            return jsonify({
                "reply":"<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score":score,"strategy_explanation":strat
            })

        context_blocks = [f"{m['text']}\n[Info source]({m.get('url','')})" if m.get('url') else m['text']
                          for _,m in top]
        top_context = "\n---\n".join(context_blocks)

        # 4) main reply prompt
        prompt = f"""
You are Mrs Powell De Caires , Director of Admissions and Marketing at More House School , a UK all girls school with a pre senior years 5 and 6, and a senior section from year 7 through to Sixth Form. The school is based in Knightsbridge London.

You must always use British spelling (e.g. prioritise, organise, programme). Never use American spellings such as prioritize, organize, inquire or center.
This is strictly enforced in all emails and replies.


Sentiment score: {score}/10
Strategy: {strat}
Additional instruction: "{instruction}"

Write a warm, professional email reply using only the school info provided. Follow these strict formatting and tone rules:

- Use British spelling at all times (e.g. organise, programme, enrolment)
- Never say "click here" under any circumstances
- Do not bold or italicise any text using asterisks or markdown (e.g. **Super Curriculum** → just write “Super Curriculum”)
- Do not CAPITALISE programmes like “Gifted & Talented” or “Super Curriculum” — write them naturally
- Embed links using Markdown format, with natural, meaningful anchor text
- Do not show raw URLs
- Never list links at the bottom
- Weave all links naturally into the body of the email like a professional school reply

Never use anchor phrases like “click here”, “learn more”, or “register here”. 
Instead, write anchor text that clearly describes the link’s destination.

❌ Never use markdown bold (e.g. **Nursery**)  
❌ Do not return bullet-point lists for fees or other structured content  
✅ Fee information should be summarised in warm, natural sentences suitable for email

❌ Never mention specific event dates such as Open Mornings unless they are current.  
✅ Instead, invite the parent to visit our [Open Events page](https://www.morehouse.org.uk/admissions/our-open-events/) for up-to-date details.

Never repeat the parent’s full name. If a name is present, redact the surname or address them generically.

Parent Email:
\"\"\"{question}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"

Sign off:
Mrs Powell De Caires  
Director of Admissions and Marketing  
More House School
""".strip()

        reply_md = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        reply_html = markdown_to_html(reply_md)
        reply_html = reply_html.replace("<strong>", "").replace("</strong>", "")

# 🧼 Replace common American spellings with British ones (safety net)
        reply_html = reply_html.replace("prioritize", "prioritise") \
                       .replace("organize", "organise") \
                       .replace("personalize", "personalise") \
                       .replace("center", "centre") \
                       .replace("analyze", "analyse") \
                       .replace("program", "programme")

# Strip leftover markdown bold symbols if any
        reply_html = reply_html.replace("**", "")

# 🔗 Enforce correct links using keyword-based mapping
        from url_mapping import URL_MAP
        for phrase, correct_url in URL_MAP.items():
            pattern = rf'<a href="[^"]+">([^<]*{phrase}[^<]*)</a>'
            replacement = rf'<a href="{correct_url}">\1</a>'
            reply_html = re.sub(pattern, replacement, reply_html, flags=re.IGNORECASE)

        reply_html = markdown_to_html(reply_md)

        return jsonify({
            "reply": reply_html,
            "sentiment_score": score,
            "strategy_explanation": strat
        })

    except Exception as e:
        print(f"❌ REPLY ERROR: {e}")
        return jsonify({"error":"Internal server error."}), 500

# ──────────────────────────────
# ✏️  POST /revise
# ──────────────────────────────
@app.route("/revise", methods=["POST"])
def revise():
    try:
        body = request.get_json(force=True)
        message_raw  = (body.get("message") or "").strip()
        prev_reply   = (body.get("previous_reply") or "").strip()  # already HTML
        instruction_raw = (body.get("instruction") or "").strip()

        if not (message_raw and prev_reply and instruction_raw):
            return jsonify({"error":"Missing fields."}), 400

        message    = remove_personal_info(message_raw)
        instruction= remove_personal_info(instruction_raw)

        prompt = f"""
Revise the admissions reply below according to the instruction.

Instruction: {instruction}

Parent enquiry:
\"\"\"{message}\"\"\"

Current reply (Markdown):
\"\"\"{prev_reply}\"\"\"

Return only the revised reply without code blocks or formatting fences. Do not use backticks.

""".strip()

        new_md = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()

        # 🧼 Remove accidental markdown fences like ```markdown
        import re
        new_md = re.sub(r"^```(?:markdown)?\n?", "", new_md).rstrip("`").strip()

        return jsonify({"reply": markdown_to_html(new_md)})


    except Exception as e:
        print(f"❌ REVISION ERROR: {e}")
        return jsonify({"error":"Revision failed."}), 500

# ──────────────────────────────
# 💾  POST /save-standard
# ──────────────────────────────
@app.route("/save-standard", methods=["POST"])
def save_standard():
    try:
        body = request.get_json(force=True)
        msg_raw = (body.get("message") or "").strip()
        reply   = (body.get("reply")   or "").strip()

        if not (msg_raw and reply):
            return jsonify({"status":"error","message":"Missing fields"}), 400

        msg_redacted = remove_personal_info(msg_raw)

        # append & persist
        record = {"timestamp":datetime.now().isoformat(),"message":msg_redacted,"reply":reply}
        path="standard_responses.json"
        data=[]
        if os.path.exists(path):
            with open(path,"r") as f: data=json.load(f)
        data.append(record)
        with open(path,"w") as f: json.dump(data,f,indent=2)

        # in-memory
        standard_messages.append(msg_redacted)
        standard_embeddings.append(embed_text(msg_redacted))
        standard_replies.append(reply)

        return jsonify({"status":"ok"})
    except Exception as e:
        print(f"❌ SAVE ERROR: {e}")
        return jsonify({"status":"error","message":"Save failed"}),500

# ──────────────────────────────
# 🌐  SERVE FRONT END
# ──────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ──────────────────────────────
# ▶️  MAIN
# ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
