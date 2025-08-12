#!/usr/bin/env python3
"""
OpenSAT Coach — Unified App (PDF analyzer + domain detector + quiz)

What this script does
---------------------
• Upload a SAT score report PDF (College Board Practice Score Report works best).
  - Extracts section scores (heuristic text approach) and/or
  - Uses computer vision to detect empty Knowledge & Skills boxes by domain
    (Information & Ideas, Craft & Structure, etc.) using OpenCV + Tesseract.
• Computes focus domains from whichever signal is available (vision or text).
• Loads a question bank (questions.csv) and serves a targeted quiz in the browser.
• Clean, minimal UI (Bootstrap, Alpine.js, Chart.js + MathJax for math rendering).

Quick start
-----------
1) Save this file as `opensat_coach_full.py`
2) (Recommended) Create a virtualenv
   python3 -m venv .venv && source .venv/bin/activate
3) Install dependencies:
   pip install flask PyPDF2 pandas opencv-python-headless pytesseract PyMuPDF
   
   # If you want to run locally with desktop OpenCV you can also use `opencv-python`.
   # Ensure Tesseract is installed and on PATH. On Windows, the script will
   # auto-detect at C:\\Program Files\\Tesseract-OCR\\tesseract.exe if present.

4) (Optional) Put a `questions.csv` in the project root (the app can also accept one via UI).
5) Run the app:
   flask --app opensat_coach_full.py --debug run
6) Open http://127.0.0.1:5000

Notes
-----
• Vision detection params can be overridden via `detect_params.json` in the working directory.
• If CV/ocr detection fails (e.g., missing Tesseract), we fall back to text-only heuristics.
• This script intentionally avoids remote LLM calls; when an answer is wrong, it shows an
  "Official Explanation" field from your CSV if present.

CSV format (flexible headers)
-----------------------------
- id
- domain (e.g., Algebra, Advanced Math, Expression of Ideas, etc.)
- paragraph / passage (optional)
- prompt / question
- choice_A / choice_B / choice_C / choice_D (or A/B/C/D)
- correct_answer_letter / correct_answer_text (either works)
- explanation (optional)

"""
from __future__ import annotations

import os
import re
import io
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

# --------- Web app deps ---------
from flask import (
    Flask, request, redirect, url_for, session, flash,
    render_template, jsonify
)
from jinja2 import DictLoader
import pandas as pd
from PyPDF2 import PdfReader

# --------- Vision deps ---------
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pytesseract import Output

# ---------------------------
# Config
# ---------------------------
SECRET = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
DEFAULT_QUESTIONS_CSV = os.getenv("QUESTIONS_CSV", "")  # if set, preload
MAX_QUIZ_QUESTIONS = int(os.getenv("MAX_QUIZ_QUESTIONS", "15"))
PARAMS_FILE = "detect_params.json"
EXPECTED_PER_DOMAIN = 14  # expected bar boxes per domain

# Optional: auto-path Tesseract on Windows
if os.name == "nt":
    t_path = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if os.path.exists(t_path):
        pytesseract.pytesseract.tesseract_cmd = t_path

app = Flask(__name__)
app.secret_key = SECRET
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# UI Templates
# ---------------------------
TEMPLATES: Dict[str, str] = {}

TEMPLATES["base.html"] = r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>OpenSAT Coach</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- MathJax for equations -->
    <script>
      MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\(', '\\)']],
          displayMath: [['$$', '$$'], ['\\[', '\\]']],
          processEscapes: true,
          packages: {'[+]': ['noerrors']}
        },
        loader: { load: ['[tex]/noerrors'] }
      };
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
      :root { --background:#fff; --foreground:#0a0a0a; --muted:#fafafa; --muted-foreground:#525252; --card:#fff; --border:#e5e5e5; --brand:#111; --brand-dark:#000; --ring:rgba(0,0,0,.08); --radius:.5rem; }
      body{background:var(--muted);color:var(--foreground);font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
      .navbar{background:var(--background);border-bottom:1px solid var(--border)}
      .brand{font-weight:700;font-size:1.25rem;letter-spacing:-.025em}
      .card{border:1px solid var(--border);border-radius:12px;background:var(--card);box-shadow:0 1px 0 rgba(0,0,0,.04)}
      .hero{background:var(--card);border:1px solid var(--border);border-radius:16px}
      .btn{border-radius:var(--radius)}
      .btn-brand{background:var(--brand);color:#fff;border:1px solid #111;font-weight:600}
      .btn-brand:hover{background:var(--brand-dark);border-color:var(--brand-dark)}
      .btn-ghost,.btn-outline-secondary,.btn-light{background:transparent;color:var(--foreground);border:1px solid var(--border)}
      .btn-ghost:hover,.btn-outline-secondary:hover,.btn-light:hover{background:#f5f5f5;border-color:#d4d4d4}
      .badge-soft{background:#f4f4f5;color:var(--foreground);border:1px solid var(--border);font-weight:500}
      .domain-chip{border-radius:9999px;padding:.3rem .7rem;background:#f4f4f5;color:var(--foreground);border:1px solid var(--border);font-weight:500;font-size:.85rem}
      .progress{height:8px;border-radius:4px;background:#e5e5e5}.progress-bar{border-radius:4px;background:#0a0a0a}
      .question-content{background:#fff;border-radius:12px;padding:1.25rem;border:1px solid #e2e8f0}
      .choice-option{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:.75rem;transition:all .2s}
      .choice-option:hover{background:#fafafa;border-color:#111}
      .choice-option.selected{background:#f5f5f5;border-color:#111}
      .explanation-box{background:#fff;border:1px solid var(--border);border-radius:12px;padding:1.25rem}
    </style>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg">
      <div class="container py-2">
        <a class="navbar-brand brand" href="{{ url_for('home') }}"><i class="bi bi-mortarboard"></i> OpenSAT Coach</a>
        <div class="ms-auto d-flex gap-2">
          <a href="{{ url_for('home') }}" class="btn btn-sm btn-outline-secondary">Home</a>
        </div>
      </div>
    </nav>
    <main class="container my-4">
      {% with messages = get_flashed_messages() %}
        {% if messages %}<div class="alert alert-info">{{ messages[0] }}</div>{% endif %}
      {% endwith %}
      {% block content %}{% endblock %}
    </main>
    <footer class="container pb-5 small text-secondary"><hr>Official explanations are shown automatically for incorrect answers.</footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', function(){
        document.querySelectorAll('.choice-option').forEach(opt=>{
          opt.addEventListener('click', function(){
            const radio = this.querySelector('input[type="radio"]');
            if(radio){ radio.checked = true; document.querySelectorAll('.choice-option').forEach(x=>x.classList.remove('selected')); this.classList.add('selected'); }
          });
        });
        if(window.MathJax){ MathJax.typesetPromise().catch(()=>{}); }
      });
    </script>
  </body>
</html>
"""

TEMPLATES["home.html"] = r"""
{% extends 'base.html' %}
{% block content %}
  <div class="p-4 p-lg-5 rounded-4 hero mb-4">
    <div class="row align-items-center">
      <div class="col-lg-7">
        <h1 class="display-6 fw-bold mb-3">Know exactly what to practice next.</h1>
        <p class="lead text-secondary">Upload your SAT score report PDF. We'll analyze your performance, detect weak domains visually, and serve the right questions.</p>
        <div class="d-flex gap-2">
          <a href="#upload" class="btn btn-brand btn-lg"><i class="bi bi-upload"></i> Upload PDF</a>
          <a href="{{ url_for('recommend') }}" class="btn btn-outline-secondary btn-lg"><i class="bi bi-graph-up"></i> See recommendations</a>
        </div>
      </div>
      <div class="col-lg-5 mt-4 mt-lg-0">
        <div class="card p-3">
          <div class="card-body">
            <h5 class="fw-bold mb-3">Current Diagnosis</h5>
            {% if analysis %}
              <div class="row g-3">
                <div class="col-12 col-md-6">
                  <div class="border rounded-3 p-3">
                    <div class="d-flex justify-content-between mb-1"><strong>Reading & Writing</strong><span>{{ analysis.rw_score or '—' }}</span></div>
                    <div class="progress"><div class="progress-bar" role="progressbar" style="width: {{ analysis.rw_pct }}%"></div></div>
                    <div class="mt-2 small text-secondary">Incorrect: {{ analysis.rw_incorrect }}</div>
                  </div>
                </div>
                <div class="col-12 col-md-6">
                  <div class="border rounded-3 p-3">
                    <div class="d-flex justify-content-between mb-1"><strong>Math</strong><span>{{ analysis.math_score or '—' }}</span></div>
                    <div class="progress"><div class="progress-bar" role="progressbar" style="width: {{ analysis.math_pct }}%"></div></div>
                    <div class="mt-2 small text-secondary">Incorrect: {{ analysis.math_incorrect }}</div>
                  </div>
                </div>
              </div>
              <div class="mt-3">
                <strong>Focus areas:</strong>
                {% for d in analysis.focus_domains %}
                  <span class="domain-chip me-2">{{ d }}</span>
                {% endfor %}
                <div class="mt-3"><a href="{{ url_for('recommend') }}" class="btn btn-brand">See recommendations</a></div>
              </div>
            {% else %}
              <p class="text-secondary">No analysis yet. Upload your PDF to get started.</p>
            {% endif %}
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="row" id="upload">
    <div class="col-lg-6">
      <div class="card p-4 h-100">
        <h5 class="fw-bold mb-2"><i class="bi bi-filetype-pdf me-2"></i>Upload SAT PDF</h5>
        <p class="form-help text-secondary">Best results with College Board Practice Score Reports (PDF). If parsing fails, enter scores manually.</p>
        <form class="mt-2" action="{{ url_for('upload_pdf') }}" method="post" enctype="multipart/form-data">
          <div class="mb-3"><input class="form-control" type="file" name="pdf" accept="application/pdf" required></div>
          <button class="btn btn-brand" type="submit">Analyze PDF</button>
        </form>
      </div>
    </div>
    <div class="col-lg-6 mt-4 mt-lg-0">
      <div class="card p-4 h-100">
        <h5 class="fw-bold mb-2"><i class="bi bi-pencil-square me-2"></i>Manual scores (fallback)</h5>
        <form action="{{ url_for('manual_scores') }}" method="post" class="row g-3">
          <div class="col-6"><label class="form-label">R&W Score</label><input type="number" name="rw_score" min="200" max="800" step="10" class="form-control" placeholder="740"></div>
          <div class="col-6"><label class="form-label">Math Score</label><input type="number" name="math_score" min="200" max="800" step="10" class="form-control" placeholder="780"></div>
          <div class="col-6"><label class="form-label">R&W Incorrect</label><input type="number" name="rw_incorrect" min="0" max="54" class="form-control" placeholder="e.g., 6"></div>
          <div class="col-6"><label class="form-label">Math Incorrect</label><input type="number" name="math_incorrect" min="0" max="44" class="form-control" placeholder="e.g., 8"></div>
          <div class="col-12"><button class="btn btn-outline-secondary" type="submit">Save diagnosis</button></div>
        </form>
      </div>
    </div>
  </div>
{% endblock %}
"""

TEMPLATES["recommend.html"] = r"""
{% extends 'base.html' %}
{% block content %}
  <div class="row g-4">
    <div class="col-lg-7">
      <div class="card p-4">
        <div class="d-flex align-items-center justify-content-between mb-2">
          <h4 class="fw-bold mb-0">Your recommended focus</h4>
          <a class="btn btn-sm btn-outline-secondary" href="{{ url_for('home') }}">Change PDF</a>
        </div>
        {% if analysis %}
          <canvas id="sectionChart" class="mb-3" height="120"></canvas>
          <div>
            <div class="mb-2"><strong>Primary domains:</strong></div>
            {% for d in analysis.focus_domains %}
              <span class="domain-chip me-2">{{ d }}</span>
            {% endfor %}
          </div>
        {% else %}
          <p class="text-secondary">Upload a PDF or enter scores to generate recommendations.</p>
        {% endif %}
      </div>

      <div class="card p-4 mt-4">
        <div class="d-flex align-items-center justify-content-between">
          <h5 class="fw-bold mb-0">Start a targeted quiz</h5>
          <form action="{{ url_for('start_quiz') }}" method="post" class="d-flex align-items-center gap-2">
            <input type="hidden" name="domains" value="{{ ','.join(analysis.focus_domains) if analysis else '' }}">
            <input type="number" name="n" class="form-control form-control-sm" min="3" max="50" value="{{ suggested_n }}" style="width: 100px;">
            <button class="btn btn-brand btn-sm" type="submit"><i class="bi bi-play-fill"></i> Start</button>
          </form>
        </div>
        <p class="text-secondary small mt-2">We’ll pull from your built-in question bank (<code>questions.csv</code>).</p>
      </div>
    </div>

    <div class="col-lg-5">
      <div class="card p-4">
        <h5 class="fw-bold">Question bank status</h5>
        <p class="mb-1">Loaded questions: <span class="badge badge-soft">{{ qstats.total }}</span></p>
        <div class="small text-secondary">By domain:</div>
        <ul class="small mt-2">
          {% for dom, cnt in qstats.by_domain %}
            <li>{{ dom }} — {{ cnt }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>
  </div>

  {% if analysis %}
  <script>
    const ctx = document.getElementById('sectionChart');
    if (ctx) {
      new Chart(ctx, {
        type: 'bar',
        data: { labels: ['Reading & Writing', 'Math'], datasets: [{ label: 'Incorrect', data: [{{ analysis.rw_incorrect or 0 }}, {{ analysis.math_incorrect or 0 }}] }] },
        options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
      });
    }
  </script>
  {% endif %}
{% endblock %}
"""

TEMPLATES["questions.html"] = r"""
{% extends 'base.html' %}
{% block content %}
  <div class="row g-4">
    <div class="col-lg-7">
      <div class="card p-4">
        <h4 class="fw-bold">Manage question bank</h4>
        <p class="text-secondary small">Upload a CSV of questions (OpenSAT format compatible). Existing questions are replaced.</p>
        <form action="{{ url_for('upload_questions') }}" method="post" enctype="multipart/form-data" class="d-flex gap-2">
          <input type="file" class="form-control" name="csv" accept=".csv" required>
          <button class="btn btn-brand" type="submit">Upload CSV</button>
        </form>
        {% if qstats.total %}
        <div class="mt-4">
          <div class="d-flex gap-2 align-items-center">
            <span class="badge badge-soft">Total: {{ qstats.total }}</span>
            <span class="small text-secondary">Domains: {{ qstats.distinct_domains }}</span>
          </div>
          <ul class="small mt-2">
            {% for dom, cnt in qstats.by_domain %}
              <li>{{ dom }} — {{ cnt }}</li>
            {% endfor %}
          </ul>
        </div>
        {% endif %}
      </div>
    </div>
    <div class="col-lg-5">
      <div class="card p-4">
        <h5 class="fw-bold">Tips</h5>
        <ul class="small text-secondary mb-0">
          <li>CSV columns are flexible — we map common headings automatically.</li>
          <li>Include a <em>domain</em> for better targeting (e.g., Advanced Math, Algebra, Craft and Structure).</li>
          <li>If available, include an <em>explanation</em> column. Wrong answers will show it automatically.</li>
        </ul>
      </div>
    </div>
  </div>
{% endblock %}
"""

TEMPLATES["quiz.html"] = r"""
{% extends 'base.html' %}
{% block content %}
  <div class="d-flex align-items-center justify-content-between mb-4">
    <h4 class="fw-bold mb-0"><i class="bi bi-puzzle me-2"></i>Targeted Practice</h4>
    <a class="btn btn-outline-secondary btn-sm" href="{{ url_for('recommend') }}"><i class="bi bi-arrow-left me-1"></i>Back to recommendations</a>
  </div>
  {% if not quiz %}
    <div class="alert alert-warning"><i class="bi bi-exclamation-triangle me-2"></i>No quiz loaded. Start one from the recommendations page.</div>
  {% else %}
    <div class="card"><div class="card-body p-4">
      <div class="d-flex justify-content-between align-items-center mb-4">
        <div class="d-flex align-items-center gap-3"><span class="badge bg-primary fs-6">Question {{ idx+1 }} of {{ total }}</span><span class="badge badge-soft">{{ q.domain or 'General' }}</span></div>
        <div class="progress" style="width:200px"><div class="progress-bar" role="progressbar" style="width: {{ ((idx+1)/total*100)|round }}%"></div></div>
      </div>
      {% if q.paragraph %}
      <div class="question-content mb-4"><h6 class="text-muted mb-2">Reading Passage:</h6><div class="fs-6">{{ q.paragraph|safe }}</div></div>
      {% endif %}
      <div class="question-content mb-4"><h5 class="mb-3">{{ q.prompt|safe }}</h5></div>
      <form action="{{ url_for('submit_answer', qidx=idx) }}" method="post" id="quiz-form">
        <div class="row g-2">
          {% for letter, text in q.choices.items() %}
          <div class="col-12"><div class="choice-option {% if user_answer == letter %}selected{% endif %}">
            <div class="form-check">
              <input class="form-check-input" type="radio" name="answer" id="opt{{ letter }}" value="{{ letter }}" {% if user_answer == letter %}checked{% endif %} required>
              <label class="form-check-label w-100" for="opt{{ letter }}"><strong class="me-2">{{ letter }}.</strong>{{ text|safe }}</label>
            </div>
          </div></div>
          {% endfor %}
        </div>
        <div class="d-flex gap-2 mt-4"><button class="btn btn-brand btn-lg" type="submit"><i class="bi bi-check2-circle me-1"></i>Submit Answer</button></div>
      </form>

      {% if result %}
        <hr class="my-4">
        {% if result == 'correct' %}
          <div class="alert alert-success d-flex align-items-center"><i class="bi bi-check2-circle fs-4 me-3"></i><div><strong>Excellent!</strong> You selected <strong>{{ user_answer }}</strong>, which is correct.</div></div>
          <div class="d-flex gap-2"><a href="{{ url_for('next_question', qidx=idx) }}\" class=\"btn btn-success btn-lg\"><i class=\"bi bi-arrow-right me-1\"></i>Next Question</a></div>
        {% else %}
          <div class="alert alert-danger d-flex align-items-center"><i class="bi bi-x-circle fs-4 me-3"></i><div><strong>Not quite right.</strong> You selected <strong>{{ user_answer }}</strong>, but the correct answer is <strong>{{ q.answer_letter }}</strong>.</div></div>
          {% if q.explanation %}
            <div class="explanation-box mt-3"><h6 class="fw-bold mb-3"><i class="bi bi-lightbulb me-2"></i>Official Explanation</h6><div class="fs-6">{{ q.explanation|safe }}</div></div>
          {% endif %}
          <div class="d-flex gap-2 mt-4"><a href="{{ url_for('next_question', qidx=idx) }}\" class=\"btn btn-primary btn-lg\"><i class=\"bi bi-arrow-right me-1\"></i>Continue</a></div>
        {% endif %}
      {% endif %}
    </div></div>
  {% endif %}
{% endblock %}
"""

app.jinja_loader = DictLoader(TEMPLATES)

# ---------------------------
# Domains & helpers
# ---------------------------
DOMAINS_RW = [
    "Information and Ideas",
    "Expression of Ideas",
    "Craft and Structure",
    "Standard English Conventions",
]
DOMAINS_MATH = [
    "Algebra",
    "Advanced Math",
    "Problem-Solving and Data Analysis",
    "Geometry and Trigonometry",
]
ALL_DOMAINS = DOMAINS_RW + DOMAINS_MATH

@dataclass
class Question:
    id: str
    domain: Optional[str]
    paragraph: Optional[str]
    prompt: str
    choices: Dict[str, str]
    answer_letter: Optional[str]
    answer_text: Optional[str]
    explanation: Optional[str]
    extra: Dict[str, Any]

class QuestionBank:
    def __init__(self):
        self.questions: List[Question] = []

    def load_csv(self, file_like) -> None:
        df = pd.read_csv(file_like)
        cols = {c.lower().strip(): c for c in df.columns}
        def first(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None
        id_col = first('id')
        domain_col = first('domain')
        paragraph_col = first('paragraph','passage')
        prompt_col = first('prompt','question')
        choices_cols = {'A': first('choice_a','a'), 'B': first('choice_b','b'), 'C': first('choice_c','c'), 'D': first('choice_d','d')}
        ans_letter_col = first('correct_answer_letter','correct_answer','answer_letter')
        ans_text_col = first('correct_answer_text','answer_text')
        expl_col = first('explanation','rationale')
        questions: List[Question] = []
        for _, row in df.iterrows():
            qid = str(row[id_col]) if id_col and not pd.isna(row.get(id_col)) else str(len(questions)+1)
            domain = str(row[domain_col]).strip() if domain_col and not pd.isna(row.get(domain_col)) else None
            paragraph = str(row[paragraph_col]).strip() if paragraph_col and not pd.isna(row.get(paragraph_col)) else None
            prompt = str(row[prompt_col]).strip() if prompt_col and not pd.isna(row.get(prompt_col)) else ""
            choices = {}
            for L, c in choices_cols.items():
                if c and not pd.isna(row.get(c)):
                    choices[L] = str(row[c])
            ans_letter = None
            if ans_letter_col and not pd.isna(row.get(ans_letter_col)):
                ans_letter = str(row[ans_letter_col]).strip().upper()[:1]
            ans_text = None
            if ans_text_col and not pd.isna(row.get(ans_text_col)):
                ans_text = str(row[ans_text_col]).strip()
            elif ans_letter and ans_letter in choices:
                ans_text = choices.get(ans_letter)
            explanation = str(row[expl_col]).strip() if expl_col and not pd.isna(row.get(expl_col)) else None
            extra = {c: row[c] for c in df.columns if c not in {id_col, domain_col, paragraph_col, prompt_col, *[x for x in choices_cols.values() if x], ans_letter_col, ans_text_col, expl_col}}
            questions.append(Question(qid, domain, paragraph, prompt, choices, ans_letter, ans_text, explanation, extra))
        self.questions = questions

    def pick(self, domains: List[str], n: int) -> List[Question]:
        pool = [q for q in self.questions if (not domains or (q.domain and q.domain in domains))]
        if not pool:
            pool = self.questions[:]
        random.shuffle(pool)
        return pool[:n]

    def stats(self) -> Dict[str, Any]:
        total = len(self.questions)
        by_domain: Dict[str, int] = {}
        for q in self.questions:
            d = q.domain or "(none)"
            by_domain[d] = by_domain.get(d, 0) + 1
        return {
            'total': total,
            'by_domain': sorted(by_domain.items(), key=lambda x: (-x[1], x[0])),
            'distinct_domains': len(by_domain),
        }

QB = QuestionBank()
if DEFAULT_QUESTIONS_CSV and os.path.exists(DEFAULT_QUESTIONS_CSV):
    with open(DEFAULT_QUESTIONS_CSV, 'r', encoding='utf-8') as f:
        QB.load_csv(f)

# ---------------------------
# PDF text parsing & diagnosis (heuristic fallback)
# ---------------------------
SCORE_REPORT_MARKERS = [r"TOTAL\s+SCORE", r"Reading\s+and\s+Writing", r"Math", r"Questions\s+Overview"]

@dataclass
class Diagnosis:
    total_score: Optional[int]
    rw_score: Optional[int]
    math_score: Optional[int]
    rw_incorrect: Optional[int]
    math_incorrect: Optional[int]
    rw_pct: int
    math_pct: int
    focus_domains: List[str]


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)


def looks_like_score_report(text: str) -> bool:
    return sum(1 for m in SCORE_REPORT_MARKERS if re.search(m, text, re.I)) >= 2


def parse_score_report(text: str) -> Diagnosis:
    def pick_int(patterns: List[str]) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
        return None
    total = pick_int([r"TOTAL\s*SCORE\D+(\d{3,4})"])
    rw = pick_int([r"Reading\s*and\s*Writing\D+(\d{3})", r"Reading\s*&\s*Writing\D+(\d{3})"])
    math = pick_int([r"Math\D+(\d{3})"])
    rw_incorrect = pick_int([r"Reading\s*and\s*Writing[\s\S]*?Incorrect\s*Answers:\s*(\d+)", r"Reading\s*and\s*Writing[\s\S]*?Incorrect:\s*(\d+)"])
    math_incorrect = pick_int([r"Math[\s\S]*?Incorrect\s*Answers:\s*(\d+)", r"Math[\s\S]*?Incorrect:\s*(\d+)"])
    rw_pct = int(round(((rw or 0) - 200) / 6)) if rw else 0
    math_pct = int(round(((math or 0) - 200) / 6)) if math else 0
    # Basic focus guess if nothing else is available
    rw_err = rw_incorrect or 0
    math_err = math_incorrect or 0
    focus = ["Craft and Structure", "Standard English Conventions"] if rw_err >= math_err else ["Advanced Math", "Algebra"]
    return Diagnosis(total, rw, math, rw_incorrect, math_incorrect, rw_pct, math_pct, focus)

# ---------------------------
# Vision-based domain detector (from detect_empty_categories.py)
# ---------------------------

def load_params() -> dict:
    defaults = {
        "threshold": "100",  # or "otsu"
        "morph_size": 3,
        "erosion": 2,
        # crop percents for the Knowledge & Skills block
        "crop_x1": 0.45,
        "crop_x2": 1.00,
        "crop_y1": 0.25,
        "crop_y2": 0.80,
    }
    if not os.path.exists(PARAMS_FILE):
        return defaults
    try:
        with open(PARAMS_FILE) as f:
            data = json.load(f)
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return defaults


def pdf_pages_to_images(pdf_path: str, dpi: int = 350) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return pages


def crop_region(img: np.ndarray, cx1: float, cx2: float, cy1: float, cy2: float) -> np.ndarray:
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    if cx1 > cx2: cx1, cx2 = cx2, cx1
    if cy1 > cy2: cy1, cy2 = cy2, cy1
    x1 = max(0, min(w - 1, int(round(w * cx1))))
    x2 = max(0, min(w,     int(round(w * cx2))))
    y1 = max(0, min(h - 1, int(round(h * cy1))))
    y2 = max(0, min(h,     int(round(h * cy2))))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,1,3), dtype=np.uint8)
    return img[y1:y2, x1:x2]


def half_split(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if img.size == 0 or img.shape[1] == 0:
        return img, img
    h, w = img.shape[:2]
    mid = w // 2
    return img[:, :mid], img[:, mid:]


def _detect_boxes(img: np.ndarray, threshold: str, morph_size: int, erosion: int) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray]:
    if img.size == 0 or img.shape[1] == 0 or img.shape[0] == 0:
        return [], np.zeros((1,1,3), dtype=np.uint8)
    target_w = 900
    scale = target_w / float(img.shape[1])
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    if str(threshold).lower() == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        tval = max(0, min(255, int(float(threshold))))
        _, th = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    proc = cv2.erode(opened, k, iterations=max(0, int(erosion)))
    cnts, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int,int,int,int]] = []
    H = scaled.shape[0]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if 30 <= area <= 7000:
            ar = w / float(h + 1e-6)
            if 0.3 <= ar <= 7.0 and h <= H * 0.18:
                boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, scaled


def _group_boxes_into_rows(boxes: List[Tuple[int,int,int,int]], y_tol: int = 14) -> List[Dict]:
    rows: List[Dict] = []
    for (x, y, w, h) in boxes:
        cy = y + h / 2.0
        placed = False
        for row in rows:
            if abs(cy - row["cy"]) <= y_tol:
                row["boxes"].append((x, y, w, h))
                row["cy"] = float(np.mean([b[1] + b[3]/2.0 for b in row["boxes"]]))
                placed = True
                break
        if not placed:
            rows.append({"cy": cy, "boxes": [(x, y, w, h)]})
    for r in rows:
        r["boxes"].sort(key=lambda b: b[0])
    rows.sort(key=lambda r: r["cy"])
    return rows


def _ocr_words(image: np.ndarray) -> List[dict]:
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except Exception:
        return []
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf_raw = str(data.get("conf", ["-1"][0])[i])
        try:
            conf = int(conf_raw) if conf_raw.isdigit() else -1
        except Exception:
            conf = -1
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({"text": txt, "conf": conf, "bbox": (x, y, w, h)})
    return words


def _associate_rows_with_domains(rows: List[Dict], ocr: List[dict], domain_list: List[str]) -> Dict[str, List[Tuple[int,int,int,int]]]:
    assoc: Dict[str, List[Tuple[int,int,int,int]]] = {}
    used = set()
    def find_y_for_phrase(phrase: str) -> Optional[int]:
        parts = phrase.split()
        n = len(parts)
        best_y, best_conf = None, -1
        for i in range(len(ocr) - n + 1):
            seq = [ocr[i+j]["text"].lower() for j in range(n)]
            if seq == [p.lower() for p in parts]:
                conf_vals = [ocr[i+j]["conf"] for j in range(n) if ocr[i+j]["conf"] >= 0]
                conf = int(np.mean(conf_vals)) if conf_vals else 0
                y = ocr[i]["bbox"][1]
                if conf > best_conf:
                    best_conf, best_y = conf, y
        return best_y
    domain_y = {d: find_y_for_phrase(d) for d in domain_list}
    for d, ly in domain_y.items():
        if ly is None:
            continue
        best_row, best_dist = None, 1e9
        for idx, r in enumerate(rows):
            if idx in used:
                continue
            dist = abs(r["cy"] - ly)
            if dist < best_dist:
                best_dist, best_row = dist, idx
        if best_row is not None:
            assoc[d] = rows[best_row]["boxes"]
            used.add(best_row)
    remaining_rows = [rows[i] for i in range(len(rows)) if i not in used]
    remaining_domains = [d for d in domain_list if d not in assoc]
    for d, r in zip(remaining_domains, remaining_rows):
        assoc[d] = r["boxes"]
    for d in domain_list:
        assoc.setdefault(d, [])
    return assoc

@dataclass
class DomainResult:
    filled: int
    total: int
    empty: int


def infer_domain_results_for_half(half_img: np.ndarray, params: dict, domain_list: List[str]) -> Tuple[Dict[str, DomainResult], int]:
    boxes, scaled = _detect_boxes(half_img, params["threshold"], params["morph_size"], params["erosion"])
    rows = _group_boxes_into_rows(boxes, y_tol=14)
    ocr = _ocr_words(scaled)
    assoc = _associate_rows_with_domains(rows, ocr, domain_list)
    results: Dict[str, DomainResult] = {}
    for dom in domain_list:
        dom_boxes = assoc.get(dom, [])
        filled = 0
        for (x, y, w, h) in dom_boxes:
            crop = scaled[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            mean_val = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
            if mean_val < 140:  # dark = filled
                filled += 1
        total_expected = EXPECTED_PER_DOMAIN
        empty = max(0, total_expected - filled)
        results[dom] = DomainResult(filled=filled, total=total_expected, empty=empty)
    return results, len(boxes)


def detect_empty_categories(pdf_path: str, page_index: int = 0, dpi: int = 350) -> Dict[str, int]:
    params = load_params()
    pages = pdf_pages_to_images(pdf_path, dpi=dpi)
    if not pages:
        return {}
    page = pages[min(page_index, len(pages) - 1)]
    region = crop_region(page, params["crop_x1"], params["crop_x2"], params["crop_y1"], params["crop_y2"])
    left, right = half_split(region)
    rw_res, _ = infer_domain_results_for_half(left, params, DOMAINS_RW)
    m_res, _  = infer_domain_results_for_half(right, params, DOMAINS_MATH)
    all_results: Dict[str, DomainResult] = {**rw_res, **m_res}
    empties = {cat: dr.empty for cat, dr in all_results.items() if dr.empty > 0}
    return dict(sorted(empties.items(), key=lambda kv: kv[1], reverse=True))

# ---------------------------
# Flask routes
# ---------------------------

# CORS for local Next.js dev (optional)
@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get('Origin')
    if origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
        resp.headers['Access-Control-Allow-Credentials'] = 'true'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return resp

@app.route('/api/<path:path>', methods=['OPTIONS'])
def api_options(path: str):
    return ('', 204)

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({'ok': True, 'message': 'OpenSAT Coach backend ready', 'questions_loaded': len(QB.questions), 'max_quiz_questions': MAX_QUIZ_QUESTIONS})

@app.before_request
def maybe_autoload_local_csv():
    if not QB.questions:
        for candidate in ["questions.csv", os.path.join("data", "questions.csv"), "/mnt/data/questions.csv"]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, 'r', encoding='utf-8') as f:
                        QB.load_csv(f)
                        app.logger.info("Auto-loaded %s (%d questions)", candidate, len(QB.questions))
                except Exception:
                    pass
                break

@app.route("/")
def home():
    analysis = session.get('analysis')
    return render_template('home.html', analysis=analysis)

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    f = request.files.get('pdf')
    if not f:
        flash("No file uploaded.")
        return redirect(url_for('home'))
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)

    # Try vision-based domain analysis first
    focus_domains: List[str] = []
    rw_incorrect = None
    math_incorrect = None
    rw_score = None
    math_score = None

    try:
        empties = detect_empty_categories(path)
        if empties:
            # Choose up to 2 domains with the most empties
            focus_domains = [d for d, _ in list(empties.items())[:2]]
    except Exception as e:
        app.logger.warning(f"Vision detection failed: {e}")

    # Also try to parse textual scores as a supplement/fallback
    try:
        text = extract_text_from_pdf(path)
        if text.strip() and looks_like_score_report(text):
            diag = parse_score_report(text)
            rw_score = diag.rw_score
            math_score = diag.math_score
            rw_incorrect = diag.rw_incorrect
            math_incorrect = diag.math_incorrect
            # if focus from vision empty, borrow text-based focus
            if not focus_domains:
                focus_domains = diag.focus_domains
        else:
            if not focus_domains:
                focus_domains = ["Craft and Structure", "Standard English Conventions"]
    except Exception as e:
        app.logger.warning(f"Text parse failed: {e}")
        if not focus_domains:
            focus_domains = ["Craft and Structure", "Standard English Conventions"]

    # Compute simple progress bars from scores if present
    rw_pct = int(round(((rw_score or 0) - 200) / 6)) if rw_score else 0
    math_pct = int(round(((math_score or 0) - 200) / 6)) if math_score else 0

    session['analysis'] = {
        'total_score': None,
        'rw_score': rw_score,
        'math_score': math_score,
        'rw_incorrect': rw_incorrect,
        'math_incorrect': math_incorrect,
        'rw_pct': rw_pct,
        'math_pct': math_pct,
        'focus_domains': focus_domains,
    }
    flash("Analysis saved. See recommendations below.")
    return redirect(url_for('recommend'))

@app.route("/manual-scores", methods=["POST"])
def manual_scores():
    def val(name, cast=int):
        v = request.form.get(name)
        try:
            return cast(v) if v else None
        except Exception:
            return None
    rw_score = val('rw_score')
    math_score = val('math_score')
    rw_incorrect = val('rw_incorrect')
    math_incorrect = val('math_incorrect')
    rw_pct = int(round(((rw_score or 0) - 200) / 6)) if rw_score else 0
    math_pct = int(round(((math_score or 0) - 200) / 6)) if math_score else 0
    rw_err = rw_incorrect or 0
    math_err = math_incorrect or 0
    focus = ["Craft and Structure", "Standard English Conventions"] if rw_err >= math_err else ["Advanced Math", "Algebra"]
    session['analysis'] = {
        'total_score': None,
        'rw_score': rw_score,
        'math_score': math_score,
        'rw_incorrect': rw_incorrect,
        'math_incorrect': math_incorrect,
        'rw_pct': rw_pct,
        'math_pct': math_pct,
        'focus_domains': focus,
    }
    flash("Manual analysis saved.")
    return redirect(url_for('recommend'))

@app.route("/recommend")
def recommend():
    analysis = session.get('analysis')
    qstats = QB.stats()
    suggested_n = min(MAX_QUIZ_QUESTIONS, max(6, qstats['total'] // 20 or 6))
    return render_template('recommend.html', analysis=analysis, qstats=qstats, suggested_n=suggested_n)

@app.route("/questions")
def manage_questions():
    return render_template('questions.html', qstats=QB.stats())

@app.route("/upload-questions", methods=["POST"])
def upload_questions():
    f = request.files.get('csv')
    if not f:
        flash("No CSV provided.")
        return redirect(url_for('manage_questions'))
    try:
        QB.load_csv(io.StringIO(f.stream.read().decode('utf-8')))
        flash(f"Loaded {len(QB.questions)} questions.")
    except Exception as e:
        app.logger.exception("CSV load failed")
        flash(f"CSV load failed: {e}")
    return redirect(url_for('manage_questions'))

# ---------- Quiz flow ----------

@app.route("/start-quiz", methods=["POST"])
def start_quiz():
    analysis = session.get('analysis') or {}
    domains_str = request.form.get('domains', '')
    domains = [d.strip() for d in domains_str.split(',') if d.strip()] or analysis.get('focus_domains', [])
    n = request.form.get('n', type=int) or 10
    n = min(max(3, n), MAX_QUIZ_QUESTIONS)

    questions = QB.pick(domains, n)
    quiz = {
        'domains': domains,
        'qids': [q.id for q in questions],
        'answers': [None]*len(questions),
        'results': [None]*len(questions),  # 'correct' / 'wrong'
    }
    session['quiz'] = quiz
    return redirect(url_for('show_question', qidx=0))

@app.route("/quiz/<int:qidx>")
def show_question(qidx: int):
    quiz = session.get('quiz')
    if not quiz:
        return render_template('quiz.html', quiz=None)
    total = len(quiz['qids'])
    qid = quiz['qids'][qidx]
    # Find the question in the main question bank
    q = None
    for question in QB.questions:
        if question.id == qid:
            q = question
            break
    if not q:
        # Question not found, redirect to recommendations
        return redirect(url_for('recommend'))
    user_answer = quiz['answers'][qidx]
    result = quiz['results'][qidx]
    return render_template('quiz.html', quiz=quiz, q=q, idx=qidx, total=total, user_answer=user_answer, result=result)

@app.route("/quiz/<int:qidx>/submit", methods=["POST"])
def submit_answer(qidx: int):
    quiz = session.get('quiz')
    if not quiz:
        return redirect(url_for('recommend'))
    answer = request.form.get('answer')
    if answer is None:
        return redirect(url_for('show_question', qidx=qidx))
    qid = quiz['qids'][qidx]
    # Find the question in the main question bank
    q = None
    for question in QB.questions:
        if question.id == qid:
            q = question
            break
    if not q:
        return redirect(url_for('recommend'))
    quiz['answers'][qidx] = answer
    result = 'correct' if (q.answer_letter and answer.upper() == q.answer_letter.upper()) else 'wrong'
    quiz['results'][qidx] = result
    session['quiz'] = quiz
    return redirect(url_for('show_question', qidx=qidx))

@app.route("/quiz/<int:qidx>/next")
def next_question(qidx: int):
    quiz = session.get('quiz')
    if not quiz:
        return redirect(url_for('recommend'))
    if qidx + 1 < len(quiz['qids']):
        return redirect(url_for('show_question', qidx=qidx+1))
    # Done — simple summary
    correct = sum(1 for r in quiz['results'] if r == 'correct')
    total = len(quiz['results'])
    flash(f"Quiz complete: {correct}/{total} correct. Nice work!")
    return redirect(url_for('recommend'))

# Route aliases for forms in templates
app.add_url_rule('/upload_pdf', 'upload_pdf', upload_pdf, methods=['POST'])
app.add_url_rule('/manual_scores', 'manual_scores', manual_scores, methods=['POST'])
app.add_url_rule('/upload_questions', 'upload_questions', upload_questions, methods=['POST'])
app.add_url_rule('/submit/<int:qidx>', 'submit_answer', submit_answer, methods=['POST'])
app.add_url_rule('/next/<int:qidx>', 'next_question', next_question)

if __name__ == "__main__":
    app.run(debug=True)
