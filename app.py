import html
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Text Detector", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, textarea, input { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; visibility: hidden; }
.stApp {
  color:#0F172A; background-attachment: fixed;
  background-color:#EEF2FF;
  background-image:
    radial-gradient(900px 480px at 6% -6%, rgba(99,102,241,.20), transparent 62%),
    radial-gradient(760px 420px at 98% 2%, rgba(56,189,248,.22), transparent 60%),
    radial-gradient(700px 500px at 60% 105%, rgba(167,139,250,.16), transparent 60%);
}
.block-container { max-width: 1140px; padding: 1.6rem 2rem 4rem 2rem; }

/* hero */
.hero { display:flex; justify-content:space-between; align-items:center; gap:32px; padding:34px 38px; margin-bottom:22px;
  background: linear-gradient(135deg, #1E2A78 0%, #3347C9 55%, #4F7CF3 100%); border-radius:20px; color:#fff;
  box-shadow: 0 18px 40px rgba(51,71,201,.25); position:relative; overflow:hidden; }
.hero:before { content:""; position:absolute; right:-80px; top:-90px; width:300px; height:300px; border-radius:50%; background:rgba(255,255,255,.07); }
.hero:after { content:""; position:absolute; right:140px; bottom:-120px; width:240px; height:240px; border-radius:50%; background:rgba(255,255,255,.05); }
.hero-left { position:relative; z-index:1; max-width:560px; }
.pill { display:inline-block; font-size:.74rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; padding:5px 12px; border-radius:999px; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); }
.hero h1 { font-size:2.15rem; line-height:1.15; font-weight:800; letter-spacing:-0.025em; margin:14px 0 10px 0; color:#fff; padding:0; }
.hero p { color:rgba(255,255,255,.82); font-size:.98rem; line-height:1.6; margin:0; }
.hero-right { display:flex; gap:14px; position:relative; z-index:1; }
.hstat { background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.25); border-radius:14px; padding:16px 20px; min-width:118px; backdrop-filter: blur(4px); }
.hstat b { display:block; font-size:1.65rem; font-weight:800; letter-spacing:-0.02em; }
.hstat span { font-size:.72rem; color:rgba(255,255,255,.8); text-transform:uppercase; letter-spacing:.06em; }

/* tabs as segmented control */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.75); border:1px solid rgba(99,102,241,.14); border-radius:14px; padding:6px; width:fit-content; box-shadow:0 4px 14px rgba(30,58,138,.06); }
.stTabs [data-baseweb="tab"] { height: 40px; padding: 0 18px; border-radius:10px; background: transparent; font-weight:600; font-size:.9rem; color:#64748B; }
.stTabs [aria-selected="true"] { background:#fff !important; color:#3347C9 !important; box-shadow:0 2px 8px rgba(51,71,201,.15); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }

/* headings and text */
.eyebrow { font-size:.74rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#4F5BD5; margin: 28px 0 10px 0; display:flex; align-items:center; gap:10px; }
.eyebrow:after { content:""; flex:1; height:1px; background:linear-gradient(90deg, rgba(79,91,213,.25), transparent); }
h3.page-title { font-size:1.45rem; font-weight:800; letter-spacing:-0.02em; margin: 26px 0 4px 0; color:#0F172A; }
.lead { color:#475569; font-size:.95rem; margin-bottom:14px; max-width:780px; line-height:1.65; }
.muted { color:#64748B; font-size:.86rem; }

/* cards */
.card { background:#fff; border:1px solid rgba(99,102,241,.13); border-radius:16px; padding:22px 24px; box-shadow:0 10px 28px rgba(30,58,138,.07); }
.result { display:flex; align-items:center; gap:30px; }
.ring { position:relative; width:150px; height:150px; flex:none; }
.ring-n { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:2.5rem; font-weight:800; letter-spacing:-0.03em; }
.ring-n small { font-size:1.1rem; font-weight:700; margin-left:2px; margin-top:8px; }
.res-text { flex:1; }
.verdict { font-size:1.35rem; font-weight:800; letter-spacing:-0.01em; margin-top:4px; }
.scale { position:relative; height:10px; border-radius:5px; margin: 22px 0 6px 0; display:flex; }
.scale div.z { height:10px; }
.z1 { width:35%; background:#86EFAC; border-radius:5px 0 0 5px; }
.z2 { width:30%; background:#FCD34D; }
.z3 { width:35%; background:#FCA5A5; border-radius:0 5px 5px 0; }
.pin { position:absolute; top:-5px; width:4px; height:20px; background:#1E2A78; border-radius:2px; box-shadow:0 0 0 2px #fff; }
.scale-labels { display:flex; justify-content:space-between; font-size:.72rem; color:#64748B; }
.kv { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid #EEF2FF; font-size:.9rem; }
.kv:last-child { border-bottom:none; }
.kv span:first-child { color:#64748B; }
.kv span:last-child { font-weight:700; color:#0F172A; }
.note { background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid #F59E0B; border-radius:10px; padding:12px 16px; color:#78350F; font-size:.88rem; margin: 10px 0 14px 0; }

/* sentence view */
.sentbox { background:#fff; border:1px solid rgba(99,102,241,.13); border-radius:16px; padding:22px 26px; line-height:2.2; font-size:.98rem; color:#1E293B; box-shadow:0 10px 28px rgba(30,58,138,.07); }
.s { padding:2px 3px; border-radius:4px; border-bottom:2px solid transparent; }
.legend { display:flex; flex-wrap:wrap; gap:18px; font-size:.78rem; color:#64748B; margin: 2px 0 12px 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:middle; }

/* stat tiles */
.tiles { display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin: 6px 0; }
.tile { background:#fff; border:1px solid rgba(99,102,241,.13); border-top:3px solid #4F5BD5; border-radius:14px; padding:16px 18px; box-shadow:0 8px 22px rgba(30,58,138,.06); }
.tile .v { font-size:1.6rem; font-weight:800; letter-spacing:-0.02em; color:#1E2A78; }
.tile .l { font-size:.72rem; color:#64748B; margin-top:4px; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.tile.alert { border-top-color:#F59E0B; }
.tile.alert .v { color:#B45309; }

/* tables */
table.t { width:100%; border-collapse:separate; border-spacing:0; background:#fff; border:1px solid rgba(99,102,241,.13); border-radius:14px; overflow:hidden; font-size:.9rem; margin: 6px 0 14px 0; box-shadow:0 8px 22px rgba(30,58,138,.06); }
table.t th { text-align:left; background:#EEF2FF; color:#3B4BB8; font-weight:700; font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; padding:12px 16px; }
table.t td { padding:12px 16px; border-top:1px solid #F1F4FB; color:#1E293B; }
table.t td:first-child { font-weight:600; color:#0F172A; }

/* bars */
.bar-row { display:grid; grid-template-columns: 220px 1fr 64px; align-items:center; gap:14px; margin:12px 0; font-size:.9rem; }
.bar-track { height:11px; background:#E6EBFA; border-radius:6px; overflow:hidden; }
.bar-fill { height:11px; background:linear-gradient(90deg,#3347C9,#4F9DF3); border-radius:6px; }
.bar-val { text-align:right; font-weight:700; color:#1E2A78; }

/* inputs and buttons */
.stTextArea textarea { background:#fff; border:1px solid #C7D2FE; border-radius:14px; font-size:.97rem; line-height:1.65; padding:16px 18px; box-shadow:0 8px 22px rgba(30,58,138,.06); }
.stTextArea textarea:focus { border-color:#4F5BD5; box-shadow:0 0 0 3px rgba(79,91,213,.18); }
.stTextArea label { font-weight:700; font-size:.86rem; color:#334155; }
.stButton > button { border-radius:10px; font-weight:600; font-size:.88rem; padding:.5rem 1.2rem; border:1px solid #C7D2FE; background:#fff; color:#3347C9; transition: all .15s; }
.stButton > button:hover { border-color:#3347C9; background:#F5F7FF; color:#1E2A78; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#3347C9,#4F7CF3); border:none; color:#fff; box-shadow:0 8px 18px rgba(51,71,201,.32); }
.stButton > button[kind="primary"]:hover { filter:brightness(1.07); color:#fff; }
.stButton > button:disabled { opacity:.5; }
[data-testid="stExpander"] { background:#fff; border:1px solid rgba(99,102,241,.13); border-radius:14px; }
</style>
""", unsafe_allow_html=True)

SAMPLES = {
    "ai": "Artificial intelligence has transformed numerous industries in recent years. "
          "Moreover, it offers significant benefits, including increased efficiency and improved decision-making. "
          "However, it is important to consider the ethical implications of its widespread adoption. "
          "In conclusion, a balanced approach is essential to harness its full potential responsibly.",
    "human": "Honestly I didn't expect the bus to be late again, so I just walked home in the rain. "
             "My shoes were soaked by the time I got to the door, and mum laughed at me for ten minutes straight. "
             "We ended up eating leftover dal and watching an old cricket match. Best evening this week, weirdly.",
}


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model")
def load_models():
    from src.detector import Detector
    det = Detector("models/transformer")
    base = None
    if os.path.exists("models/baseline.joblib"):
        import joblib
        base = joblib.load("models/baseline.joblib")
    return det, base


def band(p):
    if p < 0.35:
        return "Likely human-written", "#15803D"
    if p <= 0.65:
        return "Inconclusive", "#B45309"
    return "Likely AI-generated", "#B91C1C"


def sent_style(p):
    if p > 0.65:
        rgb = "185,28,28"
    elif p < 0.35:
        rgb = "21,128,61"
    else:
        rgb = "180,83,9"
    strength = abs(p - 0.5) * 2
    return f"background:rgba({rgb},{0.06 + 0.16 * strength});border-bottom-color:rgba({rgb},{0.45 + 0.4 * strength})"


def html_table(df, index_name=None):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in ([index_name or ""] + list(df.columns)))
    body = ""
    for idx, row in df.iterrows():
        body += "<tr><td>" + html.escape(str(idx)) + "</td>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
    st.markdown(f'<table class="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def bars(data):
    out = ""
    for k, v in data.items():
        out += (f'<div class="bar-row"><div>{html.escape(k)}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{v*100:.1f}%"></div></div>'
                f'<div class="bar-val">{v*100:.1f}%</div></div>')
    st.markdown(f'<div class="card">{out}</div>', unsafe_allow_html=True)


def tiles(m, flag_fpr=True):
    items = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
             ("f1", "F1 score"), ("false_positive_rate", "False-positive rate")]
    out = ""
    for k, lab in items:
        v = f"{m[k]*100:.1f}%" if k != "f1" else f"{m[k]:.3f}"
        cls = "tile alert" if (k == "false_positive_rate" and flag_fpr) else "tile"
        out += f'<div class="{cls}"><div class="v">{v}</div><div class="l">{lab}</div></div>'
    st.markdown(f'<div class="tiles">{out}</div>', unsafe_allow_html=True)


def load_results():
    if os.path.exists("report/results.json"):
        return json.load(open("report/results.json"))
    return None


def set_text(key):
    st.session_state.text = SAMPLES.get(key, "")


# ------------------------------------------------------------------ header
_R = load_results()
if _R:
    _stats = [(f"{_R['transformer']['accuracy']*100:.1f}%", "Test accuracy"),
              (f"{_R['transformer']['false_positive_rate']*100:.1f}%", "False positives"),
              ("120k", "Training texts")]
else:
    _stats = [("120k", "Training texts"), ("2", "Models"), ("4", "Stress tests")]
_stat_html = "".join(f'<div class="hstat"><b>{v}</b><span>{l}</span></div>' for v, l in _stats)
st.markdown(
    '<div class="hero"><div class="hero-left"><div class="pill">AI Text Detector</div>'
    '<h1>Detect AI-generated writing, and understand where detection fails.</h1>'
    '<p>A fine-tuned transformer estimates whether a passage is human or machine-written, '
    'then shows how reliable that estimate really is.</p></div>'
    f'<div class="hero-right">{_stat_html}</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Analyse", "Model performance", "Testing and validation", "Limitations and policy"])

# ------------------------------------------------------------------ Analyse
with tab1:
    st.markdown('<h3 class="page-title">Analyse a passage</h3>'
                '<div class="lead">Paste text to estimate the likelihood that it was generated by an AI model. '
                'The result is a probabilistic estimate and should not be used as proof of authorship.</div>',
                unsafe_allow_html=True)

    if not os.path.exists("models/transformer"):
        st.error("Model not found. Run the training steps in the README first.")
        st.stop()
    det, base = load_models()

    if "text" not in st.session_state:
        st.session_state.text = ""

    text = st.text_area("Input text", key="text", height=210,
                        placeholder="Paste at least 50 words for a meaningful result.")
    words = len(text.split())

    b1, b2, b3, b4, _ = st.columns([1.1, 1.3, 1.3, 0.8, 4])
    go = b1.button("Analyse", type="primary", disabled=words == 0)
    b2.button("Load AI sample", on_click=set_text, args=("ai",))
    b3.button("Load human sample", on_click=set_text, args=("human",))
    b4.button("Clear", on_click=set_text, args=("",))
    st.markdown(f'<div class="muted" style="margin-top:6px">{words} words</div>', unsafe_allow_html=True)

    if go:
        if words < 50:
            st.markdown('<div class="note">This passage is under 50 words. Detectors are unreliable on short '
                        'text, so treat the result below as close to meaningless.</div>', unsafe_allow_html=True)
        with st.spinner("Analysing"):
            overall, sents, per = det.analyze(text)
            p_base = float(base.predict_proba([text])[0, 1]) if base else None
        label, color = band(overall)

        st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)
        left, right = st.columns([3, 2], gap="medium")
        with left:
            circ = 2 * 3.14159 * 54
            dash = circ * overall
            st.markdown(
                f'<div class="card result"><div class="ring">'
                f'<svg viewBox="0 0 140 140" width="150" height="150">'
                f'<circle cx="70" cy="70" r="54" fill="none" stroke="#E6EBFA" stroke-width="12"/>'
                f'<circle cx="70" cy="70" r="54" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
                f'stroke-dasharray="{dash:.1f} {circ:.1f}" transform="rotate(-90 70 70)"/></svg>'
                f'<div class="ring-n" style="color:{color}">{overall*100:.0f}<small>%</small></div></div>'
                f'<div class="res-text"><div class="muted">Probability of AI-generated text</div>'
                f'<div class="verdict" style="color:{color}">{label}</div>'
                f'<div class="scale"><div class="z z1"></div><div class="z z2"></div><div class="z z3"></div>'
                f'<div class="pin" style="left:calc({overall*100:.1f}% - 2px)"></div></div>'
                f'<div class="scale-labels"><span>Human</span><span>Inconclusive</span><span>AI</span></div></div></div>',
                unsafe_allow_html=True)
        with right:
            rows = f'<div class="kv"><span>Words</span><span>{words}</span></div>'
            rows += f'<div class="kv"><span>Sentences</span><span>{len(sents)}</span></div>'
            rows += f'<div class="kv"><span>Transformer score</span><span>{overall*100:.1f}%</span></div>'
            if p_base is not None:
                agree = (p_base >= 0.5) == (overall >= 0.5)
                rows += f'<div class="kv"><span>Baseline score</span><span>{p_base*100:.1f}%</span></div>'
                rows += f'<div class="kv"><span>Model agreement</span><span>{"Agree" if agree else "Disagree"}</span></div>'
            st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

        st.markdown('<div class="eyebrow">Sentence-level breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="legend"><span><i style="background:#86EFAC"></i>More human-like</span>'
                    '<span><i style="background:#FCD34D"></i>Inconclusive</span>'
                    '<span><i style="background:#FCA5A5"></i>More AI-like</span>'
                    '<span>Sentence scores are noisy and indicative only.</span></div>', unsafe_allow_html=True)
        spans = " ".join(f'<span class="s" style="{sent_style(p)}" title="{p*100:.0f}% AI">{html.escape(s)}</span>'
                         for s, p in zip(sents, per))
        st.markdown(f'<div class="sentbox">{spans}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Model performance
with tab2:
    st.markdown('<h3 class="page-title">Model performance</h3>'
                '<div class="lead">Results on a held-out test set that the models never saw during training.</div>',
                unsafe_allow_html=True)
    R = load_results()
    if not R:
        st.info("Run `python -m src.evaluate` to generate the report.")
    else:
        st.markdown(f'<div class="muted">Test set: {R["n_test"]:,} texts, balanced between human and AI.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Fine-tuned DistilRoBERTa</div>', unsafe_allow_html=True)
        tiles(R["transformer"])
        st.markdown('<div class="eyebrow">TF-IDF and Logistic Regression (baseline)</div>', unsafe_allow_html=True)
        tiles(R["baseline"])

        st.markdown('<div class="eyebrow">Confusion matrices</div>', unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 1])
        if os.path.exists("report/confusion_transformer.png"):
            c1.image("report/confusion_transformer.png")
        if os.path.exists("report/confusion_baseline.png"):
            c2.image("report/confusion_baseline.png")

        st.markdown('<div class="eyebrow">Robustness to simple edits</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted" style="margin-bottom:6px">Share of AI-written texts still detected after the edit.</div>',
                    unsafe_allow_html=True)
        bars(R["robustness_ai_recall"])

        if os.path.exists("report/failures.csv"):
            with st.expander("Most confident mistakes"):
                st.dataframe(pd.read_csv("report/failures.csv").head(30), use_container_width=True)

# ------------------------------------------------------------------ Testing and validation
with tab3:
    st.markdown('<h3 class="page-title">Testing, validation and performance evaluation</h3>'
                '<div class="lead">How the detector was built, tested and stress-tested, with the actual results.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">1. Experimental setup</div>', unsafe_allow_html=True)
    setup = pd.DataFrame({"Detail": [
        "artem9k/ai-text-detection-pile (HuggingFace)",
        "120,000 texts, 50% human and 50% AI",
        "80% / 10% / 10%, stratified by label",
        "DistilRoBERTa fine-tuned; baseline is TF-IDF with Logistic Regression",
        "40,000 texts, 1 epoch, batch size 16, learning rate 3e-5, 256 tokens",
        "MacBook with Apple silicon (MPS)",
        "6,000 held-out texts",
        "0.5 on the probability of AI"]},
        index=["Dataset", "Balanced size", "Train / validation / test", "Models", "Training",
               "Hardware", "Test sample", "Decision threshold"])
    html_table(setup, "Item")

    st.markdown('<div class="eyebrow">2. Training behaviour</div>', unsafe_allow_html=True)
    losses = [0.5413, 0.2826, 0.1941, 0.2109, 0.1505, 0.1733, 0.1517, 0.1216, 0.1376, 0.1148, 0.1287,
              0.1184, 0.1213, 0.09674, 0.09068, 0.09608, 0.08456, 0.07584, 0.07031, 0.06677, 0.06898]
    st.line_chart(pd.DataFrame({"Training loss": losses}, index=[(i + 1) * 100 for i in range(len(losses))]),
                  color="#1E3A8A", height=240)
    st.markdown('<div class="muted">Loss logged every 100 steps. It fell from 0.54 to about 0.07 over one epoch with no divergence.</div>',
                unsafe_allow_html=True)

    R = load_results()
    if R:
        st.markdown(f'<div class="eyebrow">3. Test-set results ({R["n_test"]:,} unseen texts)</div>', unsafe_allow_html=True)
        rows = {}
        for name, key in [("TF-IDF + Logistic Regression", "baseline"), ("DistilRoBERTa", "transformer")]:
            m = R[key]
            rows[name] = [f"{m['accuracy']*100:.2f}%", f"{m['precision']*100:.2f}%", f"{m['recall']*100:.2f}%",
                          f"{m['f1']:.3f}", f"{m['auc']:.3f}", f"{m['false_positive_rate']*100:.2f}%"]
        html_table(pd.DataFrame(rows, index=["Accuracy", "Precision", "Recall", "F1", "AUC", "False-positive rate"]).T, "Model")
        t, b = R["transformer"], R["baseline"]
        st.markdown(
            f'<div class="card" style="font-size:.93rem;line-height:1.7;color:#334155">'
            f'The fine-tuned transformer improves on the baseline by {(t["accuracy"]-b["accuracy"])*100:.1f} accuracy points. '
            f'It detects {t["recall"]*100:.1f}% of AI texts but wrongly flags {t["false_positive_rate"]*100:.1f}% of human texts, '
            f'roughly 1 in {round(1/t["false_positive_rate"])} human writers.</div>', unsafe_allow_html=True)

        st.markdown('<div class="eyebrow">4. Stress test: can the detector be evaded?</div>', unsafe_allow_html=True)
        bars(R["robustness_ai_recall"])
        st.markdown(
            '<div class="card" style="font-size:.93rem;line-height:1.7;color:#334155;margin-top:12px">'
            'High accuracy on a clean test set does not mean the detector is robust. Small edits such as typos or removing '
            'punctuation sharply reduce detection, which suggests the model partly relies on how clean the text looks '
            'rather than on any intrinsic property of machine writing. This supports the central finding of the project: '
            'AI-text detection is fundamentally unreliable.</div>', unsafe_allow_html=True)
    else:
        st.info("Run `python -m src.evaluate` to generate the test results.")

    st.markdown('<div class="eyebrow">5. Failure analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">The most confident mistakes are saved in report/failures.csv and can be browsed in the Model performance tab. '
                'Reviewing them shows which human texts read as AI-like (formal or templated writing) and which AI texts pass as human.</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ Limitations
with tab4:
    st.markdown('<h3 class="page-title">Limitations and academic integrity</h3>'
                '<div class="lead">Why this tool, and any tool like it, cannot settle questions of authorship.</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown("""
<div class="card"><b>Why detection is fundamentally unreliable</b>
<ul style="line-height:1.8;color:#334155;font-size:.92rem;margin-top:10px">
<li><b>Paraphrasing and light editing</b> change the surface statistics detectors rely on.</li>
<li><b>New models keep appearing.</b> A detector learns the style of the generators it was trained on.</li>
<li><b>Human and AI text overlap.</b> Formal or templated human writing looks machine-like.</li>
<li><b>Non-native English writers</b> are flagged falsely more often, a documented fairness problem.</li>
<li><b>Short texts</b> carry too little signal.</li>
<li><b>Distribution shift.</b> Real essays, code and emails differ from a fixed benchmark.</li>
<li><b>Detection is adversarial.</b> Generators improve faster than detectors.</li>
</ul></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""
<div class="card"><b>Implications for academic integrity</b>
<ul style="line-height:1.8;color:#334155;font-size:.92rem;margin-top:10px">
<li>A detector score should never be the sole evidence for a misconduct allegation.</li>
<li>False positives harm real students, so the burden of proof must stay with the institution.</li>
<li>Prefer process-based assessment: drafts, version history, oral follow-ups, in-class writing.</li>
<li>Define permitted AI use per course and let students disclose it.</li>
<li>If used at all, treat a score as a prompt for conversation, not a verdict.</li>
</ul></div>""", unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Limitations of this project</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card"><ul style="line-height:1.8;color:#334155;font-size:.92rem;margin:0">
<li>English text only, trained on a limited set of generators.</li>
<li>Trained on 40,000 samples for one epoch on a laptop; larger models and more data would help.</li>
<li>Probabilities are not calibrated guarantees.</li>
<li>Not tested on mixed human and AI documents or on output from dedicated paraphrasing tools.</li>
<li>The evasion tests are simple edits; a determined adversary would do far better.</li>
</ul></div>""", unsafe_allow_html=True)