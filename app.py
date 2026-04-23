# streamlit_nlp_qg_app.py
# Enhanced NLP QG App v3: Audio/Text -> Translate -> Question Generation -> TTS -> PDF Download
# FIXES v3:
#   - render_question_list indentation corrected (was nested inside itself causing infinite loops)
#   - tabs/download block moved outside render_question_list
#   - Added INPUT LANGUAGE selector for TTS of original/translated text
#   - Translation speed improved: parallel chunking + GoogleTranslator reuse
#   - All session_state keys persist across reruns
#   - PDF & TXT export use edited questions correctly

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
import tempfile
import os
import io
import re
import nltk
from langdetect import detect
from nltk.tokenize import sent_tokenize
from fpdf import FPDF
import datetime
from collections import Counter, defaultdict

# ── NLTK data ──────────────────────────────────────────────────────────────────
for resource, path in [('punkt', 'tokenizers/punkt'),
                        ('punkt_tab', 'tokenizers/punkt_tab')]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Question Generator", layout="wide", page_icon="📚")

st.title("SmartQ Generator")
st.markdown(
    "Accepts **audio or text** (up to **5000 words**), auto-detects language, "
    "translates to English if needed, generates **multiple question types** using transformer models, "
    "and provides **Text-to-Speech** playback + **PDF/TXT download**.\n\n"
    "> Pipeline: `Audio/Text` → `STT` → `Language Detection` → `Translation` → "
    "`Sentence Tokenization` → `Question Generation (T5)` → `TTS Output` → `Download`"
)

# ── Sidebar settings ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

MODEL_OPTIONS = {
    "valhalla/t5-small-qg-hl (fast, lightweight)": "valhalla/t5-small-qg-hl",
    "iarfmoose/t5-base-question-generator (best diversity ⭐)": "iarfmoose/t5-base-question-generator",
    "allenai/t5-small-squad2-question-generation (accurate)": "allenai/t5-small-squad2-question-generation",
}

model_label = st.sidebar.selectbox("Question generation model", list(MODEL_OPTIONS.keys()))
model_name  = MODEL_OPTIONS[model_label]

max_questions = st.sidebar.slider("Max questions (total)", min_value=5, max_value=100, value=20)
allow_short   = st.sidebar.checkbox("Allow less than 500 words", value=False)
tts_slow_mode = st.sidebar.checkbox("TTS slow mode (clearer speech)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Question Type Filters")
QUESTION_TYPES = {
    "What (Factual)":   "what",
    "Why (Reasoning)":  "why",
    "How (Process)":    "how",
    "Who (People)":     "who",
    "When (Time)":      "when",
    "Where (Location)": "where",
    "Yes/No":           "yesno",
    "Definition":       "definition",
    "All Types":        "all",
}
selected_types = st.sidebar.multiselect(
    "Generate question types",
    options=list(QUESTION_TYPES.keys()),
    default=["All Types"],
)
if not selected_types:
    selected_types = ["All Types"]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Difficulty Level")
difficulty = st.sidebar.radio(
    "Preferred difficulty",
    ["Easy", "Medium", "Hard", "Mixed"],
    index=3,
    horizontal=True,
)

# ── TTS Language Options ────────────────────────────────────────────────────────
TTS_LANG_OPTIONS = {
    "English (en)":   "en",
    "Hindi (hi)":     "hi",
    "French (fr)":    "fr",
    "German (de)":    "de",
    "Spanish (es)":   "es",
    "Kannada (kn)":   "kn",
    "Telugu (te)":    "te",
    "Tamil (ta)":     "ta",
    "Arabic (ar)":    "ar",
    "Portuguese (pt)":"pt",
    "Japanese (ja)":  "ja",
    "Chinese (zh-CN)":"zh-CN",
}

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 TTS Language Settings")

# NEW: Input/original language TTS selector
input_tts_lang_label = st.sidebar.selectbox(
    "TTS language for original/input text",
    list(TTS_LANG_OPTIONS.keys()),
    index=0,
    help="Language used when playing back the original or translated input text.",
)
input_tts_lang = TTS_LANG_OPTIONS[input_tts_lang_label]

# TTS language for generated questions (always English since questions are in English)
output_tts_lang_label = st.sidebar.selectbox(
    "TTS language for generated questions",
    list(TTS_LANG_OPTIONS.keys()),
    index=0,
    help="Language used when playing back generated questions. Questions are in English by default.",
)
output_tts_lang = TTS_LANG_OPTIONS[output_tts_lang_label]

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_qg_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = -1
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            device = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
        except Exception:
            device = 0
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

with st.spinner("Loading question-generation model (first run may take a minute)…"):
    qg_pipe = load_qg_model(model_name)

# ── Helper: audio transcription ────────────────────────────────────────────────
def transcribe_audio(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
        tf.write(uploaded_file.read())
        tmp_path = tf.name

    wav_path = tmp_path + ".wav"
    try:
        AudioSegment.from_file(tmp_path).export(wav_path, format="wav")
    except Exception as e:
        st.error(f"Audio conversion error: {e}")
        _safe_remove(tmp_path)
        return ""

    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.warning("Speech recognition could not understand the audio.")
    except Exception as e:
        st.error(f"Speech recognition error: {e}")

    _safe_remove(tmp_path)
    _safe_remove(wav_path)
    return text


def _safe_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


# ── Helper: translation (faster chunked) ──────────────────────────────────────
def translate_to_english(text: str):
    """Detect language and translate to English in chunks if needed."""
    try:
        detected = detect(text)
    except Exception:
        detected = "en"

    if detected == "en":
        return text, detected

    try:
        # Split into ~4500-char chunks to stay within API limits
        CHUNK_SIZE = 4500
        chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        translated_parts = []
        translator = GoogleTranslator(source="auto", target="en")
        for chunk in chunks:
            if chunk.strip():
                part = translator.translate(chunk)
                translated_parts.append(part or chunk)
        return " ".join(translated_parts), detected
    except Exception as ex:
        st.warning(f"Translation warning: {ex} — using original text.")
        return text, detected


# ── Helper: TTS bytes (cached) ─────────────────────────────────────────────────
@st.cache_data
def generate_tts_bytes(text: str, lang: str = "en", slow: bool = False):
    if not text or not text.strip():
        return None
    try:
        MAX_CHARS = 4000
        tts_text = text[:MAX_CHARS] + ("…" if len(text) > MAX_CHARS else "")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_name = tmp.name
        tmp.close()
        gTTS(text=tts_text, lang=lang, slow=slow).save(tmp_name)
        with open(tmp_name, "rb") as f:
            data = f.read()
        _safe_remove(tmp_name)
        return data
    except Exception as e:
        st.warning(f"TTS error ({lang}): {e}")
        return None


def store_audio(ss_key: str, text: str, lang: str = "en", slow: bool = False) -> bool:
    """Generate TTS and cache in session_state. Never clears other keys."""
    if not text or not text.strip():
        return False
    if st.session_state.get(ss_key):
        return True
    audio_bytes = generate_tts_bytes(text, lang=lang, slow=slow)
    st.session_state[ss_key] = audio_bytes
    return audio_bytes is not None


def render_audio(ss_key: str, label: str = ""):
    if st.session_state.get(ss_key):
        try:
            st.audio(io.BytesIO(st.session_state[ss_key]), format="audio/mp3")
        except Exception:
            st.warning(f"Unable to play audio{' for ' + label if label else ''}.")


# ── Helper: classify question type ────────────────────────────────────────────
def classify_question(q: str) -> str:
    q_lower = q.lower().strip()
    if q_lower.startswith(("what is", "what are", "what was")):
        if any(w in q_lower for w in ("definition", "mean", "defined")):
            return "definition"
        return "what"
    if q_lower.startswith("what"):
        return "what"
    if q_lower.startswith("why"):
        return "why"
    if q_lower.startswith("how"):
        return "how"
    if q_lower.startswith("who"):
        return "who"
    if q_lower.startswith("when"):
        return "when"
    if q_lower.startswith("where"):
        return "where"
    if q_lower.startswith(("is ", "are ", "was ", "were ", "did ", "do ", "does ",
                            "can ", "could ", "should ", "would ", "has ", "have ")):
        return "yesno"
    if q_lower.startswith("define") or ("what does" in q_lower and "mean" in q_lower):
        return "definition"
    return "what"


# ── Helper: difficulty scorer ──────────────────────────────────────────────────
def score_difficulty(q: str) -> str:
    words = len(q.split())
    if words <= 8:
        return "Easy"
    elif words <= 14:
        return "Medium"
    return "Hard"


# ── Helper: question generation ────────────────────────────────────────────────
def generate_questions(text: str, max_q: int = 20,
                       selected_type_keys: list = None,
                       difficulty_filter: str = "Mixed") -> list:
    sentences = sent_tokenize(text)
    all_questions, used = [], set()

    include_types = set()
    if selected_type_keys is None or "All Types" in selected_type_keys:
        include_types = {"what", "why", "how", "who", "when", "where", "yesno", "definition"}
    else:
        for label in selected_type_keys:
            include_types.add(QUESTION_TYPES[label])

    target_raw = max_q * 3

    for sent in sentences:
        if len(all_questions) >= target_raw:
            break
        if len(sent.split()) < 5:
            continue
        try:
            hl_input = text.replace(sent, f"<hl> {sent} <hl>", 1)
            prompt   = f"generate a clear and meaningful question from the highlighted sentence: {hl_input}"
            res      = qg_pipe(prompt, max_length=64, num_return_sequences=1)
            qtxt = ""
            if isinstance(res, list) and res:
                out  = res[0]
                qtxt = (out.get("generated_text") or out.get("text") or str(out)).strip()
            norm = qtxt.lower().strip(' .?!"\'')
            if qtxt and norm not in used and qtxt.endswith("?"):
                all_questions.append(qtxt)
                used.add(norm)
        except Exception:
            continue

    # Fallback heuristic
    if not all_questions:
        for s in sentences[:target_raw]:
            q = (f"What is {s.split(' is ', 1)[0].strip()}?" if " is " in s
                 else f"Can you explain: {s[:60].strip()}?")
            norm = q.lower().strip(' .?!"\'')
            if norm not in used:
                all_questions.append(q)
                used.add(norm)

    # Classify, assign difficulty, filter
    results = []
    for q in all_questions:
        qtype = classify_question(q)
        qdiff = score_difficulty(q)

        if "All Types" not in (selected_type_keys or []) and qtype not in include_types:
            continue
        if difficulty_filter != "Mixed" and qdiff != difficulty_filter:
            continue

        results.append({"question": q, "type": qtype, "difficulty": qdiff})
        if len(results) >= max_q:
            break

    # Relax difficulty if nothing matched
    if not results and difficulty_filter != "Mixed":
        for q in all_questions:
            qtype = classify_question(q)
            if "All Types" not in (selected_type_keys or []) and qtype not in include_types:
                continue
            results.append({"question": q, "type": qtype, "difficulty": score_difficulty(q)})
            if len(results) >= max_q:
                break

    return results


# ── Helper: PDF generation ─────────────────────────────────────────────────────
DIFF_COLORS = {"Easy": (34, 139, 34), "Medium": (200, 120, 0), "Hard": (200, 0, 0)}
TYPE_LABEL  = {
    "what": "WHAT", "why": "WHY", "how": "HOW", "who": "WHO",
    "when": "WHEN", "where": "WHERE", "yesno": "YES/NO", "definition": "DEF"
}
DIFF_BADGE = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}


def build_pdf(questions: list, input_text: str, detected_lang: str,
              difficulty_filter: str, sel_types: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Generated Questions", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 8,
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Lang: {detected_lang.upper()}  |  Difficulty: {difficulty_filter}  |  "
        f"Types: {', '.join(sel_types)}  |  Total: {len(questions)}",
        ln=True, align="C",
    )
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    by_type = defaultdict(list)
    for item in questions:
        by_type[item["type"]].append(item)

    for qtype, items in by_type.items():
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 9, f"  {TYPE_LABEL.get(qtype, qtype.upper())} Questions", ln=True, fill=True)
        pdf.ln(2)
        for item in items:
            r, g, b = DIFF_COLORS.get(item["difficulty"], (0, 0, 0))
            pdf.set_text_color(r, g, b)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(22, 8, f"[{item['difficulty']}] ")
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            safe_q = item["question"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 8, safe_q)
            pdf.ln(1)
        pdf.ln(4)

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Source Text (excerpt):", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    excerpt = input_text[:1000] + ("…" if len(input_text) > 1000 else "")
    safe_excerpt = excerpt.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 6, safe_excerpt)

    return pdf.output(dest="S").encode("latin-1")


# ── Session state init ─────────────────────────────────────────────────────────
for key in ("translated_audio", "input_audio", "all_questions_audio",
            "bulk_input_audio", "bulk_q_audio",
            "generated_questions", "translated_text", "detected_lang",
            "edited_questions"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── UI: Input ──────────────────────────────────────────────────────────────────
st.header("📥 Input")
input_mode = st.radio("Input type:", ["Paste text", "Upload audio"], horizontal=True)

input_text = ""
if input_mode == "Paste text":
    input_text = st.text_area(
        "Paste or type your text here (500–5000 words supported)",
        height=250,
        help="You can paste up to 5000 words for richer question generation.",
    )
else:
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file is not None:
        with st.spinner("Transcribing audio…"):
            transcribed = transcribe_audio(audio_file)
        if not transcribed:
            st.warning("Could not transcribe audio or audio was empty.")
        input_text = st.text_area("Transcribed text (editable)", value=transcribed, height=250)

word_count = len(input_text.split()) if input_text else 0
wc_color   = "🟢" if word_count >= 500 else ("🟡" if word_count >= 100 else "🔴")
st.caption(f"{wc_color} Word count: **{word_count}** / 5000 recommended")

if word_count > 5000:
    st.warning("⚠️ Text exceeds 5000 words. Only the first 5000 words will be processed.")

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    process = st.button("🚀 Generate Questions", use_container_width=True, type="primary")
with col_btn2:
    clear_btn = st.button("🗑️ Clear Results", use_container_width=False)

if clear_btn:
    for key in ("generated_questions", "translated_text", "detected_lang",
                "translated_audio", "input_audio", "all_questions_audio",
                "bulk_input_audio", "bulk_q_audio", "edited_questions"):
        st.session_state[key] = None
    for k in list(st.session_state.keys()):
        if k.startswith("q_audio_"):
            st.session_state[k] = None
    st.rerun()

# ── Main processing ─────────────────────────────────────────────────────────────
if process:
    if not input_text:
        st.error("Please provide some text or upload audio.")
    elif word_count < 50:
        st.error("Text is too short. Please provide at least 50 words.")
    elif word_count < 500 and not allow_short:
        st.error("Please provide at least 500 words (or enable 'Allow less than 500 words' in Settings).")
    else:
        if word_count > 5000:
            input_text = " ".join(input_text.split()[:5000])

        with st.spinner("Detecting language and translating if needed…"):
            translated, detected = translate_to_english(input_text)

        with st.spinner(f"Generating questions… (target: {max_questions}, types: {', '.join(selected_types)})"):
            questions = generate_questions(
                translated,
                max_q=max_questions,
                selected_type_keys=selected_types,
                difficulty_filter=difficulty,
            )

        st.session_state["generated_questions"] = questions
        st.session_state["translated_text"]     = translated
        st.session_state["detected_lang"]       = detected
        st.session_state["edited_questions"]    = [q["question"] for q in questions]
        # Clear stale per-question audio when regenerating
        for k in list(st.session_state.keys()):
            if k.startswith("q_audio_"):
                st.session_state[k] = None

# ── Results rendering ──────────────────────────────────────────────────────────
if st.session_state.get("generated_questions") is not None:
    questions  = st.session_state["generated_questions"]
    translated = st.session_state["translated_text"]
    detected   = st.session_state["detected_lang"]

    st.write("---")

    # ── Language / Translation section ──
    if detected != "en":
        st.success(f"Detected language: **{detected.upper()}** — translated to English for analysis.")
        with st.expander("🌐 Show Translated Text (English)"):
            st.write(translated)
        col1, _ = st.columns([1, 3])
        with col1:
            if st.button("🔊 Play translated text", key="play_translated_btn"):
                ok = store_audio("translated_audio", translated, lang=input_tts_lang, slow=tts_slow_mode)
                if not ok:
                    st.warning("TTS generation failed.")
        render_audio("translated_audio", "translated text")
    else:
        st.success("Language: **English** — no translation needed.")
        with st.expander("📄 Show Input Text"):
            st.write(translated)
        col1, _ = st.columns([1, 3])
        with col1:
            if st.button("🔊 Play input text", key="play_input_btn"):
                ok = store_audio("input_audio", translated, lang=input_tts_lang, slow=tts_slow_mode)
                if not ok:
                    st.warning("TTS generation failed.")
        render_audio("input_audio", "input text")

    # ── Questions ──
    st.header("❓ Generated Questions")

    if not questions:
        st.warning("No questions generated. Try longer/clearer text, different type filters, or a different model.")
    else:
        st.success(f"✅ Generated **{len(questions)}** questions!")

        type_counts = Counter(q["type"] for q in questions)
        diff_counts = Counter(q["difficulty"] for q in questions)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Questions", len(questions))
        with col_s2:
            top_type = type_counts.most_common(1)[0][0].upper() if type_counts else "—"
            st.metric("Most Common Type", top_type)
        with col_s3:
            top_diff = diff_counts.most_common(1)[0][0] if diff_counts else "—"
            st.metric("Most Common Difficulty", top_diff)

        with st.expander("📊 Question Type Breakdown"):
            cols_bar = st.columns(max(len(type_counts), 1))
            for i, (qtype, count) in enumerate(sorted(type_counts.items())):
                with cols_bar[i % len(cols_bar)]:
                    st.metric(TYPE_LABEL.get(qtype, qtype.upper()), count)

        st.divider()

        # ── Bulk TTS buttons ──
        col_play1, col_play2 = st.columns(2)
        with col_play1:
            if st.button("🔊 Play all questions", key="play_all_qs_btn"):
                all_q_text = " ".join(
                    [f"Question {i}. {q['question']}" for i, q in enumerate(questions, 1)]
                )
                ok = store_audio("all_questions_audio", all_q_text, lang=output_tts_lang, slow=tts_slow_mode)
                if not ok:
                    st.warning("TTS generation failed for all questions.")
        with col_play2:
            if st.button("🔊 Play input text (bulk TTS)", key="bulk_input_tts_btn"):
                ok = store_audio("bulk_input_audio", translated, lang=input_tts_lang, slow=tts_slow_mode)
                if not ok:
                    st.warning("Bulk TTS failed for input text.")

        render_audio("all_questions_audio", "all questions")
        render_audio("bulk_input_audio", "bulk input")

        st.divider()

        # ── Question editor toggle ──
        edit_mode = st.checkbox("✏️ Enable question editing", value=False)

        # Ensure edited_questions list is in sync
        edited = st.session_state.get("edited_questions") or [q["question"] for q in questions]
        if len(edited) != len(questions):
            edited = [q["question"] for q in questions]
            st.session_state["edited_questions"] = edited

        # ── Filter display by type tabs ──
        unique_types = list(dict.fromkeys(q["type"] for q in questions))
        tab_labels   = ["All"] + [TYPE_LABEL.get(t, t.upper()) for t in unique_types]
        tabs         = st.tabs(tab_labels)

        # ── render_question_list: defined at module level scope here ──────────
        def render_question_list(q_indices, prefix=""):
            """Render a list of questions by index with TTS + edit support."""
            for idx in q_indices:
                item     = questions[idx]
                ss_key   = f"{prefix}_q_audio_{idx}"
                play_key = f"{prefix}_play_q_{idx}_btn"
                edit_key = f"{prefix}_edit_q_{idx}"

                if ss_key not in st.session_state:
                    st.session_state[ss_key] = None

                diff_badge = DIFF_BADGE.get(item["difficulty"], "⚪")
                type_tag   = TYPE_LABEL.get(item["type"], item["type"].upper())

                cols = st.columns([0.05, 0.70, 0.10, 0.15])
                with cols[0]:
                    st.markdown(f"**{idx + 1}.**")
                with cols[1]:
                    if edit_mode:
                        new_val = st.text_input(
                            label=f"Q{idx + 1}",
                            value=edited[idx],
                            key=edit_key,
                            label_visibility="collapsed",
                        )
                        edited[idx] = new_val
                        st.session_state["edited_questions"] = edited
                    else:
                        st.markdown(
                            f"{edited[idx]}  "
                            f"`{type_tag}` {diff_badge} *{item['difficulty']}*"
                        )
                with cols[2]:
                    if st.button("🔊", key=play_key):
                        q_text = edited[idx]
                        ok = store_audio(ss_key, q_text, lang=output_tts_lang, slow=tts_slow_mode)
                        if not ok:
                            st.warning(f"TTS failed for question {idx + 1}.")
                with cols[3]:
                    render_audio(ss_key)

        # ── All tab ──
        with tabs[0]:
            render_question_list(list(range(len(questions))), prefix="all")

        # ── Per-type tabs ──
        for ti, qtype in enumerate(unique_types):
            with tabs[ti + 1]:
                indices = [i for i, q in enumerate(questions) if q["type"] == qtype]
                render_question_list(indices, prefix=qtype)

        st.divider()

        # ── Download section ──
        st.subheader("📥 Download")

        final_qs = edited if edit_mode else [q["question"] for q in questions]

        col1, col2, col3 = st.columns(3)

        txt_lines = [
            f"{i + 1}. [{TYPE_LABEL.get(questions[i]['type'], '?')}] [{questions[i]['difficulty']}] {final_qs[i]}"
            for i in range(len(questions))
        ]
        txt_content = "\n".join(txt_lines)

        with col1:
            st.download_button(
                "⬇️ Download TXT",
                data=txt_content,
                file_name="questions.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col2:
            try:
                enriched = [
                    {"question": final_qs[i], "type": questions[i]["type"],
                     "difficulty": questions[i]["difficulty"]}
                    for i in range(len(questions))
                ]
                pdf_bytes = build_pdf(enriched, translated, detected, difficulty, selected_types)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="questions.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}. Try TXT instead.")

        with col3:
            type_filter_export = st.selectbox(
                "Export by type",
                ["All"] + [TYPE_LABEL.get(t, t) for t in unique_types],
                key="export_type_select",
            )
            if type_filter_export == "All":
                export_data = txt_content
                fname = "questions_all.txt"
            else:
                reverse_map = {v: k for k, v in TYPE_LABEL.items()}
                et = reverse_map.get(type_filter_export, "")
                filtered_lines = [
                    f"{j + 1}. {final_qs[i]}"
                    for j, i in enumerate(
                        i for i, q in enumerate(questions) if q["type"] == et
                    )
                ]
                export_data = "\n".join(filtered_lines)
                fname = f"questions_{type_filter_export.lower()}.txt"
            st.download_button(
                f"⬇️ Export {type_filter_export}",
                data=export_data,
                file_name=fname,
                mime="text/plain",
                use_container_width=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "💡 **Tips:** "
    "• Provide well-formed factual paragraphs (500–5000 words) for best results. "
    "• Use **Question Type Filters** in the sidebar to focus on specific question styles. "
    "• Set **TTS language for input text** and **questions** separately in the sidebar. "
    "• Enable **Edit mode** to fix questions before downloading. "
    "• Audio buttons no longer clear your generated questions — safe to click anytime!"
)
