import io
import re
import time

import pytesseract
import streamlit as st
from docx import Document as DocxDocument
from PIL import Image
from pypdf import PdfReader

from app.client import send_request
from app.logger import log_request, log_retraining_candidate
from app.router import detect_provider, get_model_config, predict_tier
from app.verifier import verify_and_escalate

st.set_page_config(page_title="Chat — LLM Cost Autopilot", layout="wide")

TIER_COLORS = {"simple": "#38bdf8", "moderate": "#8b5cf6", "complex": "#ec4899"}
MAX_DOC_CHARS = 200000
CHUNK_SIZE = 10000
REDUCE_CHAR_LIMIT = 6000
REDUCE_BATCH_SIZE = 4

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; max-width: 900px;}
    h1 {color: #f8fafc; font-weight: 800;}
    [data-testid="stChatMessage"] {
        background: #141928;
        border: 1px solid #242a3d;
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 8px;
    }
    .msg-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid #8b5cf6;
        color: #c4b5fd;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-top: 6px;
    }
    .thinking-dots span {
        display: inline-block; width: 6px; height: 6px; margin: 0 2px;
        background: #8b5cf6; border-radius: 50%;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .thinking-dots span:nth-child(2) {animation-delay: 0.2s;}
    .thinking-dots span:nth-child(3) {animation-delay: 0.4s;}
    @keyframes bounce {
        0%, 80%, 100% {transform: scale(0.6); opacity: 0.5;}
        40% {transform: scale(1); opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def thinking_html(label: str) -> str:
    return (
        f'<span style="color:#94a3b8;">{label}</span> '
        f'<span class="thinking-dots"><span></span><span></span><span></span></span>'
    )


def stream_words(text: str, delay: float = 0.018):
    """Yields word by word so st.write_stream can reveal the reply the way
    a live model stream would, even though our backend returns the full
    response at once."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def build_context_prompt(history: list, latest_prompt: str, max_turns: int = 6) -> str:
    """Folds recent conversation turns into a single prompt string, since
    send_request() takes a plain string, not a message list. Classification
    still runs on latest_prompt alone (below) so a short follow-up doesn't
    get inflated into 'complex' just because the transcript is long."""
    if not history:
        return latest_prompt

    recent = history[-max_turns * 2:]
    lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent]
    lines.append(f"User: {latest_prompt}")
    transcript = "\n".join(lines)

    return (
        "The following is a conversation. Continue it naturally by responding "
        "only to the final user message, using the earlier turns as context.\n\n"
        f"{transcript}\n\nAssistant:"
    )


def clean_extracted_text(text: str) -> str:
    """Collapses the excess blank lines and repeated whitespace that PDF/OCR
    extraction tends to leave behind -- pure token waste, no informational
    value, and it directly inflates cost since it's all billed as input
    tokens."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    elif name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(image)
    else:
        raise ValueError(f"Unsupported file type: {name}")

    return clean_extracted_text(text)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def combine_summaries_recursively(summaries: list, doc_name: str, model, status, level: int = 1):
    """Recursively combines a list of summaries into fewer, larger ones
    until the combined text is short enough to safely fit in one model
    call. This is the fix for the bug where a flat concatenation of all
    chunk-summaries was too long for the model's context window --
    whatever got silently truncated meant only the last few parts ever
    reached the model. Combining in small batches first, repeatedly if
    needed, keeps every level's input safely within context regardless of
    how many chunks the original document produced."""
    combined = "\n\n".join(summaries)
    if len(combined) <= REDUCE_CHAR_LIMIT or len(summaries) <= 1:
        return combined, 0.0, 0.0

    batches = [summaries[i:i + REDUCE_BATCH_SIZE] for i in range(0, len(summaries), REDUCE_BATCH_SIZE)]
    next_level = []
    total_cost = 0.0
    total_latency = 0.0

    for i, batch in enumerate(batches, start=1):
        status.markdown(
            thinking_html(f"Combining summaries — level {level}, batch {i} of {len(batches)}"),
            unsafe_allow_html=True,
        )
        batch_text = "\n\n".join(batch)
        combine_prompt = (
            f"Combine the following summaries from a document called '{doc_name}' into "
            "one shorter summary of at most 150 words that preserves every distinct "
            "fact, name, and number mentioned across all of them. Do not omit any "
            "topic or section.\n\n"
            f"{batch_text}"
        )
        response = send_request(combine_prompt, model)
        next_level.append(response.output_text)
        total_cost += response.cost
        total_latency += response.latency

    deeper_text, deeper_cost, deeper_latency = combine_summaries_recursively(
        next_level, doc_name, model, status, level + 1
    )
    return deeper_text, total_cost + deeper_cost, total_latency + deeper_latency


def summarize_document_map_reduce(doc_name: str, doc_text: str, user_question: str, provider: str, status):
    """Map-reduce over a long document instead of stuffing the whole thing
    into one context window. Each chunk is summarized independently on the
    cheap 'simple' tier (map). Those chunk-summaries are then combined
    hierarchically (see combine_summaries_recursively) until short enough
    for one final synthesis call on 'moderate' (reduce), alongside the
    user's actual question."""
    chunks = split_into_chunks(doc_text)
    map_model = get_model_config(provider, "simple")
    reduce_model = get_model_config(provider, "moderate")

    chunk_summaries = []
    map_cost = 0.0
    map_latency = 0.0

    for i, chunk in enumerate(chunks, start=1):
        status.markdown(thinking_html(f"Summarizing document part {i} of {len(chunks)}"), unsafe_allow_html=True)
        map_prompt = (
            f"This is part {i} of {len(chunks)} of a document called '{doc_name}'. "
            "Summarize the key points of this excerpt in at most 100 words, "
            "preserving specific names, numbers, and facts. Do not add commentary.\n\n"
            f"{chunk}"
        )
        chunk_response = send_request(map_prompt, map_model)
        chunk_summaries.append(f"[Part {i} of {len(chunks)}]: {chunk_response.output_text}")
        map_cost += chunk_response.cost
        map_latency += chunk_response.latency

    combined_summary, reduce_combine_cost, reduce_combine_latency = combine_summaries_recursively(
        chunk_summaries, doc_name, reduce_model, status
    )

    reduce_prompt = (
        f"Below is a summary covering the entire document '{doc_name}', from start "
        "to finish. Using it, answer the user's request as completely as possible.\n\n"
        f"{combined_summary}\n\n"
        f"User's request: {user_question}"
    )
    total_cost = map_cost + reduce_combine_cost
    total_latency = map_latency + reduce_combine_latency
    return reduce_prompt, reduce_model, total_cost, total_latency


st.title("Chat")
st.caption(
    "Every message is classified by complexity, routed to the cheapest capable model, "
    "and scored by an AI judge for quality -- the same pipeline that handles production traffic."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "attached_doc" not in st.session_state:
    st.session_state.attached_doc = None
if "attached_doc_name" not in st.session_state:
    st.session_state.attached_doc_name = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "processing_prompt" not in st.session_state:
    st.session_state.processing_prompt = None

with st.sidebar:
    st.subheader("Session stats")
    total_cost = sum(m.get("cost", 0) for m in st.session_state.messages if m["role"] == "assistant")
    st.metric("Messages sent", sum(1 for m in st.session_state.messages if m["role"] == "user"))
    st.metric("Session cost", f"${total_cost:.6f}")
    if st.button("Clear chat", use_container_width=True):
        st.divider()
        st.caption("© 2026 Mohit Kewalramani. All rights reserved.")
        st.session_state.messages = []
        st.session_state.processing_prompt = None
        st.rerun()

uploaded_file = st.file_uploader(
    "Attach a document (PDF, Word, text, or image) to ask questions about it",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded_file is not None and st.session_state.attached_doc_name != uploaded_file.name:
    with st.spinner("Extracting text..."):
        try:
            raw_text = extract_text_from_upload(uploaded_file)
        except Exception as e:
            st.error(f"Could not extract text from this file: {e}")
            raw_text = ""

    if raw_text:
        truncated = len(raw_text) > MAX_DOC_CHARS
        st.session_state.attached_doc = {
            "name": uploaded_file.name,
            "text": raw_text[:MAX_DOC_CHARS],
            "truncated": truncated,
            "char_count": len(raw_text),
        }
        st.session_state.attached_doc_name = uploaded_file.name

if st.session_state.attached_doc:
    doc = st.session_state.attached_doc
    n_chunks = -(-len(doc["text"]) // CHUNK_SIZE)  # ceil division
    label = f"Attached: {doc['name']} ({doc['char_count']:,} characters, ~{n_chunks} chunks"
    label += ", truncated to fit)" if doc["truncated"] else ")"
    col_a, col_b = st.columns([5, 1])
    with col_a:
        st.markdown(f'<span class="msg-badge">{label}</span>', unsafe_allow_html=True)
    with col_b:
        if st.button("Remove attachment"):
            st.session_state.attached_doc = None
            st.session_state.attached_doc_name = None
            st.session_state.uploader_key += 1  # forces a fresh, empty file_uploader widget
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(f'<span class="msg-badge">{msg["meta"]}</span>', unsafe_allow_html=True)

prompt = st.chat_input("Message the router...")

if prompt and prompt == st.session_state.processing_prompt:
    st.warning(
        "This message looks like it was already submitted (likely from navigating away "
        "mid-processing and back). Click 'Clear chat' if you want to resend it, or type "
        "a new message."
    )
elif prompt:
    st.session_state.processing_prompt = prompt
    history_before = list(st.session_state.messages)
    attached_doc = st.session_state.attached_doc

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()

        if attached_doc:
            tier = "moderate"
            provider = detect_provider(tier)
            contextual_prompt, routed_model, map_cost, map_latency = summarize_document_map_reduce(
                attached_doc["name"], attached_doc["text"], prompt, provider, status
            )
            status.markdown(thinking_html("Synthesizing final answer"), unsafe_allow_html=True)
            response = send_request(contextual_prompt, routed_model)
            # Fold in the map/combine-phase cost/latency -- that's real
            # money spent processing the document, and should count toward
            # what this request actually cost, not just the final call.
            response.cost += map_cost
            response.latency += map_latency
        else:
            status.markdown(thinking_html("Classifying prompt complexity"), unsafe_allow_html=True)
            tier = predict_tier(prompt)
            provider = detect_provider(tier)
            routed_model = get_model_config(provider, tier)
            contextual_prompt = build_context_prompt(history_before, prompt)
            status.markdown(
                thinking_html(f"Routed to <code>{routed_model.model_id}</code> ({tier} tier) — generating"),
                unsafe_allow_html=True,
            )
            response = send_request(contextual_prompt, routed_model)

        status.markdown(thinking_html("Scoring response quality"), unsafe_allow_html=True)
        verification = verify_and_escalate(contextual_prompt, response, provider, tier, force=True)
        log_request(prompt, provider, tier, routed_model, response, verification)
        if verification.escalated:
            log_retraining_candidate(prompt, tier_corrected="complex")

        status.empty()

        final_text = verification.final_response.output_text
        final_cost = response.cost + verification.cost_delta

        st.write_stream(stream_words(final_text))

        badge = f"Tier: {tier}"
        if attached_doc:
            badge += " (forced — document attached, map-reduce)"
        badge += f" \u2192 {routed_model.model_id} · Cost: ${final_cost:.6f} · Score: {verification.score}/5"
        if verification.escalated:
            top_model = get_model_config(provider, "complex").model_id
            badge += f" · Escalated to {top_model} (+${verification.cost_delta:.6f})"
        st.markdown(f'<span class="msg-badge">{badge}</span>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_text,
        "meta": badge,
        "cost": final_cost,
    })

    st.session_state.processing_prompt = None
    st.rerun()

