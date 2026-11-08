"""Sign language to text and translation: image upload, chat-style interface."""
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Make src/ importable when running from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference.predictor import SignPredictor
from src.translation.translator import Translator, LANGUAGES

MODEL_PATH = ROOT / "models" / "resnet50_sign_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading sign classifier...")
def load_predictor():
    return SignPredictor(MODEL_PATH, device=DEVICE, autocontrast=True)


@st.cache_resource(show_spinner="Loading NLLB-200")
def load_translator():
    return Translator(device=str(DEVICE))


st.set_page_config(page_title="Sign Language Translator", page_icon="🤟")
st.title("Sign Language to Text and Translation")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    chosen = st.multiselect(
        "Translate to", list(LANGUAGES.keys()), default=["French", "Hindi", "Tamil"]
    )
    threshold = st.slider("Warn below confidence", 0.0, 1.0, 0.80, 0.05)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload one or more images; each image is one letter, in order
uploads = st.file_uploader(
    "Upload one or more sign images (in order, each image is one letter)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploads and st.button("Recognize and translate", type="primary"):
    predictor = load_predictor()
    translator = load_translator()

    images = [Image.open(f) for f in uploads]
    preds = predictor.predict(images)
    word = "".join(p["letter"] for p in preds)

    targets = [LANGUAGES[name] for name in chosen]
    translations = translations = translator.translate(word, targets) if targets and len(word) >= 2 else {}

    # User turn
    st.session_state.messages.append({
        "role": "user",
        "content": f"Uploaded {len(images)} image(s): **{word}**",
    })

    # Assistant turn
    lines = ["**Predicted letters**"]
    for i, p in enumerate(preds, 1):
        flag = "" if p["confidence"] >= threshold else " ⚠️ low confidence"
        alternatives = ", ".join(f"{l} ({c:.0%})" for l, c in p["top_k"][1:])
        lines.append(f"{i}. **{p['letter']}**  {p['confidence']:.1%}{flag}  (also: {alternatives})")

    lines.append(f"\n**Text:** {word}")
    if len(images) == 1:
        lines.append("_Translation is most meaningful for words. Upload several letters to translate a word._")

    if translations:
        lines.append("\n**Translations**")
        for code, text in translations.items():
            name = next(k for k, v in LANGUAGES.items() if v == code)
            lines.append(f"- {name}: {text}")

    st.session_state.messages.append({"role": "assistant", "content": "\n".join(lines)})
    st.rerun()