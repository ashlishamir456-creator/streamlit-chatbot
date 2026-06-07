import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# =========================
# DEVICE (CPU ONLY = SAFE)
# =========================
device = torch.device("cpu")

# =========================
# LOAD MODEL (SAFE METHOD)
# =========================
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

t5_model = T5ForConditionalGeneration.from_pretrained(
    "t5-small"
)
t5_model = t5_model.eval()   # IMPORTANT: no .to(device)

# =========================
# CHAT FUNCTION
# =========================
def chat(text):
    text = text.lower().strip()

    # 🔥 RULE-BASED RESPONSES (FAST)
    if any(word in text for word in ["hi", "hello", "hey"]):
        return "Hello! How can I help you?"

    if "how are you" in text:
        return "I'm fine 😊 What about you?"

    if any(word in text for word in ["bye", "goodbye"]):
        return "Goodbye! Have a nice day 👋"

    if "thank" in text:
        return "You're welcome!"

    # 🤖 T5 FALLBACK RESPONSE
    try:
        input_text = "chatbot: " + text

        inputs = t5_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        outputs = t5_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )

        reply = t5_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return reply if reply else "I didn't understand that."

    except Exception as e:
        return "Sorry, something went wrong."

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot (Fixed Version)")

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        response = chat(user_input)
        st.write("🤖 Bot:", response)