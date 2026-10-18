"""A small Streamlit interface for the trained SMS spam classifier.

Run `python train_model.py` first, then start this app with `streamlit run app.py`.
"""

from pathlib import Path

import joblib
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "spam_classifier.joblib"

st.set_page_config(page_title="SMS Spam Checker", page_icon="🛡️", layout="centered")
st.title("🛡️ SMS Spam Checker")
st.write("Paste a text message below to see whether the model thinks it is spam.")
st.caption("A learning demo using the UCI SMS Spam Collection. Predictions can be wrong.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.stop()


@st.cache_resource
def load_model():
    """Load the saved TF-IDF and classifier pipeline once per app session."""
    return joblib.load(MODEL_PATH)


model = load_model()
message = st.text_area(
    "Your message",
    placeholder="Example: Congratulations! You won a free iPhone. Claim now!",
    height=140,
    max_chars=5000,
)

if st.button("Check message", type="primary", use_container_width=True):
    if not message.strip():
        st.info("Enter a message first, then select Check message.")
    else:
        label = model.predict([message])[0]
        spam_probability = float(model.predict_proba([message])[0][list(model.classes_).index("spam")])
        if label == "spam":
            st.error("Prediction: likely spam")
        else:
            st.success("Prediction: likely not spam")
        st.write(f"Spam score: **{spam_probability:.1%}**")
        st.progress(spam_probability)

st.divider()
st.caption("The spam score is the model's estimated probability, not a guarantee.")
