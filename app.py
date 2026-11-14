
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("Next-word-Prediction-lstm.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Reverse word index
reverse_index = {
    idx: word for word, idx in tokenizer.word_index.items()
}

max_len = 44

def generate_text(seed_text, num_words=10):
    text = seed_text

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]

        padded = pad_sequences(
            [seq],
            maxlen=max_len,
            padding='pre'
        )

        pred = model.predict(padded, verbose=0)

        pos = np.argmax(pred)

        next_word = reverse_index.get(pos, "")

        if not next_word:
            break

        text += " " + next_word

    return text


# Streamlit UI
st.title("Next Word Prediction")

seed = st.text_input("Enter The Starting Text:", "Hello")

num_words = st.slider(
    "Number of words to generate",
    1, 20, 10
)

if st.button("Generate"):
    generated_text = generate_text(seed, num_words)

    st.subheader("Generated Text:")
    st.write(generated_text)
