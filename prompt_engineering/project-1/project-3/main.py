import streamlit as st
from transformers import pipeline


generator = pipeline("text-generation", model="distilgpt2")


faqs = {
    "What are your business hours?": "We are open from 9 AM to 5 PM, Monday through Friday.",
    "Do you offer free shipping?": "Yes, we offer free shipping on orders over $50.",
    "What is your return policy?": "You can return items within 30 days of purchase for a full refund.",
    "How can I contact customer support?": "You can reach us at support@example.com or call 1-800-123-4567."
}

st.title("FAQ Assistant")

user_input = st.text_input("Ask a question:")


if st.button("Get Answer"):
    
    if user_input in faqs:
        answer = faqs[user_input]
    else:
        response = generator(user_input, max_length=50, num_return_sequences=1)
        answer = response[0]['generated_text']
    st.text_area("Answer:", answer, height=100)
