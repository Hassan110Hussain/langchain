from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

st.header("Research Tool")

selected_paper = st.selectbox(
    "Select Research Paper",
    [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
    ],
)
selected_style = st.selectbox(
    "Select Explanation Style",
    ["Simple / Beginner-friendly", "Technical / Detailed", "Summary / Concise"],
)
selected_length = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (full detailed explanation)",
    ],
)

template = load_prompt("template.json")

if st.button("summarize"):
    chain = template | model
    result = chain.invoke(
        {
            "selected_paper": selected_paper,
            "selected_style": selected_style,
            "selected_length": selected_length,
        }
    )
    st.write(result.content)
