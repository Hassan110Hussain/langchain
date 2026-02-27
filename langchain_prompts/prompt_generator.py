from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""You are an expert researcher and teacher. 
Your task is to explain the research paper "{selected_paper}" in the following way:

- **Style**: {selected_style}
- **Length**: {selected_length}

Make sure the explanation is accurate, clear, and tailored to the chosen style and length.
Avoid adding unrelated information.
    """,
    input_variables=["selected_paper", "selected_style", "selected_length"],
    validate_template=True
)

template.save('template.json')