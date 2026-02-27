from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """The su nn moon rises in fragments of gold, painting the sky with whispers of tomorrow.

Beneath the su nn moon, silence blooms, carrying secrets through the midnight breeze.

Every su nn moon night feels like a dream folded between shadows and light.

The su nn moon is not the sun, nor the moon, but a bridge where both rest together.

In the glow of the su nn moon, even time forgets to move forward."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))
