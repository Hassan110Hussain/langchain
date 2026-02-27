from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.01,
)

sample = """Cricket has always been more than just a sport in many countries; it is a celebration of skill, teamwork, and national pride. From the roar of the crowd in stadiums to the quiet determination of young players practicing in dusty fields, cricket connects communities across generations. Beyond the excitement of matches and tournaments, it also inspires discipline, perseverance, and strategic thinking. Many young athletes pursue cricket as a profession, dedicating countless hours to training, fitness, and strategy. Terrorism, on the other hand, poses a grave threat to societies worldwide, disrupting peace and instilling fear in innocent populations. It encompasses acts of violence intended to intimidate governments or communities, often driven by political, religious, or ideological motives. The consequences of terrorism are far-reaching, affecting economic stability, social cohesion, and international relations. Countering terrorism requires coordinated efforts across nations, including intelligence sharing, community engagement, and effective law enforcement, to prevent radicalization and maintain global security.

Farming is the backbone of many economies and a vital source of food and raw materials for societies around the world. It involves cultivating crops and raising livestock, requiring knowledge, skill, and careful planning to ensure sustainable yields. Modern farming combines traditional practices with advanced technology, such as irrigation systems, machinery, and data-driven crop management, to maximize productivity. Beyond providing sustenance, farming supports rural communities, creates employment, and preserves cultural heritage tied to the land. Despite challenges like climate change, pests, and fluctuating market prices, farmers continue to play a crucial role in feeding populations and sustaining livelihoods globally.
"""

docs = text_splitter.create_documents([sample])
print(docs)
print(len(docs))
