from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="social_network_users.csv")

docs = loader.load()

print(len(docs))
