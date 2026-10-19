from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")
print(llm.invoke("Say hello in one sentence.").content)

emb = OllamaEmbeddings(model="nomic-embed-text")
vec = emb.embed_query("cat activity levels")
print(len(vec))  # should print 768