from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


text = ("Artificial Intelligence (AI) is transforming various industries by automating tasks "
        "and providing insights from data. Machine Learning, a subset of AI, involves training "
        "models on data to make predictions or decisions without being explicitly programmed. "
        "Natural Language Processing (NLP) is a branch of AI that focuses on enabling computers "
        "to understand, interpret, and generate human language.")


tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.detach().numpy()

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings[0])

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
plt.title("2D PCA of BERT Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
