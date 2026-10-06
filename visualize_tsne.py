import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# โหลด embeddings และชื่อไฟล์ที่เคยบันทึกไว้
embeddings = np.load("clip_embeddings.npy")
filenames  = np.load("clip_filenames.npy", allow_pickle=True)


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(10,6))
plt.scatter(embeddings_2d_tsne[:,0], embeddings_2d_tsne[:,1], s=8, alpha=0.7)
plt.title("t-SNE Visualization of Image Embeddings (CLIP)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
