from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt

def generate_tsne(path, data, label):
    print '\nGenerating t-SNE...'
    tsne = TSNE(n_jobs=-1)
    Y = tsne.fit_transform(data)
    plt.figure(figsize=(20, 20))
    plt.scatter(Y[:, 0], Y[:, 1], c=label, s=100, cmap='Set1', alpha=0.2)
    plt.colorbar()
    plt.savefig(path)
