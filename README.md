# Dimensionality-Reduction-using-t-SNE-and-PCA

In this project, you will perform dimensionality reduction and visualize a subset of
MNIST: 784-dimensional feature vectors in mnist2500 X.txt and their corresponding labels in
mnist2500 labels.txt.

Implement t-Distributed Stochastic Neighbor Embedding (t-SNE), a (prize-winning) technique for dimensionality reduction that is particularly well suited for the
visualization of high-dimensional datasets. t-SNE maps high-dimensional data points x1, x2, . . . , xN
into two or three-dimensional embeddings y1, y2, . . . , yN that can be displayed in a scatter plot.
Unlike PCA that you learned in class, t-SNE is a non-linear dimensionality reduction technique.
It is widely used (more than 3,300 citations and counting). If you are curious about how to use it
effectively, check this out https://distill.pub/2016/misread-tsne/.
Intuitively, we want the low-dimensional data points to reflect similarities of their corresponding
data points in the high-dimensional space. In other words, after applying the mapping, similar data points are still near each other and dissimilar data points are still far apart. In t-SNE, this is
achieved in two steps. First, t-SNE constructs a joint probability distribution P over pairs of highdimensional
data points in such a way that similar points have a high probability of being picked,
whilst dissimilar points have an extremely small probability. Second, t-SNE similarly defines a
joint probability distribution Q over pairs of low-dimensional data points and then minimizes the
Kullback-Leibler (KL) divergence between P and Q. You could think of KL-divergence as a measure
of how one probability distribution diverges from another.


We first apply PCA to our very high-dimensional data points. This in practice helps speed
up the computation and reduce noise in the data points before we apply t-SNE. After implementating the function pca, run the script pca.sh. It will output a file pca.npy. 


In tsne.py, you are given the code to compute pij , and you need to finish the code that computes qij in function compute_Q.
Compute the gradient ∂C/∂yi in tsne.py. Finally, running the script tsne.sh will show cool t-SNE visualization the data in the 2-dimensional space and also output Y.npy.

