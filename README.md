# sklearn_wine_clustering
Use clustering algorithms to evaluate performance of trained dataset.

# Aim

Learn   to   select   the   parameters   of   the   clustering   algorithms,   visualize   clusters   usingdimensionality reduction method. 

# Task

1)Select sklearn wine dataset. Split to train and test subsets.
2)Train PCA model on train subset, reducing number of coordinates to 2.
3)Choose the number of clusters the same how many you have unique class labels.  Perform K-Means clustering using train subset.  Calculate a table on test subset how many samples ofdifferent classes go to every cluster. 
4)Visualize clusters with different color points using trained PCA model.
5)Repeat step 3) and 4) with bigger number of clusters than unique class labels.
6)Perform Agglomerative clustering on test subset, Set n_clusters=None and try different distancethreshold values, that to   obtain resulting number of clusters close to unique class labels.Calculate a cluster composition table. Search for threshold and linkage, that to achieve betterclass splitting to clusters. Visualize clusters using PCA.
