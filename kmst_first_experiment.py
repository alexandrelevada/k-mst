"""

Curvature-aware k-MST clustering for nonlinear data

First set of experiments

Author: Alexandre L. M. Levada

"""
import time
import warnings
import matplotlib as mpl
import numpy as np
import scipy
import umap
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.datasets as skdata
import sklearn.utils.graph as sksp
import statistics as stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import xgboost as xgb
from sklearn import preprocessing
from sklearn import metrics
from numpy import inf
from scipy import optimize
from scipy.signal import medfilt
from networkx.convert_matrix import from_numpy_array
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.covariance import LedoitWolf


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Build the KNN graph
def build_KNN_Graph(dados, k):
    n = dados.shape[0]
    # Build a graph
    CompleteGraph = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance')
    # Adjacency matrix
    W_K = CompleteGraph.toarray()
    # NetworkX format
    K_n = nx.from_numpy_array(W_K)
    # MST
    W_mst = nx.minimum_spanning_tree(K_n)
    mst = [(u, v, d) for (u, v, d) in W_mst.edges(data=True)]
    mst_edges = []
    for edge in mst:
        edge_tuple = (edge[0], edge[1], edge[2]['weight'])
        mst_edges.append(edge_tuple)
    # Create the k-NNG
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Adjacency matrix
    W = knnGraph.toarray()
    # NetworkX format
    G = nx.from_numpy_array(W)
    # To assure the k-NNG is connected we add te MST edges
    G.add_weighted_edges_from(mst_edges)
    # Convert to adjacency matrix
    return G

# Computes the curvatures of all samples in the training set
def Curvature_Estimation(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    if m > 50:
        m = 25
        dados = PCA(n_components=m).fit_transform(dados)
    # First and second fundamental forms
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto').fit(dados)
    distances, indices = nbrs.kneighbors(dados)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        ######## First form
        amostras = dados[indices[i]]
        ni = len(indices)
        if ni > 1:
            # Local first form 
            I = np.cov(amostras.T)
            I = I + 0.0001*np.eye(I.shape[0])
            # covariance = LedoitWolf().fit(amostras)
            # I = covariance.covariance_
        else:
            I = np.eye(m)
        # Compute the eigenvectors
        v, w = np.linalg.eig(I)
        # Sort the eigenvalues
        ordem = v.argsort()
        # Select the eigenvectors in decreasing order (in columns)
        Wpca = w[:, ordem[::-1]]
        # Second form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))        
        # Discard the first m columns of H
        H = Wpca[:, (m+1):]        
        # Second form
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        curvatures[i] = np.linalg.det(S)    # Gaussian curvature
        #curvatures[i] = np.trace(S)        # Mean curvature
    return curvatures

# Normalize array to interval [0, 1]
def normalize(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return norm_arr


# MST based clustering: divisive approach (Calinski-Harabasz)
def mst_clustering_divisive_CH(T, dados, rotulos):
    #Number of samples
    n = len(rotulos)
    # Number of classes
    c = len(np.unique(rotulos))
    # Number of edges to be removed
    K = c - 1
    # Number of edges in the MST
    n_edges = len(T.edges())
    # Main loop
    T_ = T.copy()
    for i in range(K):
        arestas = []
        medidas = []
        # Find edge whose removal maximizes WCSS
        for (u, v) in T_.edges():
            arestas.append((u, v))
            labels = np.zeros(n)
            T_.remove_edge(u, v)
            clusters = nx.connected_components(T_)
            code = 0
            arvores = []
            for c in clusters:  # c is a set
                indices = np.array(list(c))                
                labels[indices] = code
                code += 1
            ch = calinski_harabasz_score(dados, labels)
            medidas.append(ch)             
            T_.add_edge(u, v)
        best = np.array(medidas).argmax()
        edge_star = arestas[best]
        T_.remove_edge(*edge_star)
    return T_


# Label samples
def label_samples(T, rotulos):
    # Obtain the connected components from the tree
    n = len(rotulos)
    clusters = nx.connected_components(T)
    labels = np.zeros(n)
    code = 0
    for c in clusters:  # c is a set
        indices = np.array(list(c))
        labels[indices] = code
        code += 1
    return labels


# Plot the KNN graph
def plot_graph(G, target, layout='spring', pos=0):
	color_map = []
	for i in range(n):
		if target[i] == -1:
			color_map.append('black')
		elif target[i] == 0:
			color_map.append('blue')
		elif target[i] == 1:
			color_map.append('red')
		elif target[i] == 2:
			color_map.append('green')
		elif target[i] == 3:
			color_map.append('purple')
		elif target[i] == 4:
			color_map.append('orange')
		elif target[i] == 5:
			color_map.append('magenta')
		elif target[i] == 6:
			color_map.append('darkkhaki')
		elif target[i] == 7:
			color_map.append('brown')
		elif target[i] == 8:
			color_map.append('salmon')
		elif target[i] == 9:
			color_map.append('cyan')
		elif target[i] == 10:
			color_map.append('darkcyan')
	plt.figure(1)
	if np.isscalar(pos):
		if layout == 'spring':
			if n <= 400:
				pos = nx.spring_layout(G, iterations=1000)
			else:
				pos = nx.spring_layout(G, iterations=100)
		else:
			pos = nx.kamada_kawai_layout(G)
	if n < 1000:
		nx.draw_networkx(G, pos, node_size=25, node_color=color_map, with_labels=False, width=0.25, alpha=0.4)
	else:
		nx.draw_networkx(G, pos, node_size=12, node_color=color_map, with_labels=False, width=0.25, alpha=0.3)
	plt.show()
	return pos

##############################################
############# Data loading
##############################################
#X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='energy-efficiency', version=1) 
#X = skdata.fetch_openml(name='spectrometer', version=1)
#X = skdata.fetch_openml(name='car-evaluation', version=1)
X = skdata.fetch_openml(name='thyroid-new', version=1)  
#X = skdata.fetch_openml(name='arrhythmia', version=1)   
#X = skdata.fetch_openml(name='Touch2', version=1)   
#X = skdata.fetch_openml(name='seeds', version=1)
#X = skdata.fetch_openml(name='Satellite', version=1)
#X = skdata.fetch_openml(name='Lung', version=1)
#X = skdata.fetch_openml(name='led24', version=1)
#X = skdata.fetch_openml(name='led7', version=1)
#X = skdata.fetch_openml(name='vowel', version=1)
#X = skdata.fetch_openml(name='soybean', version=1)
#X = skdata.fetch_openml(name='user-knowledge', version=1)
#X = skdata.fetch_openml(name='hayes-roth', version=1)  
#X = skdata.fetch_openml(name='steel-plates-fault', version=3)
#X = skdata.fetch_openml(name='MLL', version=1)
#X = skdata.fetch_openml(name='vote', version=1)     
#X = skdata.fetch_openml(name='zoo', version=1)  
#X = skdata.fetch_openml(name='GLI', version=1)
#X = skdata.fetch_openml(name='SRBCT', version=1)
#X = skdata.fetch_openml(name='AP_Uterus_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Ovary', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Prostate', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)
#X = skdata.fetch_openml(name='letter', version=1)
#X = skdata.fetch_openml(name='page-blocks', version=1)

dados = X['data']
target = X['target']  

if 'details' in X.keys():
	if X['details']['name'] == 'satimage':
 		dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42) 		
	elif X['details']['name'] == 'pendigits':
 		dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)
	elif X['details']['name'] == 'letter':
 		dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42) 		

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))
k = round(np.log2(n))

print('N = ', n)
print('M = ', m)
print('C = ', c)
print('K = ', k)
print()


# Sparse matrix (for some high dimensional datasets)
if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
	# Treat categorical features
	if not isinstance(dados, np.ndarray):
		cat_cols = dados.select_dtypes(['category']).columns
		dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
		dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)
# Computes the local curvatures
curvatures = Curvature_Estimation(dados, k)
# Normalize the curvatures
K = normalize(curvatures)
# Build the KNN graph
kNN = build_KNN_Graph(dados, k)
# Save a copy of the original k-NN graph
G = kNN.copy()

# Compute the regular Euclidean MST
T_mst = nx.minimum_spanning_tree(G)

# Create the k-graph
# Weight the edges with the sum of the curvatures of the end vertices
for u, v, d in G.edges(data=True):
	d['weight'] *= (K[u] + K[v])

# Compute the MST (minimum curvature tree)
T_min = nx.minimum_spanning_tree(G)

# MST-based clustering
t = mst_clustering_divisive_CH(T_mst, dados, target)
labels = label_samples(t, target)

print('Euclidean distance MST-based indices')
print('---------------------------------------')
print('Adjusted Rand index: %f' %adjusted_rand_score(target, labels))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, labels))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, labels))
print('V-measure: %f' %v_measure_score(target, labels))
print()

# k-MST
t = mst_clustering_divisive_CH(T_min, dados, target)
labels = label_samples(t, target)

print('Minimum Information Tree based indices')
print('---------------------------------------')
print('Adjusted Rand index: %f' %adjusted_rand_score(target, labels))
print('Adjusted mutual info score: %f' %adjusted_mutual_info_score(target, labels))
print('Fowlkes Mallows index: %f' %fowlkes_mallows_score(target, labels))
print('V-measure: %f' %v_measure_score(target, labels))
print()

# Plot tree
#plot_graph(T_mst, target, layout='kawai')