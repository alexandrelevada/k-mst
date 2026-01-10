"""

Curvature-aware k-MST clustering for nonlinear data

Second set of experiments

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
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
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
            I = np.cov(amostras.T)
            I = I + 0.0001*np.eye(I.shape[0]) 
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


# MST based clustering: divisive approach (Calinski-Harabasz: WCSS)
def mst_clustering_divisive_CH(T, dados, rotulos):
    n = len(rotulos)
    c = len(np.unique(rotulos))
    K = c - 1
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
X = skdata.load_iris()
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-morphological', version=1)
#X = skdata.fetch_openml(name='mfeat-zernike', version=1)
#X = skdata.fetch_openml(name='energy-efficiency', version=1) 
#X = skdata.fetch_openml(name='spectrometer', version=1)
#X = skdata.fetch_openml(name='arrhythmia', version=1)   
#X = skdata.fetch_openml(name='seeds', version=1)
#X = skdata.fetch_openml(name='Lung', version=1)
#X = skdata.fetch_openml(name='soybean', version=1)
#X = skdata.fetch_openml(name='user-knowledge', version=1)
#X = skdata.fetch_openml(name='steel-plates-fault', version=3)
#X = skdata.fetch_openml(name='zoo', version=1)  
#X = skdata.fetch_openml(name='GLI', version=1)
#X = skdata.fetch_openml(name='SRBCT', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Uterus_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Prostate', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Kidney', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='letter', version=1)
#X = skdata.fetch_openml(name='page-blocks', version=1)
#X = skdata.fetch_openml(name='coil-20', version=1)
#X = skdata.fetch_openml(name='cnae-9', version=1)
#X = skdata.fetch_openml(name='11_Tumors', version=1)
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)    
#X = skdata.fetch_openml(name='tr41.wc', version=1)
#X = skdata.fetch_openml(name='artificial-characters', version=1)
#X = skdata.fetch_openml(name='semeion', version=1)
#X = skdata.fetch_openml(name='texture', version=1)

dados = X['data']
target = X['target']  

if 'details' in X.keys():
    if X['details']['name'][:3] == 'AP_':
        dados = umap.UMAP(n_components=100, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'UMIST_Faces_Cropped':
        dados = umap.UMAP(n_components=50, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'GLI':
        dados = umap.UMAP(n_components=80, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'SRBCT':
        dados = umap.UMAP(n_components=80, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'Lung':        
        dados = umap.UMAP(n_components=50, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'artificial-characters':
        dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)
    elif X['details']['name'][-2:] == 'wc':
        dados = umap.UMAP(n_components=100, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'coil-20':
        dados = umap.UMAP(n_components=100, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'cnae-9':
        dados = umap.UMAP(n_components=100, random_state=42).fit_transform(dados)
    elif X['details']['name'] == 'satimage':
        dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)
    elif X['details']['name'] == 'leukemia':
        dados = umap.UMAP(n_components=70, random_state=42).fit_transform(dados)
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

filename = "results.txt"
line = ''

# Open the file in 'w' (write) mode. This will overwrite the file if it exists.
with open(filename, 'w') as f:
    

    #########################################
    # Execution of Spectral clustering 
    #########################################
    inicio = time.time()
    spectral = SpectralClustering(n_clusters=c, affinity='nearest_neighbors').fit(dados)
    #spectral = SpectralClustering(n_clusters=c).fit(dados)
    fim = time.time()
    print()
    print('Spectral Clustering')
    print('---------------------')
    print('Elapsed time: %.4f' %(fim - inicio))
    print()
    ari = adjusted_rand_score(target, spectral.labels_)
    print('Rand index: %.4f' %ari)
    mi = adjusted_mutual_info_score(target, spectral.labels_)
    print('Mutual information: %.4f' %mi)
    fm = fowlkes_mallows_score(target, spectral.labels_)
    print('FM: %.4f' %fm)
    vm = v_measure_score(target, spectral.labels_)
    print('V-measure: %.4f' %vm)
    print()

    float_list = [ari, mi, fm, vm]
    for item in float_list:
        f.write(str(item) + '\t')



    #########################################
    # Execution of HDBSCAN
    #########################################
    inicio = time.time()
    hdbscan = HDBSCAN(min_cluster_size=10).fit(dados)
    fim = time.time()
    print()
    print('HDBSCAN Clustering')
    print('---------------------')
    print('Elapsed time: %.4f' %(fim - inicio))
    print()
    ari = adjusted_rand_score(target, hdbscan.labels_)
    print('Rand index: %.4f' %ari)
    mi = adjusted_mutual_info_score(target, hdbscan.labels_)
    print('Mutual information: %.4f' %mi)
    fm = fowlkes_mallows_score(target, hdbscan.labels_)
    print('FM: %.4f' %fm)
    vm = v_measure_score(target, hdbscan.labels_)
    print('V-measure: %.4f' %vm)
    print()

    float_list = [ari, mi, fm, vm]
    for item in float_list:
        f.write(str(item) + '\t')


    MAX = 100

    #########################
    # Execution of kmeans++
    #########################
    list_ari = []
    list_mi = []
    list_fm = []
    list_vm = []
    inicio = time.time()
    for i in range(MAX):
        kmeans = KMeans(n_clusters=c, init='random', n_init=1).fit(dados)
        list_ari.append(adjusted_rand_score(target, kmeans.labels_))
        list_mi.append(adjusted_mutual_info_score(target, kmeans.labels_))
        list_fm.append(fowlkes_mallows_score(target, kmeans.labels_))
        list_vm.append(v_measure_score(target, kmeans.labels_))    
    fim = time.time()

    print('K-MEANS')
    print('----------')
    print('Elapsed time: %.4f' %(fim - inicio))
    print()
    ari = sum(list_ari)/MAX
    print('Adjusted Rand index: %.4f' %ari)
    mi = sum(list_mi)/MAX
    print('Mutual information: %.4f' %mi)
    fm = sum(list_fm)/MAX
    print('FM: %.4f' %fm)
    vm = sum(list_vm)/MAX
    print('V-measure: %.4f' %vm)
    print()

    float_list = [ari, mi, fm, vm]
    for item in float_list:
        f.write(str(item) + '\t')

    ##############################
    # k-MST clustering
    ##############################
    t = mst_clustering_divisive_CH(T_min, dados, target)
    labels = label_samples(t, target)

    print('Minimum Information Tree based indices')
    print('---------------------------------------')
    ari = adjusted_rand_score(target, labels)
    print('Rand index: %.4f' %ari)
    mi = adjusted_mutual_info_score(target, labels)
    print('Mutual information: %.4f' %mi)
    fm = fowlkes_mallows_score(target, labels)
    print('FM: %.4f' %fm)
    vm = v_measure_score(target, labels)
    print('V-measure: %.4f' %vm)
    print()

    float_list = [ari, mi, fm, vm]
    for item in float_list:
        f.write(str(item) + '\t')
