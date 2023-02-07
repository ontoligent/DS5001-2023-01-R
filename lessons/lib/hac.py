import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist #, squareform
# from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt            

class HAC:
    """This class takes an arbitrary vector space and represents it 
    as a hierarhical agglomerative cluster tree. The number of observations
    should be sufficiently small to allow being plotted."""

    w:int = 10
    labelsize:int = 14
    orientation:str = 'left'
    sim_metric:str = 'cosine' # The distance metric to use. The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    tree_method:str = 'ward'
    norm:str = 'l2' # l1, l2, max
    
    def __init__(self, M, labels=None):
        self.M = M
        self.h = M.shape[0]
        if labels:
            self.labels = labels            
        else:
            self.labels = M.index.tolist()

    def get_sims(self):
        self.SIMS = pdist(normalize(self.M, norm=self.norm), metric=self.sim_metric)

    def get_tree(self):
        self.TREE = sch.linkage(self.SIMS, method=self.tree_method)        
        
    def plot_tree(self):
        plt.figure()
        fig, axes = plt.subplots(figsize=(self.w, self.h / 3))
        dendrogram = sch.dendrogram(self.TREE, labels=self.labels, orientation=self.orientation);
        plt.tick_params(axis='both', which='major', labelsize=self.labelsize)
        
    def plot(self):
        self.get_sims()
        self.get_tree()
        self.plot_tree()
