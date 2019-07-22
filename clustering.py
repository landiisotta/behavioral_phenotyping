import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class HclustEmbeddings:
    """ Performs hierarchical clustering on patient embeddings"""

    def __init__(self, min_cl, max_cl, linkage, affinity):
        self.min_cl = min_cl
        self.max_cl = max_cl
        self.linkage = linkage
        self.affinity = affinity

    def find_best_nclu(self,
                       mtx,
                       n_iter,
                       subsampl):
        """Iterate clustering of subsets anf find best number of clusters

        Parameters
        ----------
        mtx: list
            List of embeddings as returned by pt_embedding module
        n_iter: int
            number of iteration to select the best number of clusters
        subsampl: float
            Fraction of data to consider for clustering

        Returns
        -------
        int
            Best number of clusters
        """
        n_cl_selected = []
        for it in range(n_iter):
            idx = np.random.randint(0, len(mtx), int(len(mtx) * subsampl))
            sub_data = [mtx[i] for i in idx]
            best_silh = 0
            best_n_clu = 0
            for n_clu in range(self.min_cl, self.max_cl):
                hclu = AgglomerativeClustering(n_clusters=n_clu,
                                               linkage=self.linkage,
                                               affinity=self.affinity)
                lab_cl = hclu.fit_predict(sub_data)
                tmp_silh = silhouette_score(sub_data, lab_cl)
                if tmp_silh > best_silh:
                    best_silh = tmp_silh
                    best_n_clu = n_clu
            print("(*) Iter {0} -- N clusters {1}".format(it, best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    @staticmethod
    def fit(mtx, pid_list, n_clu):
        """ Perform HC on patient embeddings

        Parameters
        ----------
        mtx: list
            Embeddings list
        pid_list: list
            List of subjects id ordered as in mtx
        n_clu: int
            Number of clusters

        Returns
        -------
        dictionary
            Dictionary with cluster label per subject id
            {pid: cl}
        """
        hclu = AgglomerativeClustering(n_clusters=n_clu)
        lab_cl = hclu.fit_predict(mtx)
        silh = silhouette_score(mtx, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(pid_list, lab_cl)}


class HclustFeatures:
    """ Performs Hierarchical clustering on feature data"""

    def __init__(self, min_cl, max_cl, linkage, affinity):
        self.min_cl = min_cl
        self.max_cl = max_cl
        self.linkage = linkage
        self.affinity = affinity

    def find_best_nclu(self,
                       df_scaled,
                       n_iter,
                       subsampl):
        """ Find the best number of clusters iterating over subset of data

        Parameters
        ----------
        df_scaled: dataframe
            Scaled feature data with patient ids as index
        n_iter: int
            Number of iterations to perform
        subsampl: float
            Fraction of data to consider in the subset
            at each iteration

        Returns
        -------
        int
            best number of clusters
        """
        n_cl_selected = []
        for it in range(n_iter):
            idx = np.random.randint(0, len(df_scaled), int(len(df_scaled) * subsampl))
            sub_df = df_scaled.iloc[[i for i in idx], :]
            best_silh = 0
            best_n_clu = 2
            for n_clu in range(self.min_cl, self.max_cl):
                hclu = AgglomerativeClustering(n_clusters=n_clu)
                lab_cl = hclu.fit_predict(sub_df)
                tmp_silh = silhouette_score(sub_df, lab_cl)
                if tmp_silh > best_silh:
                    best_silh = tmp_silh
                    best_n_clu = n_clu
            print("(*) Iter {0} -- N clusters {1}".format(it,
                                                          best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    @staticmethod
    def fit(df_scaled, n_clu):
        """Fit HC on patient feature data

        Parameters
        ----------
        df_scaled: dataframe
            Dataframe of scaled feature data
        n_clu: int
            Number of clusters
        Returns
        -------
        dictionary
            Dictionary of patient ids and correspondent
            clusters {pid: cl}
        """
        hclu = AgglomerativeClustering(n_clusters=n_clu)
        lab_cl = hclu.fit_predict(df_scaled)
        silh = silhouette_score(df_scaled, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(df_scaled.index, lab_cl)}
