import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage


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
            best_n_clu = self.elbow_method(sub_data)
            # for n_clu in range(self.min_cl, self.max_cl):
            #     hclu = AgglomerativeClustering(n_clusters=n_clu,
            #                                    linkage=self.linkage,
            #                                    affinity=self.affinity)
            #     lab_cl = hclu.fit_predict(sub_data)
            #     tmp_silh = silhouette_score(sub_data, lab_cl)
            #     if tmp_silh > best_silh:
            #         best_silh = tmp_silh
            #         best_n_clu = n_clu
            # print("(*) Iter {0} -- N clusters {1}".format(it, best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    def elbow_method(self,
                     mtx):
        """Select the best number of clusters via elbow method.

        Parameters
        ----------
        mtx list:
            List of embeddings as returned by pt_embedding module

        Returns
        -------
        int:
            Best number of clusters
        """
        # Scale data.
        scaler = MinMaxScaler()
        mtx = scaler.fit_transform(mtx)

        Z = linkage(mtx, self.linkage)
        last = Z[-self.max_cl:, 2]

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]

        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

        return k

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
        # Scale data matrix
        scaler = MinMaxScaler()
        mtx = scaler.fit_transform(mtx)

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
            best_n_clu = self.elbow_method(sub_df)
            # for n_clu in range(self.min_cl, self.max_cl):
            #     hclu = AgglomerativeClustering(n_clusters=n_clu)
            #     lab_cl = hclu.fit_predict(sub_df)
            #     tmp_silh = silhouette_score(sub_df, lab_cl)
            #     if tmp_silh > best_silh:
            #         best_silh = tmp_silh
            #         best_n_clu = n_clu
            # print("(*) Iter {0} -- N clusters {1}".format(it,
            #                                               best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    def elbow_method(self,
                     df_scaled):
        """Select the best number of clusters via elbow method.

        Parameters
        ----------
        df_scaled dataframe:
            Scaled feature data with patient ids as index

        Returns
        -------
        int:
            Best number of clusters
        """

        data = df_scaled.to_numpy()

        Z = linkage(data, self.linkage)
        last = Z[-self.max_cl:, 2]

        acceleration = np.diff(last, 2)  # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters

        return k

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


class KMeansEmbeddings:
    """ Performs KMeans on patient embeddings"""

    def __init__(self, min_cl, max_cl):
        self.min_cl = min_cl
        self.max_cl = max_cl

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
            best_n_clu = self.elbow_method(sub_data)
            # for n_clu in range(self.min_cl, self.max_cl):
            #     hclu = AgglomerativeClustering(n_clusters=n_clu,
            #                                    linkage=self.linkage,
            #                                    affinity=self.affinity)
            #     lab_cl = hclu.fit_predict(sub_data)
            #     tmp_silh = silhouette_score(sub_data, lab_cl)
            #     if tmp_silh > best_silh:
            #         best_silh = tmp_silh
            #         best_n_clu = n_clu
            # print("(*) Iter {0} -- N clusters {1}".format(it, best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    def elbow_method(self,
                     mtx):
        """Select the best number of clusters via elbow method.

        Parameters
        ----------
        mtx list:
            List of embeddings as returned by pt_embedding module

        Returns
        -------
        int:
            Best number of clusters
        """
        # Scale data.
        scaler = MinMaxScaler()
        mtx = scaler.fit_transform(mtx)

        inertia = []  # Sum of square differences of samples from cluster centers
        K = range(1, self.max_cl)

        for k in K:
            kmean_model = KMeans(n_clusters=k).fit(mtx)
            inertia.append(kmean_model.inertia_)

        acceleration = np.diff(inertia, 2)

        k = acceleration.argmax() + 2  # If idx 0 is the max of this we want 2 clusters

        return k

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
        # Scale data matrix
        scaler = MinMaxScaler()
        mtx = scaler.fit_transform(mtx)

        kmclu = KMeans(n_clusters=n_clu)
        lab_cl = kmclu.fit_predict(mtx)
        silh = silhouette_score(mtx, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(pid_list, lab_cl)}


class KMeansFeatures:
    """ Performs Hierarchical clustering on feature data"""

    def __init__(self, min_cl, max_cl):
        self.min_cl = min_cl
        self.max_cl = max_cl

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
            best_n_clu = self.elbow_method(sub_df)
            # for n_clu in range(self.min_cl, self.max_cl):
            #     hclu = AgglomerativeClustering(n_clusters=n_clu)
            #     lab_cl = hclu.fit_predict(sub_df)
            #     tmp_silh = silhouette_score(sub_df, lab_cl)
            #     if tmp_silh > best_silh:
            #         best_silh = tmp_silh
            #         best_n_clu = n_clu
            # print("(*) Iter {0} -- N clusters {1}".format(it,
            #                                               best_n_clu))
            n_cl_selected.append(best_n_clu)
        unique, counts = np.unique(n_cl_selected, return_counts=True)
        print("Counts of N clusters:")
        print("N clusters -- Count")
        for un, ct in dict(zip(unique, counts)).items():
            print(un, ct)
        best_n_clu = unique[np.argmax(counts)]
        print("\nBest N cluster:{0}".format(best_n_clu))
        return best_n_clu

    def elbow_method(self,
                     df_scaled):
        """Select the best number of clusters via elbow method.

        Parameters
        ----------
        df_scaled dataframe:
            Scaled feature data with patient ids as index

        Returns
        -------
        int:
            Best number of clusters
        """

        data = df_scaled.to_numpy()

        inertia = []  # Sum of square differences of samples from cluster centers
        K = range(1, self.max_cl)

        for k in K:
            kmean_model = KMeans(n_clusters=k).fit(data)
            inertia.append(kmean_model.inertia_)

        acceleration = np.diff(inertia, 2)

        k = acceleration.argmax() + 2  # If idx 0 is the max of this we want 2 clusters

        return k

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
        kmclu = KMeans(n_clusters=n_clu)
        lab_cl = kmclu.fit_predict(df_scaled)
        silh = silhouette_score(df_scaled, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(df_scaled.index, lab_cl)}


def compare_clustering(cl1, cl2, method):
    """Compute cluster comparison score (compare favorite cluster to other clustering techniques),
    either Adjusted Mutual Information Score (AMI), or Fowlkes - Mallows Score (FM)

    Parameters
    ----------
    cl1: list, array
        first clustering labels
    cl2: list, array
        second clustering labels
    method: str
        either 'AMI' or 'FM'
    Returns
    -------
    float
        desired score
    """
    if method == 'AMI':
        return adjusted_mutual_info_score(cl1, cl2)
    else:
        return fowlkes_mallows_score(cl1, cl2)
