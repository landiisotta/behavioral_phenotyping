import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd


class HclustEmbeddings():
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

        :param mtx: list (tfidf, glove)
        :param pid_list: list
        :param n_iter: num
        :param subsampl: float
        :return: num
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

    def fit(self, mtx, pid_list, best_n_clu):
        """fit HC on patient embeddings

        :param mtx: list (tfidf, glove)
        :param pid_list: list
        :param best_n_clu: int
        :return: dictionary {pid: cl}
        """
        hclu = AgglomerativeClustering(n_clusters=best_n_clu)
        lab_cl = hclu.fit_predict(mtx)
        silh = silhouette_score(mtx, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (best_n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(pid_list, lab_cl)}


class HclustFeatures():
    def __init__(self, min_cl, max_cl, linkage, affinity):
        self.min_cl = min_cl
        self.max_cl = max_cl
        self.linkage = linkage
        self.affinity = affinity

    def find_best_nclu(self,
                       df_scaled,
                       n_iter,
                       subsampl):
        """Iterate clustering of subsets anf find best number of clusters

        :param df_scaled: dataframe with pid as index
        :param pid_list: list
        :param n_iter: num
        :param subsampl: float
        :return: num
        """
        n_cl_selected = []
        for it in range(n_iter):
            idx = np.random.randint(0, len(df_scaled), int(len(df_scaled) * subsampl))
            sub_df = df_scaled.iloc[[i for i in idx], :]
            best_silh = 0
            for n_clu in range(self.min_cl, self.max_cl):
                hclu = AgglomerativeClustering(n_clusters=n_clu)
                lab_cl = hclu.fit_predict(sub_df)
                tmp_silh = silhouette_score(sub_df, lab_cl)
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

    def fit(self, df_scaled, best_n_clu):
        """fit HC on patient embeddings

        :param df_scaled: dataframe with pid as index
        :param best_n_clu: int
        :return: dictionary {pid: cl}
        """
        hclu = AgglomerativeClustering(n_clusters=best_n_clu)
        lab_cl = hclu.fit_predict(df_scaled)
        silh = silhouette_score(df_scaled, lab_cl)
        print('(*) Number of clusters %d -- Silhouette score %.2f' % (best_n_clu, silh))

        num_count = np.unique(lab_cl, return_counts=True)[1]
        for idx, nc in enumerate(num_count):
            print("Cluster {0} -- Numerosity {1}".format(idx, nc))
        print('\n')
        print('\n')

        return {pid: cl for pid, cl in zip(df_scaled.index, lab_cl)}
