from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import utils as ut
import glove
import numpy as np


class Pembeddings():
    def __init__(self, behr, vocab):
        """ Range of possible embeddings to perform on behavioral data
        TFIDF, GLOVE

        Parameters
        ----------
        behr
            dictionary {pid: trm sequence}
        vocab
            dictionary, needed idx_to_mt
        """
        self.behr = behr
        self.vocab = vocab

    def tfidf(self):
        """performs TFIDF

        Return
        ------
        list
            pids list
        list
            svd matrix
        """
        # create document list
        doc_list = []
        for tupl_list in self.behr.values():
            sentence = []
            for tm_vect in tupl_list:
                sentence.extend(tm_vect[1])
            doc_list.append(' '.join(list(map(lambda x: str(x), sentence))))
        pid_list = [pid for pid in self.behr]

        vectorizer = TfidfVectorizer(norm='l2')
        tfidf_mtx = vectorizer.fit_transform(doc_list)

        print("Performing SVD on the TF-IDF matrix...")
        reducer = TruncatedSVD(n_components=ut.n_dim, random_state=123)
        svd_mtx = reducer.fit_transform(tfidf_mtx)

        return pid_list, svd_mtx

    def glove_pemb(self):
        """Computes Glove embeddings from co-occurrence matrix
            and returns patient embeddings

        Return
        ------
        list
            pids list
        list
            matrix of patient embeddings
        """
        # behrs wrt timeframes
        behr_tf = {}
        for pid, bseq in self.behr.items():
            for aoa, tkns in bseq:
                if pid not in behr_tf:
                    behr_tf[pid] = {_age_tf(aoa): list(map(lambda x: int(x),
                                                           tkns))}
                else:
                    behr_tf[pid].setdefault(_age_tf(aoa),
                                            list()).extend(list(map(lambda x: int(x),
                                                                    tkns)))
        corpus = _build_corpus(behr_tf)
        coocc_dict = build_cooccur(self.vocab, corpus, window_size=20)

        model = glove.Glove(coocc_dict, alpha=0.75, x_max=100.0, d=ut.n_dim)
        print("\nTraining Glove embeddings...")
        for epoch in range(ut.n_epoch):
            err = model.train(batch_size=ut.batch_size)
            if epoch % 10 == 0:
                print("epoch %d, error %.3f" % (epoch, err), flush=True)

        Wemb = model.W + model.ContextW  # as suggested in Pennington et al.
        p_emb = []
        pid_list = []
        for pid, term in corpus.items():
            if len(term) != 0:
                pid_list.append(pid)
                p_emb.append(np.mean([Wemb[int(t)].tolist() for t in term],
                                     axis=0).tolist())

        return pid_list, p_emb


def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """Build a word co-occurrence dictionary for the given corpus.

    Parameters
    ----------
    vocab
        dictionary with the vocabulary of the form idx_to_mt
    corpus
        behr dictionary as returned by _build_corpus
    window_size
        int, size of the context window
    min_count
        int, if not `None`, cooccurrence pairs where either word
        occurs in the corpus fewer than `min_count` times are ignored.
    Return
    ------
    dictionary
        {i_main: {i_context: cooccurrence}}
        see Pennington et al., (2014).
    """

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = {k: {} for k in vocab}

    for pid, sentence in corpus.items():

        for center_i, center_id in enumerate(sentence):
            # Collect all word IDs in left window of center word
            context_ids = sentence[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                if left_id in cooccurrences[center_id]:
                    cooccurrences[center_id][left_id] += increment
                    cooccurrences[left_id][center_id] += increment
                else:
                    cooccurrences[center_id][left_id] = increment
                    cooccurrences[left_id][center_id] = increment

    return cooccurrences


"""
Private functions
"""


def _age_tf(age):
    """ convert age to time slot string

    Parameter
    ---------
    age
        float
    Return
    ------
    str
    """
    if 0 < age <= 2.5:
        return 'F1'
    elif 2.5 < age <= 6.0:
        return 'F2'
    elif 6.0 < age <= 13.0:
        return 'F3'
    elif 13.0 < age < 17.0:
        return 'F4'
    else:
        return 'F5'



def _build_corpus(behr):
    """random shuffle terms in time slots

    Parameters
    ----------
    behr
        dictionary {pid: trm sequence}
    Return
    ------
    dictionary
        {pid: trm list set and shuffles wrt to time slots F1-F5}
    """
    corpus = {}
    for pid, tf_dict in behr.items():
        for tf in sorted(tf_dict.keys()):
            np.random.shuffle(behr[pid][tf])
            corpus.setdefault(pid,
                              list()).extend(behr[pid][tf])
    return corpus
