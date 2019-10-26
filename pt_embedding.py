from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import glove
import numpy as np
import utils as ut
import torch
import torch.nn.functional as F
import logging


class Pembeddings:
    def __init__(self, behr, vocab):
        """ Range of possible embeddings to perform on behavioral data
        TFIDF, GLOVE, WORD2VEC

        Parameters
        ----------
        behr
            dictionary {pid: trm sequence}
        vocab
            dictionary, needed btm_to_idx
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
                sentence.extend(tm_vect[2:])
            doc_list.append(' '.join(list(map(lambda x: str(x), sentence))))
        pid_list = [pid for pid in self.behr]

        vectorizer = TfidfVectorizer(norm='l2')
        tfidf_mtx = vectorizer.fit_transform(doc_list)

        logging.info("Performing SVD on the TF-IDF matrix...")
        reducer = TruncatedSVD(n_components=ut.n_dim_tfidf, random_state=123)
        svd_mtx = reducer.fit_transform(tfidf_mtx)

        return pid_list, svd_mtx

    def word2vec_emb(self):
        """Skip-gram word2vec

        Returns
        -------
        list
            pids list
        list
            matrix of patient embeddings
        numpy array:
            first layer weight matrix (vocab size, embedding dim)
        numpy array:
            second layer weight matrix (vocab size, embedding dim)

        """
        corpus = self.__build_corpus()
        idx_pairs = self.__get_idx_pairs(corpus, window_size=10)

        torch.manual_seed(1234)
        W1 = torch.randn(ut.n_dim_w2v, len(self.vocab),
                         dtype=torch.float32,
                         requires_grad=True)
        W2 = torch.randn(len(self.vocab), ut.n_dim_w2v,
                         dtype=torch.float32,
                         requires_grad=True)

        for epoch in range(ut.n_epoch_w2v):
            loss_val = 0
            for data, target in idx_pairs:
                x = self.__get_input_layer(data).float()
                y_true = torch.from_numpy(np.array([target])).long()

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                loss_val += loss.item()
                loss.backward()
                w1 = W1.detach()
                w2 = W2.detach()
                w1 -= ut.learning_rate_w2v * W1.grad
                w2 -= ut.learning_rate_w2v * W2.grad

                W1.grad.zero_()
                W2.grad.zero_()

            if epoch % 10 == 0:
                logging.info(f'Loss at epoch {epoch}: {loss_val/len(idx_pairs)}')
        logging.info(f'Loss at epoch {epoch}: {loss_val/len(idx_pairs)}')

        p_emb = []
        pid_list = []
        for pid, term in corpus.items():
            if len(term) != 0:
                pid_list.append(pid)
                p_emb.append(np.mean([W1[:, int(t)].tolist() for t in term],
                                     axis=0).tolist())

        return pid_list, p_emb, w1.numpy(), w2.numpy()

    def glove_pemb(self):
        """Computes Glove embeddings from co-occurrence matrix
            and returns patient embeddings

        Return
        ------
        list
            pids list
        list
            matrix of patient embeddings
        array
            word embeddings
        """

        corpus = self.__build_corpus()
        coocc_dict = self.__build_cooccur(corpus, window_size=10)
        model = glove.Glove(coocc_dict, alpha=0.75, x_max=10.0, d=ut.n_dim_glove, seed=1234)
        logging.info("\nTraining Glove embeddings...")
        for epoch in range(ut.n_epoch_glove):
            err = model.train(batch_size=ut.batch_size_glove, step_size=ut.learning_rate_glove)
            if epoch % 10 == 0:
                logging.info("epoch %d, error %.3f" % (epoch, err))
        logging.info("epoch %d, error %.3f" % (epoch, err))

        wemb = model.W + model.ContextW  # as suggested in Pennington et al.
        p_emb = []
        pid_list = []
        for pid, term in corpus.items():
            if len(term) != 0:
                pid_list.append(pid)
                p_emb.append(np.mean([wemb[int(t)].tolist() for t in term],
                                     axis=0).tolist())

        return pid_list, p_emb, wemb

    @staticmethod
    def __age_tf(age):
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

    def __build_corpus(self):
        """random shuffle terms in time slots

        Return
        ------
        dictionary
            {pid: term list set and shuffles wrt to time slots F1-F5}
        """
        # set seed
        np.random.seed(0)  # 1234 (3 ns subtypes); 47 (7 ns subtypes)
        # We structure behrs wrt timeframes to learn word embeddings.
        # Structure of bvect = [Penc, aoa, tokens].
        behr_tf = {}
        for pid, bvect in self.behr.items():
            for el in bvect:
                if pid not in behr_tf:
                    behr_tf[pid] = {self.__age_tf(el[1]): list(map(lambda x: int(self.vocab[x]),
                                                                   el[2:]))}
                else:
                    behr_tf[pid].setdefault(self.__age_tf(el[1]),
                                            list()).extend(list(map(lambda x: int(self.vocab[x]),
                                                                    el[2:])))
        corpus = {}
        for pid, tf_dict in behr_tf.items():
            for tf in sorted(tf_dict.keys()):
                np.random.shuffle(behr_tf[pid][tf])
                corpus.setdefault(pid,
                                  list()).extend(behr_tf[pid][tf])
        return corpus

    @staticmethod
    def __get_idx_pairs(corpus, window_size):
        """Creates the center-context vectors for Word2vec predictions

        Parameters
        ----------
        corpus: dictionary
            {pid: behr}
        window_size: int
            size of the context
        Returns
        -------
        numpy array
        """
        idx_pairs = []
        # for each sentence
        for sentence in corpus.values():
            # for each word, treated as center word
            for center_word_pos in range(len(sentence)):
                # for each window position
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make sure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(sentence) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = sentence[context_word_pos]
                    idx_pairs.append((sentence[center_word_pos], context_word_idx))

        return np.array(idx_pairs)

    def __get_input_layer(self, word_idx):
        """Transforms a token into a one-hot encoded representation

        Parameters
        ----------
        word_idx: int
            word token
        Returns
        -------
        torch tensor
        """
        x = torch.zeros(len(self.vocab), dtype=torch.float32)
        x[word_idx] = 1.0
        return x

    def __build_cooccur(self, corpus, window_size=10):
        """Build a word co-occurrence dictionary for the given corpus.

        Parameters
        ----------
        corpus
            behr dictionary as returned by __build_corpus
        window_size
            int, size of the context window

        Return
        ------
        dictionary
            {i_main: {i_context: cooccurrence}}
            see Pennington et al., (2014).
        """

        # Collect cooccurrences internally as a sparse matrix for passable
        # indexing speed; we'll convert into a list later
        cooccurrences = {k: {} for k in self.vocab.values()}

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
