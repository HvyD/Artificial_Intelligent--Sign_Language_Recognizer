import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            model = self.compute_model(num_states)
            if self.verbose:
                fmt = 'model created for {} with {} states'
                print(fmt.format(self.this_word, num_states))
            return model
        except:
            if self.verbose:
                fmt = 'failure on {} with {} states'
                print(fmt.format(self.this_word, num_states))
            return None

    def compute_model(self, n):
        '''Computes a GaussianHMM model.
        :param n
            The number of components.
        '''
        return GaussianHMM(
            n_components=n,
            covariance_type="diag",
            n_iter=1000,
            random_state=self.random_state,
            verbose=False
        ).fit(self.X, self.lengths)


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    '''Select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    '''

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        results = self.compute_for_all_n()
        if not results:
            return None
        results = [(self.compute_bic(n,l), m) for n, l, m in results]
        _, model = min(results, key=lambda x: x[0])
        return model

    def compute_for_all_n(self):
        '''Computes the model and log likelihood for all combinations of
        components.
        :return
            A list of tuples of the form (n, logL), where n is the number
            of components used to train the model, and logL is the log
            likelihood for the given model.
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        results = []
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            if not model: continue
            try:
                logl = model.score(self.X, self.lengths)
            except:
                continue
            results.append((n, logl, model))
        return results

    def compute_bic(self, n, logl):
        '''Computes the Bayesian Information Criterion (BIC) score given n and
        the log likelihood.
        '''
        return (-2*logl) + (self.compute_free_param(n)*np.log2(n))

    def compute_free_param(self, n):
        '''Computes the number of free paramters for a model of n components.
        :param n
            The number of components.
        :return
            The total number of free parameters.
        '''
        return n**2 + 2*len(self.X[0])*n - 1


class SelectorDIC(ModelSelector):
    '''Select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to
    hmm topology optimization." Document Analysis and Recognition, 2003.
    Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        results = self.compute_for_all_n()
        if not results:
            return None
        _, model = max(results, key=lambda x: x[0])
        return model

    def compute_for_all_n(self):
        '''Computes the DIC for all components.
        :return
            A list of tuples of the form (DIC, model).
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        results = []
        other_words = set(self.hwords.keys())
        other_words.remove(self.this_word)
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            if not model: continue
            try:
                logl = model.score(self.X, self.lengths)
                log_other = self.compute_log_other(model, other_words)
                if not log_other: continue
            except:
                continue
            results.append((logl-log_other, model))
        return results

    def compute_log_other(self, model, words):
        '''Compute the average log likelihood of the other words.'''
        if not model or not words:
            return None
        total = 0
        log_other = 0.0
        for w in words:
            xo, lo = self.hwords[w]
            try:
                log_other += model.score(xo, lo)
                total += 1
            except:
                continue
        return log_other / total


class SelectorCV(ModelSelector):
    '''Select best model based on average log Likelihood of cross-validation
    folds.
    '''

    def select(self):
        '''Selects cross-validated model with best log likelihood result.'''
        results = self.compute_for_all_n()
        if not results:
            return None
        n, _ = max(results, key=lambda x: x[1])
        return self.base_model(n)

    def compute_for_all_n(self):
        '''Computes the model and log likelihood for all combinations of
        components.
        :return
            A list of tuples of the form (n, logL), where n is the number
            of components used to train the model, and logL is the log
            likelihood for the given model.
        '''
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Save values to restore before returning
        cross_n_results = []
        len_sequence = len(self.sequences)
        if len_sequence <= 1:
            for n in range(self.min_n_components, self.max_n_components+1):
                list_logl = []  # list of cross validation scores obtained
                model = self.base_model(n)
                if not model:
                    continue
                try:
                    logl = model.score(self.X, self.lengths)
                except:
                    continue
                list_logl.append(logl)

                if list_logl:
                    avg_logl = np.mean(list_logl)
                    cross_n_results.append((n, avg_logl))
            return cross_n_results

        save_x, save_lens = self.X, self.lengths
        for n in range(self.min_n_components, self.max_n_components+1):
            split_method = KFold(len_sequence if len_sequence<3  else 3)
            list_logl = []  # list of cross validation scores obtained

            for train_idx, test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                model = self.base_model(n)
                if not model: continue
                test_x, test_len = combine_sequences(test_idx, self.sequences)
                try:
                    logl = model.score(test_x, test_len)
                except:
                    continue
                list_logl.append(logl)

            if list_logl:
                avg_logl = np.mean(list_logl)
                cross_n_results.append((n, avg_logl))
        # Restore values
        self.X, self.lengths = save_x, save_lens
        return cross_n_results
