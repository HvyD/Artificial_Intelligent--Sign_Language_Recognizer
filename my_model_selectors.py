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
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


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
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        
        
        p: number of parameters ( p = n_components  : is number of states in HMM)
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # list of BIC score
        list_scores = []
        
        # list of number states in HMM
        list_num_hidden_states = []
        
        #num_features = len(self.X[0])
        num_features = self.X.shape[1]
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                logL = model.score(self.X, self.lengths)
                
                
                ''' GaussianHMM
                source : https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
                GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, 
                random_state=self.random_state, verbose=False).fit(self.X, self.lengths).
                
                From hmmlearn, calculating the following parameters that are used in BIC
                
                p is the number of model parameters in the test.
                 https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/4
                Initial state occupation probabilities = numStates
                Transition probabilities = numStates*(numStates - 1)
                Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
                Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                '''
                
                initial_state_occupation_probabilities = num_hidden_states
                transition_probabilities = num_hidden_states * (num_hidden_states - 1)
                emission_probabilities = num_hidden_states * num_features * 2
                
                p = initial_state_occupation_probabilities + transition_probabilities + emission_probabilities
                
                bic = -2 * logL + p * np.log(N)
                
                list_scores.append(bic)
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
          
        if list_scores:
            best_num_hidden_states = list_num_hidden_states[np.argmin(list_scores)] 
        else:
            best_num_hidden_states = self.n_constant
            
        #best_num_states = list_num_states[np.argmin(list_scores)] if list_scores else self.n_constant
        
        #print("[SelectorBIC] Result model n_compents: {}".format(best_num_hidden_states))
        
        return self.base_model(best_num_hidden_states)           
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    '''
    Note:
        self.words = all_word_sequences
        self.sequences = all_word_sequences[this_word]
        i.e  self.sequences = self.words[this_word]
        
        self.hwords = all_word_Xlengths
        self.X, self.lengths = all_word_Xlengths[this_word]
        i.e self.X, self.lengths = self.hwords[this_word]
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        # list of number states in HMM
        list_num_hidden_states = []
        
        list_scores = []
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states)
                logL = model.score(self.X, self.lengths)
                
                # calculate average score over all the other words other than the current word
                sum_logL = 0
                K = 0
                
                for word in self.words:
                    if word != self.this_word:
                        other_word_X, other_word_lengths = self.hwords[word]
                        try:
                            sum_logL += hmm_model.score(other_word_X, other_word_lengths)
                            K += 1
                            
                        except:
                            pass
                
                if K > 0:
                    average_logL = sum_logL/K
                else:
                    average_logL = 0
                    
                #calculate the total score
                dic = logL - average_logL

                list_scores.append(dic)
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
            
            
        M = len(list_num_hidden_states) # length of list of number of hidden states
        if M > 2:
            best_num_hidden_states = list_num_hidden_states[np.argmax(list_scores)]
            
        elif M == 2:
            best_num_hidden_states = list_num_hidden_states[0]
            
        else:
            best_num_hidden_states = self.n_constant
            
        #print("[SelectorDIC] Result model n_compents: {}".format(best_num_hidden_states))
        return self.base_model(best_num_hidden_states) 
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # n_components : is number of hidden states in HMM
        
        split_method = KFold()
        
        # list of mean score of each Cross-Validation
        list_scores = []
        
        # list of number states in HMM
        list_num_hidden_states = []
        
        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                if len(self.sequences) > 2: # Check if there are enough data to split
                    scores = []
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # training sequences
                        self.X, self.lengths =  combine_sequences(cv_train_idx, self.sequences)
                        # testing sequences
                        test_X, test_lengths =  combine_sequences(cv_test_idx, self.sequences)
            
                        model = self.base_model(num_hidden_states)
                        scores.append(model.score(test_X, test_lengths))
   
                    list_scores.append(np.mean(scores))
                
                else:
                    model = self.base_model(num_hidden_states)
                    list_scores.append(model.score(self.X, self.lengths))
                    
                list_num_hidden_states.append(num_hidden_states)
                
            except:
                # eliminate non-viable models from consideration
                pass
            
        if list_scores:
            best_num_hidden_states = list_num_hidden_states[np.argmax(list_scores)] 
        else:
            best_num_hidden_states = self.n_constant
            
        #best_num_states = list_num_hidden_states[np.argmax(list_scores)] if list_scores else self.n_constant
        
        #print("[SelectorCV] Result model n_compents: {}".format(best_num_hidden_states))
        
        return self.base_model(best_num_hidden_states)
