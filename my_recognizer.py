import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set
   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    all_sequences = test_set.get_all_sequences()
    all_Xlengths = test_set.get_all_Xlengths()
    for sequence in all_sequences:
        probability = {}
        X, length = all_Xlengths[sequence]
        for word_model, model in models.items():
            try:
                score = model.score(X, length)
                probability[word_model] = score
            except:
                score = -float("inf")
                probability[word_model] = score
        probabilities.append(probability)
        values = list(probability.values())
        keys = list(probability.keys())
        guesses.append(keys[values.index(max(values))])
    return probabilities, guesses


