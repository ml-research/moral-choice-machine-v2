import sys
import os
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

# ----------------------------------------------------------------------
def _calculate_languages_ratios(text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

    @param text: Text whose language want to be detected
    @type text: str

    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """

    languages_ratios = {}

    '''
    nltk.wordpunct_tokenize() splits all punctuations into separate tokens

    >>> wordpunct_tokenize("That's thirty minutes away. I'll be there in ten.")
    ['That', "'", 's', 'thirty', 'minutes', 'away', '.', 'I', "'", 'll', 'be', 'there', 'in', 'ten', '.']
    '''

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    return languages_ratios


# ----------------------------------------------------------------------
def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    @param text: Text whose language want to be detected
    @type text: str

    @return: Most scored language guessed
    @rtype: str
    """

    ratios = _calculate_languages_ratios(text)

    most_rated_language = max(ratios, key=ratios.get)

    return most_rated_language


if __name__ == '__main__':
    files = []
    language_dict = dict()
    mypath = '/media/disk2/datasets/moralchoicemachine/data/books/XML_FILES/1800_1809/parsed_data'
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        files.extend(filenames)
    for file_path in tqdm(files):
        f = open(os.path.join(mypath, file_path), "r")
        text = f.read()

        language = detect_language(text)
        if language not in list(language_dict.keys()):
            language_dict[language] = 1
        else:
            language_dict[language] += 1

    for language in list(language_dict.keys()):
        print(language, language_dict[language])
    print('Total number of files:', len(list(language_dict.keys())), "| {} are in english".format(language_dict['english']))
