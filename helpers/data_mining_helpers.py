import nltk

"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""


def _get_doc_iterable(docs):
    """Return an iterable over document strings regardless of container type."""
    if hasattr(docs, "data"):
        return docs.data
    return docs


def format_rows(docs):
    """Format text rows and strip newline / tab characters for DataFrame loading."""
    D = []
    for d in _get_doc_iterable(docs):
        temp_d = " ".join(str(d).split("\n")).strip('\n\t')
        D.append([temp_d])
    return D


def format_labels(target, docs):
    """Return a human-readable label using helper metadata from docs or mappings."""
    if isinstance(docs, dict):
        return docs.get(target, target)

    target_names = getattr(docs, "target_names", None)

    if target_names is None:
        return target

    doc_targets = getattr(docs, "target", None)
    if doc_targets is not None:
        unique_targets = list(dict.fromkeys(sorted(set(doc_targets))))
        if len(unique_targets) == len(target_names):
            mapping = dict(zip(unique_targets, target_names))
            return mapping.get(target, target)

    try:
        index = int(target)
        return target_names[index]
    except (TypeError, ValueError, IndexError):
        return target


def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter += 1
    return ("The amoung of missing records is: ", counter)


def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens
