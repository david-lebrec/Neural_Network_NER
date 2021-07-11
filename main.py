from typing import AnyStr
import pandas as pd
import nltk
import pycrfsuite
import sklearn
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

DATA_DIRECTORY: str = 'data'
TEST_FILE_NAME: str = 'test.txt'
VALID_FILE_NAME: str = 'valid.txt'

"""
    string management
"""
SPACE: str = ' '
EMPTY: str = ''
NEW_LINE: str = '\n'
OPEN_PARENTHESIS: str = '('
END_PARENTHESIS: str = ')'
DOT: str = '.'
COMMA: str = ','
COLONS: str = ':'
SLASH: str = '/'
QUOTE: str = "'"

"""
    document mnagament
"""
DOCUMENT_START: str = '-DOCSTART-'
LINE_START: str = '- : O O'
LINE_END: str = '. . O O'

"""
    data files columns
"""
INDEX_WORD: int = 0
INDEX_POST_TAG: int = 2
INDEX_ENTITY: int = 3


class DocumentsHelper:

    def remove_last_char(self, value: str) -> str:
        return value.rstrip(self.get_last_char(value))

    def get_last_char(self, value: str) -> str:
        if len(value) > 0:
            return value[-1]
        return ''

    def is_parenthesis(self, value: str) -> bool:
        return value == OPEN_PARENTHESIS or value == END_PARENTHESIS

    def is_special_char_without_left_space(self, value: str) -> bool:
        return value == COMMA or value == DOT or value == SLASH or value == QUOTE or value == END_PARENTHESIS

    def is_special_char_without_right_space(self, value: str) -> bool:
        return value == OPEN_PARENTHESIS or value == SLASH

    def clean_string(self, value: str) -> str:
        return value.replace(NEW_LINE, EMPTY).replace('\'', "'")

    def is_string_start_line(self, line: str) -> bool:
        return line.__contains__(LINE_START)

    def is_string_end_line(self, line: str) -> bool:
        return line.__contains__(LINE_END)

    def is_string_not_start_or_end_line(self, line: str) -> bool:
        return not (self.is_string_start_line(line) or self.is_string_end_line(line))

    def is_string_not_document_start(self, line: str) -> bool:
        return not line.__contains__(DOCUMENT_START)

    def is_string_not_empty_or_none(self, value: str) -> bool:
        return not (value is None or value == '')

    def is_valid_token(self, line: str) -> bool:
        return self.is_string_not_empty_or_none(line) and self.is_string_not_document_start(
            line) and self.is_string_not_start_or_end_line(line)

    def is_documents_property_empty(self, documents: list) -> bool:
        return self.get_documents_size(documents) == 0

    def get_documents_size(self, documents: list) -> int:
        return len(documents)


def load(file_name: str):
    doc_lines = []
    h = DocumentsHelper()
    file = open(file_name, 'r')
    lines: [AnyStr] = file.readlines()

    for line in lines:
        split_line: [str] = line.split(SPACE)
        word: str = split_line[INDEX_WORD]
        try:
            if h.is_string_start_line(line):
                doc_line = []
            if word != NEW_LINE and not h.is_string_end_line(line) and not h.is_string_start_line(line):
                entity: str = split_line[INDEX_ENTITY]
                post_tag: str = split_line[INDEX_POST_TAG]
                if not entity.__contains__('I-'):
                    line_component = (word, post_tag, entity)
                    doc_line.append(line_component)
            if h.is_string_end_line(line):
                doc_lines.append(doc_line)
        except Exception as e:
            pass
    return doc_lines


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def get_X_Train():
    X_Tr = []
    for s in train_sents:
        X_Tr[s] = sent2features(s)
    return X_Tr
    # X_train = [sent2features(s) for s in train_sents]


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))


if __name__ == "__main__":
    valid_data = load(VALID_FILE_NAME)
    TRAIN_FILE_NAME = 'train.txt'
    train_sents = load(TRAIN_FILE_NAME) + valid_data
    test_sents = load(TEST_FILE_NAME)

    nltk.download('conll2002')
    print(sklearn.__version__)
    nltk.corpus.conll2002.fileids()

    # train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    # test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    print(train_sents[0])
    print(test_sents[0])

    print(sent2features(train_sents[0])[0])

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.params()

    trainer.train('conll2002-en.crfsuite')

    print(trainer.logparser.last_iteration)

    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

    tagger = pycrfsuite.Tagger()
    tagger.open('conll2002-en.crfsuite')

    example_sent = test_sents[0]
    print(' '.join(sent2tokens(example_sent)), end='\n\n')

    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))

    y_pred = [tagger.tag(xseq) for xseq in X_test]

    print(bio_classification_report(y_test, y_pred))

    info = tagger.info()

    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-15:])

    ' '.split('-', 1)

    print("Top positive:")
    print_state_features(Counter(info.state_features).most_common(20))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-20:])