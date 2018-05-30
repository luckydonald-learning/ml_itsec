import os
from os import path
import numpy
from numpy.linalg import norm
from luckydonaldUtils.logger import logging

logger = logging.getLogger(__name__)
logging.add_colored_handler(level=logging.DEBUG, date_formatter='%Y-%m-%d %H:%M:%S')


from exec3 import split_word, do_count, calc_k, merge_alphabets


FOLDER = path.join('..','..',"Machine Learning for Computer Security", "Exercises", "mlsec-exer04-data")


TRAIN_DATA_FOLDER = path.join(FOLDER, 'spam-train')
TEST_DATA_FOLDER = path.join(FOLDER, 'spam-test')
CACHE_FILE_WORDS = path.join(FOLDER, 'cache-words.json')
CACHE_FILE_MORE = path.join(FOLDER, 'cache-more.json')

DELIMITERS_D = [' ', '\n']  # space


def list_files(dir=TRAIN_DATA_FOLDER):
    spam_files = []
    good_files = []
    for f in os.listdir(dir):
        if f.endswith(".ham.txt"):
            good_files.append(f)
        elif f.endswith(".spam.txt"):
            spam_files.append(f)
        else:
            logger.error("file {} neiter ham nor spam.".format(f))
        # end if
    # end for
    return spam_files, good_files
# end def


def get_words(folder, filename_list, delemiters=[' '], per_file=False, omit_words=[]):
    """
    :param per_file: False: Will yield every word in all files. True: Will yield a list of words for each file.
        So either it is a continuous stream of words, or they are grouped into lists by files.
    returns list of words in file
    """
    for file in filename_list:
        file = path.join(folder, file)
        if per_file:
            words_of_file = []
        # end if
        with open(file, 'r', errors='replace') as f:
            try:
                for line in f.readlines():
                    words = split_word(line, delemiters, strip=True, omit_words=omit_words)
                    if per_file:
                        words_of_file.extend(words)
                    else:
                        yield from words
                    # end if
                # end for
            except UnicodeDecodeError:
                print("UnicodeDecodeError: {}".format(file))
            # end try
        # end with
        if per_file:
            yield words_of_file
        # end if
    # end for
# end def


def center_of_mass(alphabet, counts):
    n = len(alphabet)
    return (1 / n) * sum(counts[alphabet[i]] for i in range(0, n))
# end def


z_de = """
Subject: Wir machen dich mit Schönheiten bekannt
Ich habe erfahren , dass mein Mann fremdgeht!
Nun will ihm mit gleicher Münze zahlen !
Suchen nach einem netten Kerl für Sex .
Hier sind meine Fotos und Profil ."""

z_en = """
Subject: EMMANUEL SAMS & ASSOCIATES
EMMANUEL SAMS & ASSOCIATES
SOLICITORS AND ADVOCATES
214 / GRE APPAPA LAGOS NIGERIA.
POST CODE 23401
PRIVATE EMAIL : ( emmanuelsams 59 @ yahoo . com )

Complements ! ! . firstly , I must apologize for barging into your mailbox without a formal introduction of myself to you .

Actually , I got your contact information from a reputable business / professional directory of your country which gives me

assurance of your legibility as a person while trying to get a good and capable business person in your country for both business and investment purposes . Let me start by introducing myself ; I am Barrister EMMANUEL SAMS   ( Esq . ) I was the Personal Attorney to late Mr . Morris Thompson an American who was a private businessman in my country unfortunately lost his life in the plane crash of Alaska Airlines Flight 261 which crashed on January 31 st 2000 , including his wife and only daughter .

Before the plane crash , Mr . Morris Thompson made a deposit value of US $ 20 , 000 , 000 . 00 ( TWENTY MILLION UNITED STATE DOLLARS ) in the STANDARDTRUST BANK OF NIGERIA  Upon maturity several notice was sent to him , another notification was sent and still no response came from him . I later found out that Mr . Morris Thompson and his family had been killed in that plane crash .

After further investigation it was also discovered that Mr . Morris Thompson ’ s next of kin was his daughter who died with him in the crash . What borders me most is that according to the laws of my country at the expiration of 8 years the funds will revert to the ownership of the NIGERIA GOVERNMENT , if nobody applies to claim the funds . Against this backdrop , my suggestion to you is that I will like you as a
foreigner to stand as the next of kin to Mr . Morris Thompson so that you will be able to receive these funds for both of us . WHAT IS TO BE DONE ?

I want you to know that I have had everything planned out so that we shall come out successfully . As a barrister , I will prepare the necessary document that will back you up as the next of kin to Mr . Morris Thompson . PLEASE YOU HAVE TO PROVIDE YOUR FULL INFORMATION SUCH AS STATED BELOW ,

[ 1 ] NAME :
[ 2 ] ADDRESS :
[ 3 ] AGE :
[ 4 ] SEX :
[ 5 ] TELEPHONE :
[ 6 ] FAX :
[ 7 ] OCCUPATION STATUS :

Also be informed that this Transaction will take us just 15 working days to accomplish beginning from when I receive your Data ’ s . After you have been made the next of kin , I will also file an application to the bank on your behalf as soon as I secure the necessary approval and letter of probate in your favor for the movement of the funds to an account that will be provided by you . This process is 100 % risk free as I have set out all the modalities to see that a legalized method is used because then I will prepare all the necessary documents .

Please note that utmost secrecy and confidentiality is required at all times during this transaction .

Once the funds have been transferred into your nominated bank account we shall share in the ratio of 55 % for me , 45 % for you . I will prefer that you reach me via this email address : ( emmanuelsams59 @ yahoo . com )

Your earliest response to this offer will be appreciated

Best Regards
Barr . EMMANUEL SAMS"""

z_nospam = """Liebe Studis ,

mich hat der Hinweis erreicht , dass die Dokumentation zur Template
Engine Pug nirgends auf dem Aufgabenblatt verlinkt ist . Stimmt ! Darum
per Mail :

https : / / pugjs . org

David

- - 
David Goltzsche , M . Sc .
Institute of Operating Systems and Computer Networks
Distributed Systems Group
TU Braunschweig

www : https : / / www . ibr . cs . tu - bs . de / users / goltzsch
mail : goltzsche @ ibr . cs . tu - bs . de
tel : + 49 531 391 3249
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Web - sys mailing list
Web - sys @ ibr . cs . tu - bs . de
http : / / mail . ibr . cs . tu - bs . de / listinfo / web - sys"""


import matplotlib.pyplot as plt
import numpy as np

# This is the ROC curve
#plt.plot(x,y)
#plt.show()

# This is the AUC
#auc = np.trapz(y,x)





import math
import os
import re
import sys
from collections import Counter

def bowk(x, z, d=1.0, normalize=True):
    """
    Bag Of Words
    :param x:
    :param z:
    :param d:
    :param normalize:
    :type  normalize: bool
    :return:
    """
    # {'word': 3}   key = word, value = count
    result = 0.0
    for word in (x if len(x) < len(z) else z):  # iterate through the smaller set
        # because if the word is in x and not in z, we would multiply with 0, so we can skip them.
        # the bigger list will have elements
        # if word not in smaller_list: continue
        result += x[word] * z[word]
    # end for
    if normalize:
        kxx = bowk(x, x, d, normalize=False)
        kzz = bowk(z, z, d, normalize=False)
        result /= math.sqrt(kxx * kzz)
        # k'(x,z) = k(x
    # end if
    return result ** d
# end def


def kmat(X, Y, k=bowk):
    n = len(X)
    m = len(Y)
    mat = np.empty((n,m))
    for i in range(n):  # file_a
        for j in range(m):  # file_b
            xi = X[i]
            yj = Y[j]
            mat[i, j] = k(xi, yj)
        # end for
    # end for
    return mat
# end def


def bowk_pair(X, Z, d=1.0): # TODO: deleteme
    result = 0.0
    for x in X:
        for z in Z:
            result += bowk(x, z, d)
        # end for
    # end for
    return result
# end def


class Classifier(object):
    def __init__(self, k):
        self.cache = load_cache(CACHE_FILE_MORE)
        self._k_func = k
    # end def

    def train(self, ham, spam):
        """ calculates the center of mass for ham and spam.
        """
        self._ham_n = len(ham)
        self._spam_n = len(spam)

        self._ham = ham
        self._spam = spam

        self._c_ham = self._calc_if_not_cached('c_o_m_ham', self._our_kmat(ham, ham))
        """ the center of mass for ham"""

        self._c_spam = self._calc_if_not_cached('c_o_m_ham', self._our_kmat(spam, spam))
        """ the center of mass for spam"""

        logger.debug('center of mass (ham): {ham!r}\ncenter of mass (spam): {spam!r}'.format(
            ham=self._c_ham, spam=self._c_spam
        ))

    def _calc_if_not_cached(self, key, func):
        if key in self.cache and self.cache[key]:
            logger.debug('loaded the center of mass for {}'.format(key))
            value = self.cache[key]
        else:
            logger.debug('calculating the center of mass for {}'.format(key))
            value = func(self._k_func)
            self.cache[key] = value
            write_cache(self.cache, CACHE_FILE_MORE)
        # end if
        return value
    # end def

    def _our_kmat(self, X, Y):
        def _kmat_inner():
            return kmat(X, Y, self._k_func).mean()
        # end def
        return _kmat_inner
    # end def

    def classify(self, message, mode='classic', threshold=0.0):
        """
        :return: Tuple with 'ham'/'spam' as first element, and the value as second.
        :rtype: tuple(str, float)
        """
        if mode == 'classic':
            value = self._classic(message)
        elif mode == 'reverse':
            value = self._reverse(message)
        elif mode == 'simple':
            value = self._simple(message)
        else:
            raise ValueError('invalid mode: {}'.format(mode))
        # end if
        return (ham_or_spam(value, threshold), value)

    def _classic(self, message):
        """
        Learning ham (non-spam)
        :param message:
        :return:
        """
        a = self._k_func(message, message)
        b = kmat([message], self._ham, self._k_func).mean()
        c = self._c_ham
        return a - (b + b) + c

    def _reverse(self, message):
        """
        same as _classic, but using spam instead of ham
        :param message:
        :return:
        """
        a = self._k_func(message, message)
        b = kmat([message], self._spam, self._k_func).mean()
        c = self._c_spam
        return -(a - (b + b) + c)

    def _simple(self, message):
        return self._classic(message) + self._reverse(message)
    # end def


### Prerocessing
import nltk
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
# end try


def word_filter(word):
    return word not in stopwords.words('english')
# end def


def read_messages(directory, suffix='.spam.txt'):
    for filename in filter(lambda x: x.endswith(suffix), os.listdir(directory)):
        path = os.path.join(directory, filename)
        logger.debug('Loading message {:s}'.format(path))
        with open(path, encoding='latin-1') as f:
            message = f.read()
            X = Counter(filter(word_filter, re.split("\W+", message)))
            yield X
        # end with
    # end for
# end def


import json
def write_cache(data, file):
    logger.info('writeing cache file to {}.'.format(path.abspath(file)))
    with open(file, 'w') as f:
        json.dump(data, f)
        logger.info('cache file written to {}.'.format(path.abspath(file)))
    # end with
# end def

def load_cache(file):
    with open(file, 'r') as f:
        cache = json.load(f)
    # end with
    logger.info('cache file loaded, parsing data.')
    return cache
# end def


# load / write cache
if not os.path.exists(CACHE_FILE_WORDS):
    logger.info('no cache file found, generating data.')
    ham_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.spam.txt'))

    ham_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.spam.txt'))

    cache = {
        'train_ham': ham_messages_train, 'train_spam': spam_messages_train,
        'test_ham': ham_messages_test, 'test_spam': spam_messages_test,
    }
    logger.debug('writing cache file...')
    write_cache(cache, CACHE_FILE_WORDS)
else:
    logger.info('cache file found.')
    cache = load_cache(CACHE_FILE_WORDS)

    ham_messages_train = [Counter(message) for message in cache['train_ham']]
    spam_messages_train = [Counter(message) for message in cache['train_spam']]

    ham_messages_test = [Counter(message) for message in cache['test_ham']]
    spam_messages_test = [Counter(message) for message in cache['test_spam']]

    logger.info('cache file parsed.')
# end if

def ham_or_spam(score, threshold=0.5):
    return 'ham' if score <= threshold else 'spam'
# end def


#for d in [1, 2, 3, 4]:
for d in [1]:
    logger.debug('calculating for d={}'.format(d))
    def k(x, y):
        return bowk(x, y, d, True)
    # end def
    classifier = Classifier(k)
    logger.debug('training ham.')
    classifier.train(ham_messages_train, spam_messages_train)
    # for mode in ['classic', 'reverse', 'simple']:
    for mode in ['classic', 'reverse', 'simple']:
        logger.debug('calculating for mode={!r}'.format(mode))
        labels = []
        scores = []
        logger.debug('classifying ham tests')
        for i, message in enumerate(ham_messages_test):
            labels.append('ham')
            _, score = classifier.classify(message, mode=mode)
            logger.debug('classifyed ham test {}: {} ({})'.format(i, score, _))
            scores.append(score)
        # end for
        logger.debug('classifying spam tests')
        for i, message in enumerate(spam_messages_test):
            labels.append('spam')
            _, score = classifier.classify(message, mode=mode)
            logger.debug('classifyed spam test {}: {} ({})'.format(i, score, _))
            scores.append(score)
        # end for
        for threshold in sorted(scores, reverse=True)[::100]:

            fp = 0
            """ false positives """

            tp = 0
            """ true positives """

            tn = 0
            """ true negatives """

            fn = 0
            """ false negatives"""

            for score, label in zip(scores, labels):
                predicted_label = ham_or_spam(score, threshold)
                if (label, predicted_label) == ('ham', 'ham'):
                    tn += 1
                elif (label, predicted_label) == ('ham', 'spam'):
                    fp += 1
                elif (label, predicted_label) == ('sham', 'ham'):
                    fn += 1
                elif (label, predicted_label) == ('ham', 'ham'):
                    tp += 1
                # end if
            # end for
            logging.success('true negative: {tn}\n false positive: {fp}\n false negative: {fn}\n true positive {tp}'.format(
                tn=tn, fp=fp, fn=fn, tp=tp
            ))
        # end for
    # end for
# end def

