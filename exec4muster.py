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
from collections import Counter


def bowk(x, z, d=1.0, normalize=True):
    """
    Bag Of Words

    The values `x` and `z` are word counts in a mail,
    a dict of format `{'word': 3}`,
    with the key bing a word and the value the count.

    :type x: dict
    :type z: dict
    """
    # {'word': 3}   key = word, value = count
    result = 0.0
    for word in (x if len(x) < len(z) else z):  # iterate through the smaller set
        # because if the word is in x and not in z, we would multiply with 0, so we can skip them.
        # the bigger list will have elements
        # if word not in smaller_list: continue
        result += x[word] * z[word]
    # end for
    result **= d  # TODO: `result ** d` correct here, or at the return below?
    if normalize:
        kxx = bowk(x, x, d, normalize=False)
        kzz = bowk(z, z, d, normalize=False)
        result /= math.sqrt(kxx * kzz)
    # end if
    return result # TODO: `result ** d` here?
# end def

def bow_k(x, z, d=1.0):
    """
    Bag Of Words

    The values `x` and `z` are word counts in a mail,
    a dict of format `{'word': 3}`,
    with the key bing a word and the value the count.

    :type x: dict
    :type z: dict
    """
    result = 0.0
    for word in (x if len(x) < len(z) else z):  # iterate through the smaller set
        # because if the word is in x and not in z, we would multiply with 0, so we can skip them.
        # the bigger list will have elements
        # if word not in smaller_list: continue
        result += x[word] * z[word]
    # end for
    return result ** d
# end def

def normalized_bow_k(x, z, d=1.0):
    """
    Bag Of Words

    The values `x` and `z` are word counts in a mail,
    a dict of format `{'word': 3}`,
    with the key bing a word and the value the count.

    :type x: dict
    :type z: dict
    """
    kxz = bow_k(x, z, d)
    kxx = bow_k(x, x, d)
    kzz = bow_k(z, z, d)

    return kxz / math.sqrt(kxx * kzz)
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


class Classifier(object):
    def __init__(self, k, cache, cache_key):
        """

        :param k:
        :param cache:
        :type  cache: Cache
        :param cache_key:
        :type  cache_key: str[]
        """
        self._cache = cache
        self._k_func = k
        self._cache_key = cache_key
    # end def

    def set_cache_key(self, array):
        self._cache_key = array
    # end def

    def train(self, ham, spam):
        """ calculates the center of mass for ham and spam.
        """
        self._ham_n = len(ham)
        self._spam_n = len(spam)

        self._ham = ham
        self._spam = spam

        self._c_ham = self._calc_if_not_cached('c_o_m_ham', self._our_kmat(ham, ham), save=True)
        """ the center of mass for ham"""

        self._c_spam = self._calc_if_not_cached('c_o_m_spam', self._our_kmat(spam, spam), save=True)
        """ the center of mass for spam"""
        self._cache.save()

        logger.debug('center of mass (ham): {ham!r}\ncenter of mass (spam): {spam!r}'.format(
            ham=self._c_ham, spam=self._c_spam
        ))

    def _calc_if_not_cached(self, key, func, *args, save=True, **kwargs):
        """
        :param key: Key for the current value
        :param func:
        :return:
        """
        cache = self._cache
        for access_key in self._cache_key:
            if access_key not in cache:
                cache[access_key] = dict()
            # end def
            cache = cache[access_key]
        # end if
        if key in cache and cache[key]:
            logger.debug('loaded {} from cache.'.format(key))
            value = cache[key]
        else:
            logger.debug('calculating value for {}.'.format(key))
            value = func(*args, **kwargs)
            cache[key] = value
        # end if
        if save:
            self._cache.save()
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
        value = self._calc_if_not_cached('classify', self._classify, message, save=False)
        return (ham_or_spam(value, threshold), value)
    # end def

    def _classify(self, message, ):
        if mode == 'classic':
            value = self._classic(message)
        elif mode == 'reverse':
            value = self._reverse(message)
        elif mode == 'simple':
            value = self._simple(message)
        else:
            raise ValueError('invalid mode: {}'.format(mode))
        # end if
        return value

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
    if word in stopwords.words('english'):
        return False
    # end if
    if word == '':
        return False
    # end if
    return
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


from DictObject import DictObject

def write_cache(data, file):
    logger.info('writing cache file to {}.'.format(path.abspath(file)))
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

class Cache(DictObject):
    def __init__(self, file, do_load=False, *args, **kwargs):
        self.__file = file
        super().__init__(*args, **kwargs)
        if do_load:
            self.load()
        # end if
    # end def

    def save(self):
        logger.info('writing cache file to {}.'.format(path.abspath(self.__file)))
        data = dict(self)
        with open(self.__file, 'w') as f:
            json.dump(data, f)
            logger.info('cache file written to {}.'.format(path.abspath(self.__file)))
        # end with
    # end def

    def read_from_disk(self):
        with open(self.__file, 'r') as f:
            cache = json.load(f)
        # end with
        logger.info('cache file loaded from {}.'.format(path.abspath(self.__file)))
        return cache
    # end def

    def load(self):
        new_data = self.read_from_disk()
        for old_key in self.keys():
            del self[old_key]
        # end for
        self.merge_dict(new_data)
        logger.info('cache file merged.')
    # end def
# end class



import json

# load / write cache
words_cache = Cache(CACHE_FILE_WORDS)
if not os.path.exists(CACHE_FILE_WORDS):
    logger.info('no cache file found, generating -data.')
    ham_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.spam.txt'))

    ham_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.spam.txt'))

    words_cache = {
        'train_ham': ham_messages_train, 'train_spam': spam_messages_train,
        'test_ham': ham_messages_test, 'test_spam': spam_messages_test,
    }
    logger.debug('writing cache file...')
    write_cache(words_cache, CACHE_FILE_WORDS)
else:
    logger.info('cache file found.')
    words_cache = load_cache(CACHE_FILE_WORDS)

    ham_messages_train = [Counter(message) for message in words_cache['train_ham']]
    spam_messages_train = [Counter(message) for message in words_cache['train_spam']]

    ham_messages_test = [Counter(message) for message in words_cache['test_ham']]
    spam_messages_test = [Counter(message) for message in words_cache['test_spam']]

    logger.info('cache file parsed.')
# end if

def ham_or_spam(score, threshold=0.5):
    return 'ham' if score <= threshold else 'spam'
# end def


#for d in [1, 2, 3, 4]:
cache = Cache(CACHE_FILE_MORE, do_load=os.path.exists(CACHE_FILE_MORE))

for d in [1, 2, 3, 4]:
    logger.debug('calculating for d={}'.format(d))
    def k(x, y):
        return normalized_bow_k(x, y, d)
        # return bowk(x, y, d, normalize=True)
    # end def

    classifier = Classifier(k, cache=cache, cache_key=['d_{}'.format(d)])
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
            classifier.set_cache_key(['d_{}'.format(d), 'mode_{}'.format(mode), 'ham', str(i)])
            _, score = classifier.classify(message, mode=mode)
            logger.info('classifyed ham test {}: {}'.format(i, score))
            scores.append(score)
        # end for
        cache.save()
        logger.debug('classifying spam tests')
        for i, message in enumerate(spam_messages_test):
            labels.append('spam')
            classifier.set_cache_key(['d_{}'.format(d), 'mode_{}'.format(mode), 'spam', str(i)])
            _, score = classifier.classify(message, mode=mode)
            logger.info('classifyed spam test {}: {}'.format(i, score))
            scores.append(score)
        # end for
        cache.save()
        max_count = len(scores)
        tps = []
        fps = []
        for threshold in sorted(scores, reverse=True):

            fp = 0
            """ false positives 
            nein, is kein spam, falsch als spam erkannt
            """

            tp = 0
            """ true positives 
            ja, is spam, richtig erkannt
            """

            tn = 0
            """ true negatives
            ja, ist spam, richtig erkannt
            """

            fn = 0
            """ false negatives
            nein, ist spam, falsch als spam erkannt
            """

            false_count = 0
            true_count = 0

            for score, label in zip(scores, labels):
                predicted_label = ham_or_spam(score, threshold)
                if (label, predicted_label) == ('ham', 'ham'):
                    tn += 1
                    false_count += 1
                elif (label, predicted_label) == ('ham', 'spam'):
                    fp += 1
                    false_count += 1
                elif (label, predicted_label) == ('spam', 'ham'):
                    fn += 1
                    true_count += 1
                elif (label, predicted_label) == ('spam', 'spam'):
                    true_count += 1
                    tp += 1
                else:
                    raise ValueError('dafuq: {!r}'.format((label, predicted_label)))
                # end if
            # end for
            logging.success(' threshold: {t}\n true negative: {tn}\n false positive: {fp}\n false negative: {fn}\n true positive {tp}'.format(
                t=threshold, tn=tn, fp=fp, fn=fn, tp=tp
            ))
            tps.append(tp/true_count)
            fps.append(fp/false_count)
        # end for

        import matplotlib.pyplot as plt
        import numpy as np

        # https://stackoverflow.com/a/25009504/3423324
        # This is the ROC curve
        plt.plot(fps, tps)
        plt.show()

        pass

        # This is the AUC (area under curve)
        # auc = np.trapz(y, x)
    # end for
# end def

