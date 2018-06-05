from luckydonaldUtils.logger import logging
from matplotlib import pyplot as plt
from DictObject import DictObject
from collections import Counter
import numpy as np
import math
import nltk
import json
import os
import re

from exec3 import words_omit
from exec4 import list_files


# package related inits
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
# end try

logger = logging.getLogger(__name__)
logging.add_colored_handler(level=logging.DEBUG, date_formatter='%Y-%m-%d %H:%M:%S')


# settings

FOLDER = os.path.join('..','..',"Machine Learning for Computer Security", "Exercises", "mlsec-exer04-data")

TRAIN_DATA_FOLDER = os.path.join(FOLDER, 'spam-train')
TEST_DATA_FOLDER = os.path.join(FOLDER, 'spam-test')
CACHE_FILE_WORDS = os.path.join(FOLDER, 'cache-words.json')
CACHE_FILE_CLASSIFIER = os.path.join(FOLDER, 'cache-classify.json')

DELIMITERS_D = [' ', '\n']  # space

s0 = nltk.stem.PorterStemmer()
s1 = nltk.stem.SnowballStemmer('english')

def get_words(filename_list, delemiters=[' '], omit_words=[]):
    """
    :param per_file: False: Will yield every word in all files. True: Will yield a list of words for each file.
        So either it is a continuous stream of words, or they are grouped into lists by files.
    returns list of words in file
    """
    for file in filename_list:
        words_of_file = []
        # end if
        logger.debug('Loading message {:s}.'.format(file))
        with open(file, 'r', encoding='latin-1', errors='replace') as f:
            message = f.read()
            # Replace the "Subject: " which is the start of every mail.
            message.replace('Subject: ', '', 1)
            # Split at non-word characters
            words = words_omit(re.split("\W+", message), '')
            words = (w.lower() for w in words)
            words = filter(word_filter, words)
            words = (s0.stem(s1.stem(w)) for w in words)
            words = filter(word_filter, words)
            X = Counter(words)
            yield X
        # end with
    # end for
# end def





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

        self._c_ham = self._calc_if_not_cached('c_o_m_ham', self._kmat_mean(ham, ham))
        self._cache.save_if_changed()
        """ the center of mass for ham"""

        self._c_spam = self._calc_if_not_cached('c_o_m_spam', self._kmat_mean(spam, spam))
        self._cache.save_if_changed()
        """ the center of mass for spam"""

        logger.debug('center of mass (ham): {ham!r}\ncenter of mass (spam): {spam!r}'.format(
            ham=self._c_ham, spam=self._c_spam
        ))

    def _calc_if_not_cached(self, key, func, *args, **kwargs):
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
        return value
    # end def

    def _kmat_mean(self, X, Y):
        def _kmat_inner():
            return kmat(X, Y, self._k_func).mean()
        # end def
        return _kmat_inner
    # end def

    def classify(self, message, mode='simple', threshold=0.0):
        """
        :return: Tuple with 'ham'/'spam' as first element, and the value as second.
        :rtype: tuple(str, float)
        """
        value = self._calc_if_not_cached('classify', self._classify, message, mode=mode)
        return (ham_or_spam(value, threshold), value)
    # end def

    def _classify(self, message, mode):
        """
        :rtype: float
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
# end class


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


def write_cache(data, file):
    logger.info('writing cache file to {}.'.format(os.path.abspath(file)))
    with open(file, 'w') as f:
        json.dump(data, f)
        logger.info('cache file written to {}.'.format(os.path.abspath(file)))
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
        self.__unsaved_changes = False
        super().__init__(*args, **kwargs)
        if do_load:
            self.load()
        # end if

    def on_set(self, key, value_to_set):
        if not self.__unsaved_changes:
            logger.debug('now marked unsaved.')
        # end if
        self.__unsaved_changes = True
        return super().on_set(key, value_to_set)

    def on_del(self, key):
        if not self.__unsaved_changes:
            logger.debug('now marked unsaved.')
        # end if
        self.__unsaved_changes = True
        return super().on_del(key)
    # end def

    def save(self):
        logger.info('writing cache file to {}.'.format(os.path.abspath(self.__file)))
        data = dict(self)
        with open(self.__file, 'w') as f:
            json.dump(data, f)
            logger.info('cache file written to {}.'.format(os.path.abspath(self.__file)))
        # end with
        if self.__unsaved_changes:
            logger.debug('now marked saved.')
        # end if
        self.__unsaved_changes = False
    # end def

    def save_if_changed(self):
        if self.__unsaved_changes:
            logger.debug('will save changes.')
            self.save()
        else:
            logger.debug('has no unsaved changes.')
        # end if
    # end def

    def read_from_disk(self):
        with open(self.__file, 'r') as f:
            cache = json.load(f)
        # end with
        logger.info('cache file loaded from {}.'.format(os.path.abspath(self.__file)))
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



# load / write cache
words_cache = Cache(CACHE_FILE_WORDS)
if not os.path.exists(CACHE_FILE_WORDS):
    logger.info('no cache file found, generating word count data.')
    spam_files, good_files = list_files(TRAIN_DATA_FOLDER)
    ham_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_train = list(read_messages(TRAIN_DATA_FOLDER, suffix='.spam.txt'))

    t_spam_files, t_good_files = list_files(TEST_DATA_FOLDER)
    ham_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.ham.txt'))
    spam_messages_test = list(read_messages(TEST_DATA_FOLDER, suffix='.spam.txt'))

    words_cache = {
        'train_ham': ham_messages_train, 'train_spam': spam_messages_train,
        'test_ham': ham_messages_test, 'test_spam': spam_messages_test,
    }
    logger.debug('writing cache file...')
    write_cache(words_cache, CACHE_FILE_WORDS)
else:
    logger.info('cache file for word count found.')
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


classifier_cache = Cache(CACHE_FILE_CLASSIFIER, do_load=os.path.exists(CACHE_FILE_CLASSIFIER))

# This is the ROC curve
plt.figure()
# diagonal line
plt.plot([0, 1], [0, 1], 'k--')
# scale to 0.0-1.0
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# title
plt.title('Characteristic for spam detection ROC')

for d in [1, 2, 3, 4]:
    logger.debug('calculating for d={}'.format(d))
    def k(x, y):
        return normalized_bow_k(x, y, d)
        # return bowk(x, y, d, normalize=True)
    # end def

    classifier = Classifier(k, cache=classifier_cache, cache_key=['d_{}'.format(d)])
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
        classifier_cache.save_if_changed()
        logger.debug('classifying spam tests')
        for i, message in enumerate(spam_messages_test):
            labels.append('spam')
            classifier.set_cache_key(['d_{}'.format(d), 'mode_{}'.format(mode), 'spam', str(i)])
            _, score = classifier.classify(message, mode=mode)
            logger.info('classifyed spam test {}: {}'.format(i, score))
            scores.append(score)
        # end for
        classifier_cache.save_if_changed()
        max_count = len(scores)
        tps = []
        fps = []
        for threshold in sorted(scores, reverse=True):

            fp = 0
            """ false positives 
            nein, is kein spam, falsch als spam erkannt
            F1 = -1 als  1 erkannt | FalseNeg
            """

            tp = 0
            """ true positives 
            ja, is spam, richtig erkannt
            T 1 =  1 als  1 erkannt | TruePos
            """

            tn = 0
            """ true negatives
            ja, ist kein spam, richtig erkannt
            T-1 = -1 als -1 erkannt | TrueNeg
            """

            fn = 0
            """ false negatives
            nein, ist kein spam, falsch als spam erkannt
            F-1 =  1 als -1 erkannt | FalseNeg
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
            # max_len = max(len(str(i)) for i in (tn,fp,fn,tp))
            # logging.success('threshold: {t}\n'
            #                 'true  negative: {tn:>{len}} ({tn_avg})\n'
            #                 'false positive: {fp:>{len}} ({fp_avg})\n'
            #                 'false negative: {fn:>{len}} ({fn_avg})\n'
            #                 'true  positive: {tp:>{len}} ({tp_avg})'.format(
            #     t=threshold, tn=tn, fp=fp, fn=fn, tp=tp, len=max_len,
            #     tn_avg=tn/false_count, fp_avg=fp/false_count, fn_avg=fn/true_count, tp_avg=tp/true_count
            # ))
            tps.append(tp/true_count)
            fps.append(fp/false_count)
        # end for

        # https://stackoverflow.com/a/25009504/3423324
        # https://stackoverflow.com/a/37113381/3423324

        # This is the AUC (area under curve)
        auc = np.trapz(tps, fps)


        # ROC
        plt.plot(fps, tps, label='{mode} d={d} (area = {auc:0.2f})'.format(d=d, mode=mode, auc=auc))
        plt.legend(loc="lower right")
        # done

        pass
    # end for
# end for

plt.show()

