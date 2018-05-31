import os
from os import path


def split_word(word, delimiters, strip=True, omit_words=[]):
    """
    Splits a text by delemiters, which must be a list
    :param word:
    :type  word: str

    :param delimiters:
    :type  delimiters: list of str

    :param strip:
    :type  strip: bool

    :param omit_words: Words which should be ignored.
    :type  omit_words: list of str

    :return:
    """
    if strip:
        word = word.strip()
    # end if
    if len(delimiters) <= 0:
        yield word
        return
    # end if
    d = delimiters[0]
    words = word.split(d)
    for w in words:
        if w == "":  # skip empty splits
            continue
        # end if
        if w in omit_words:
            continue
        # end if
        yield from split_word(w, delimiters[1:])
    # end for
# end def


def count_in(search_word, text):
    counter = 0
    for word in text:
        if word == search_word:
            counter += 1
        # end if
    # end for
    return counter
# end def

sentences = [
     "They call it a Royale with cheese.",
     "A Royale with cheese. What do they call a Big Mac?",
     "Well, a Big Mac is a Big Mac, but they call it le Big-Mac.",
     "Le Big-Mac. Ha ha ha ha. What do they call a Whopper?"
 ]


def get_stuff(sentence):
    return do_count(split_word(sentence, [' ', ',', '.', '?']))
# end def


def do_count(word_generator):
    """
    :param sentence:
    :return: count bag. Keys are the alphabet, value the count.  `{"foo": 12}`
    :rtype: dict
    """
    bagsofwords = {}

    for word in word_generator:
        if word not in bagsofwords:
            bagsofwords[word] = 1
        else:
            bagsofwords[word] += 1
        # end if
    # end for
    return bagsofwords
# end for


def calc_k(counts_x, counts_z, alphabet, d=1):
    k = sum(counts_x[w] * counts_z[w] for w in alphabet) ** d
    # kernel k = ( ∑( #w(x) • #w(z) ) ) ^ d
    return k
# end def


def merge_alphabets(counts_x, counts_z):
    alphabet = set()
    # fill in count of missing values
    for word in counts_x.keys():
        alphabet.add(word)
        counts_z.setdefault(word, 0)
    # end for
    for word in counts_z.keys():
        alphabet.add(word)
        counts_x.setdefault(word, 0)
    # end for
    return alphabet, counts_x, counts_z
# end def


def calc_k_matrix(sentences, d):
    matrix = []
    for x in sentences:
        foo = []
        counts_x = get_stuff(x)
        matrix.append(foo)
        for z in sentences:
            counts_z = get_stuff(z)
            # collect alphabet
            alphabet, counts_x, counts_z = merge_alphabets(counts_x, counts_z)
            k = calc_k(counts_x, counts_z, alphabet, d)
            foo.append(k)
        # end for
    # end for
    return matrix
# end def


def main():
    for d in [1, 2, 3, 4]:
        print(calc_k_matrix(sentences, d))
    # end for
# end def


# "They call it a Royale with cheese.",
# "A Royale with cheese. What do they call a Big Mac?",
#  => 'call', 'a', 'Royale', 'with', 'cheese'
#  =>  1|1    1|1   1|1       1|1     1|1   (count in x|z)
# ==>  1*1 + 1*1 + 1*1 + 1*1 + 1*1  = 5

# "A Royale with cheese. What do they call a Big Mac?",
# "Well, a Big Mac is a Big Mac, but they call it le Big-Mac.",
#  => 'they', 'call', 'a', 'Big', 'Mac'
#  =>  1|1    1|1   1|2    1|2     1|2   (count in x|z)
# ==>   1*1 + 1*1 + 1*2 +  1*2  +  1*2
# ==> 1+1+2+2+2 = 8


if __name__ == '__main__':
    main()
# end if
