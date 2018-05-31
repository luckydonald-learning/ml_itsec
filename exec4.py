import os
from os import path
import numpy
from numpy.linalg import norm

from exec3 import split_word, do_count, calc_k, merge_alphabets


FOLDER = path.join('..','..',"Machine Learning for Computer Security", "Exercises", "mlsec-exer04-data")


TRAIN_DATA_FOLDER = path.join(FOLDER, 'spam-train')
TEST_DATA_FOLDER = path.join(FOLDER, 'spam-test')

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
            print("ERROR: file {} neiter ham nor spam.".format(f))
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


def compare(spam_files, good_files, text, method=center_of_mass, d=1):
    count_text = do_count(split_word(text, [' ', "\n"]))
    count_spam = do_count(get_words(TRAIN_DATA_FOLDER, spam_files, delemiters=[" "]))
    count_good = do_count(get_words(TRAIN_DATA_FOLDER, good_files, delemiters=[" "]))
    alphabet, count_text, count_spam = merge_alphabets(count_text, count_spam)
    alphabet, count_good, count_spam = merge_alphabets(count_good, count_spam)
    alphabet, count_text, count_good = merge_alphabets(count_text, count_good)
    alphabet.remove('Subject:')
    alphabet = list(alphabet)
    del count_text['Subject:']
    del count_spam['Subject:']
    del count_good['Subject:']

    # 1nμ = n ∑φ(xj).
    # CENTER OF MASS
    center_good = method(alphabet, count_good)
    center_spam = method(alphabet, count_spam)

    # f1(z) = ||φ(z) − μ_h|| ^ 2
    f1 = norm(numpy.array([count_text[w] * center_good for w in alphabet])) ** d
    # f1 =  sum(count_z[w] * center_good for w in alphabet) ** (2*d)

    # f2(z) = ||φ(z) − μ_s|| ^ 2
    f2 = -norm(numpy.array([count_text[w] * center_spam for w in alphabet])) ** d
    #f2 = -sum(count_z[w] * center_spam for w in alphabet) ** (2*d)

    # f3(z) = ||φ(z) − μ_h|| ^ 2 − ||φ(z) − μ_s|| ^ 2
    f3 = f1+f2

    return f1, f2, f3
# end def


def calc_k_matrix_from_counts(count_spam_files):
    matrix = []
    for i, counts_x in enumerate(count_spam_files):
        foo = []
        matrix.append(foo)
        for ii, counts_z in enumerate(count_spam_files):
            # collect alphabet
            alphabet, counts_x, counts_z = merge_alphabets(counts_x, counts_z)
            k = calc_k(counts_x, counts_z, alphabet, d)
            foo.append(k)
            if (ii+1) % 100 == 0:
                print("Col {i} done.".format(i=ii+1))
            # end if
        # end for
        print("Row {i} done.".format(i=i))
    # end for
    return matrix
# end def

if __name__ == '__main__':
    spam_files, good_files = list_files()
    t_spam_files, t_good_files = list_files(TEST_DATA_FOLDER)

    #count_text = do_count(split_word(text, [' ', "\n"]))
    count_spam_files = [do_count(file) for file in get_words(TRAIN_DATA_FOLDER, spam_files, delemiters=[" "], per_file=True, omit_words=['Subject:'])]
    calc_k_matrix_from_counts(count_spam_files)
    for d in [1, 2, 3, 4]:
        # end def
        print(d)
        print(compare(spam_files, good_files, z_en,     method=center_of_mass, d=d))
        print(compare(spam_files, good_files, z_de,     method=center_of_mass, d=d))
