'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Philip Booth
CS 251: Data Analysis and Visualization
Fall 2024
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    word_freq = {}
    num_emails = 0
    
    for root, dir, files in os.walk(email_path):
        for file in files:
            if file == ".DS_Store":
                continue
            else:
                num_emails+=1
            file_path = os.path.join(root, file)
            if ".DS_Store" not in file_path:
                with open(file_path, "r") as file:
                    content = file.readlines()
                    #break txt into fill
                    for lines in content:
                        words = tokenize_words(lines)
                        for word in words:
                            if word in word_freq:
                                word_freq[word] += 1
                            else:
                                word_freq[word] = 1
                                
    return word_freq, num_emails
                    

def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    # tools you need 
    #dict
    # some way to sort the keys (sort by)
    # index the top 200 of those keys
    #word_freq = sorted(word_freq, reverse=True)
    # sorted spits out a tuple so I couldn't use .keys()
    word_freq_sort = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_features]
    keys = [a for a, b in word_freq_sort]
    values = [b for a, b in word_freq_sort]
    
    return keys ,values 


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    feats = []
    class_index = np.zeros(num_emails)
    
    for root, dir, files in os.walk(email_path):
        for index,file in enumerate(files):
            feats_per_email = np.zeros(len(top_words))
            if file == ".DS_Store":
                continue
            else:
                if 'ham' in root:
                    class_index[index] = 1
            file_path = os.path.join(root, file)
            if ".DS_Store" not in file_path:
                with open(file_path, "r") as file:
                    content = file.readlines()
                    #break txt into fill
                    for lines in content:
                        words = tokenize_words(lines) #list of words in a txt
                        for word in words:
                            if word in top_words:
                                feats_per_email[top_words.index(word)] += 1
            feats.append(feats_per_email)
    feats = np.array(feats)
    class_index = abs(class_index - 1)
    return feats, class_index


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    split_index = int(test_prop*y.size)
    
    x_train = features[split_index:]
    y_train = y[split_index:]
    train_inds = inds[split_index:]
    
    x_test = features[:split_index]
    y_test = y[:split_index]
    test_inds = inds[:split_index]
    
    return x_train, y_train, train_inds, x_test, y_test, test_inds


    
    
    


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    (POSSIBLE EXTENSION. NOT REQUIRED FOR BASE PROJECT)

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    pass
