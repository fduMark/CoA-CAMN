import h5py
import numpy as np
import cPickle
import os
from collections import Counter
from collections import defaultdict


def load_data():
    sequence_len = 5
    splited_file_name = '../data/splited_data.pkl'
    saved = False

    if os.path.isfile(splited_file_name) and saved:
        train_data, test_data, maxlen, maxmentioned = cPickle.load(open(splited_file_name))
    else:
        print os.getcwd()
        user_mentioned_dict = get_mentioned_dict()
        user_text_history_dict, user_image_history_dict = get_user_history_dict()
        mentioned_text_history_dict, mentioned_image_history_dict = get_mentioned_history_dict()

        dataset, maxlen, maxmentioned = load_data_and_labels(sequence_len, user_mentioned_dict, user_text_history_dict, user_image_history_dict, mentioned_text_history_dict, mentioned_image_history_dict)

        userlist = list(set([data[0] for data in dataset])) #use set to make list member be unique
        print 'Data size: %d, Max tweet length: %d' %(len(dataset), maxlen)
        print 'Max mentioned count: %d' % maxmentioned

        np.random.seed(12345)
        np.random.shuffle(userlist)
        np.random.shuffle(dataset)
        split_index = int(len(userlist)*0.8)
        train_user_id = set(userlist[:split_index])
        test_uset_id = set(userlist[split_index:])

        temp_data = [data for data in dataset if data[0] in train_user_id]
        np.random.shuffle(temp_data)
        split_index = int(len(temp_data)*0.8)
        train_data = temp_data[:split_index]
        valid_data = temp_data[split_index:]
        test_data = defaultdict(list)
        for data in dataset:
            if data[0] in test_uset_id:
                test_data[data[0]].append(data)

    vocabulary, vocabulary_inv, user_vocabulary, user_vocabulary_inv = build_vocab(train_data, valid_data, test_data)
    train_dataset = build_train_data(train_data, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned)
    valid_dataset = build_valid_data(valid_data, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned)
    test_dataset = build_test_data(test_data, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned)

    '''y=[1,0] means match success
       y=[0,1] means match fail
    '''

    return [train_dataset, valid_dataset, test_dataset,\
            vocabulary, vocabulary_inv, \
            user_vocabulary, user_vocabulary_inv,\
            sequence_len, maxlen, maxmentioned]

def get_mentioned_dict():
    user_mentioned_dict = defaultdict(list)
    mentioned_reader = open('../Dataset/user_mentioned.txt', 'r')

    for line in mentioned_reader:
        line = line.replace('\n', ' ').strip().split('\t')
        uid = line[0]
        mentioned = line[1:]
        user_mentioned_dict[uid] = mentioned
    mentioned_reader.close()
    return user_mentioned_dict

def get_user_history_dict():
    user_text_history_dict = defaultdict(list)
    user_image_history_dict = defaultdict(list)
    history_reader = open('../Dataset/author/users_history_random_5.txt', 'r')
    for line in history_reader:
        line = line.strip().split('\t')
        uid = line[1]
        history_tweet = line[2].lower().split(' ')
        user_text_history_dict[uid].append(history_tweet)
        tid = line[0]
        history_image = tid
        user_image_history_dict[uid].append(history_image)
    history_reader.close()
    return user_text_history_dict, user_image_history_dict


def get_mentioned_history_dict():
    mentioned_text_history_dict = defaultdict(list)
    mentioned_image_history_dict = defaultdict(list)
    history_reader = open('../Dataset/mentioned_user/mentioned_history_random_5.txt', 'r')
    for line in history_reader:
        line = line.strip().split('\t')
        tid = line[0]
        uid = line[1]
        history_tweet = line[2].lower().split(' ')
        mentioned_text_history_dict[uid].append(history_tweet)
        history_image = tid
        mentioned_image_history_dict[uid].append(history_image)
    history_reader.close()
    return mentioned_text_history_dict, mentioned_image_history_dict


#load_data_and_labels() is use to generate raw data
def load_data_and_labels(sequence_len, user_mentioned_dict, user_text_history_dict, user_image_history_dict, mentioned_text_history_dict, mentioned_image_history_dict):
    tweet_reader = open('../Dataset/tweet/tweet_data.txt', 'r')
    dataset = []
    maxlen = 0
    maxmentioned = 0
    for line in tweet_reader:
        line = line.replace('\n', ' ').strip().split('\t')#replace('\n', '').split('\t')
        tid = line[0]
        uid = line[1]
        tweet_x = line[2].lower().split(' ')
        if maxlen < len(tweet_x):
            maxlen = len(tweet_x)

        um = user_mentioned_dict[uid]

        if maxmentioned < len(um):
            maxmentioned = len(um)

        tweet_i = tid

        y = line[3].split('||')

        user_x = user_text_history_dict[uid][:sequence_len]
        user_i = user_image_history_dict[uid][:sequence_len]

        mentioned_text_history = defaultdict(list)
        mentioned_image_history = defaultdict(list)

        for mentioned_id in um:
            mentioned_text_history[mentioned_id] = mentioned_text_history_dict[mentioned_id][:sequence_len]
            mentioned_image_history[mentioned_id] = mentioned_image_history_dict[mentioned_id][:sequence_len]

        dataset.append((uid, user_x, user_i, tweet_x, tweet_i, um, mentioned_text_history, mentioned_image_history, y))

    for uid, user_text_history in user_text_history_dict.items():
        for t in user_text_history:
            if maxlen < len(t):
                maxlen = len(t)

    for uid, mentioned_text_history in mentioned_text_history_dict.items():
        for t in mentioned_text_history:
            if maxlen < len(t):
                maxlen = len(t)

    return [dataset, maxlen, maxmentioned]


def build_vocab(train_data, valid_data, test_data):
    word = []
    user_id = []
    for data in train_data:
        uid, user_x, user_i, tweet_x, tweet_i, um, mentioned_text_history, mentioned_image_history, y = data
        user_id.append(uid)
        user_id.extend(um)
        word.extend(tweet_x)
        for t in user_x:
            word.extend(t)
        for mentioned_id, d in mentioned_text_history.items():
            for t in d:
                word.extend(t)

    for data in valid_data:
        uid, user_x, user_i, tweet_x, tweet_i, um, mentioned_text_history, mentioned_image_history, y = data
        user_id.append(uid)
        user_id.extend(um)
        word.extend(tweet_x)
        for t in user_x:
            word.extend(t)
        for mentioned_id, d in mentioned_text_history.items():
            for t in d:
                word.extend(t)

    for uid, dataset in test_data.items():
        for data in dataset:
            uid, user_x, user_i, tweet_x, tweet_i, um, mentioned_text_history, mentioned_image_history, y = data
            user_id.append(uid)
            user_id.extend(um)
            word.extend(tweet_x)
            for t in user_x:
                word.extend(t)
            for mentioned_id, d in mentioned_text_history.items():
                for t in d:
                    word.extend(t)

    word_counts = Counter(word)

    vocabulary_inv = [x[0] for x in word_counts.most_common()[:50000]]
    vocabulary = {x:i+1 for i, x in enumerate (vocabulary_inv)}

    user_counts = Counter(user_id)

    user_vocabulary_inv = [x[0] for x in user_counts.most_common()]
    user_vocabulary = {x:i for i,x in enumerate(user_vocabulary_inv)}

    return [vocabulary, vocabulary_inv, user_vocabulary, user_vocabulary_inv]

def build_train_data(dataset, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned):
    uid_x = []
    tweet_x = []
    tweet_i = []
    user_x = []
    user_i = []
    mentioned_id = []
    mentioned_x = []
    mentioned_i = []
    y = []
    for data in dataset:
        uid, ux, ui, tx, ti, um, mx, mi, target = data
        for m_id in um:
            uid_x.append(uid)
            tweet_x.append(tx)
            tweet_i.append(ti)
            user_x.append(ux)
            user_i.append(ui)
            mentioned_id.append(m_id)
            mentioned_x.append(mx[m_id])
            mentioned_i.append(mi[m_id])
            if m_id in target:
                y.append(1)
            else:
                y.append(0)

    uid_x = np.asarray([user_vocabulary[id] for id in uid_x], dtype=np.int32)

    tweet_x = np.asarray([[vocabulary[word] for word in t if word in vocabulary] for t in tweet_x])
    tweet_x = np.asarray([np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0,0)) for t in tweet_x], dtype=np.int32)

    user_x = np.asarray([[[vocabulary[word] for word in t if word in vocabulary] for t in x]for x in user_x])
    user_x = np.asarray([[np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0, 0)) for t in x]for x in user_x], dtype=np.int32)
    user_x = user_x.reshape(user_x.shape[0],sequence_len*maxlen)

    mentioned_id = np.asarray([user_vocabulary[id] for id in mentioned_id], dtype=np.int32)

    mentioned_x = np.asarray([[[vocabulary[word]for word in t if word in vocabulary]for t in x]for x in mentioned_x])
    mentioned_x = np.asarray([[np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0,0)) for t in x]for x in mentioned_x], dtype=np.int32)

    mentioned_x = mentioned_x.reshape(mentioned_x.shape[0], sequence_len*maxlen)
    y = np.asarray(y, dtype=np.int32)

    train_data = (uid_x, tweet_x, tweet_i, user_x, user_i, mentioned_id, mentioned_x, mentioned_i, y)
    return  train_data

def build_valid_data(dataset, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned):
    valid_data = []
    for data in dataset:
        uid_x = []
        tweet_x = []
        tweet_i = []
        user_x = []
        user_i = []
        mentioned_id = []
        mentioned_x = []
        mentioned_i = []
        y = []
        uid, ux, ui, tx, ti, um, mx, mi, target = data
        for m_id in um:
            uid_x.append(uid)
            tweet_x.append(tx)
            tweet_i.append(ti)
            user_x.append(ux)
            user_i.append(ui)
            mentioned_id.append(m_id)
            mentioned_x.append(mx[m_id])
            mentioned_i.append(mi[m_id])
            if m_id in target:
                y.append(1)
            else:
                y.append(0)
        uid_x = np.asarray([user_vocabulary[id] for id in uid_x], dtype=np.int32)

        tweet_x = np.asarray([[vocabulary[word] for word in t if word in vocabulary] for t in tweet_x])
        tweet_x = np.asarray(
            [np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0, 0)) for t in tweet_x], dtype=np.int32)

        user_x = np.asarray([[[vocabulary[word] for word in t if word in vocabulary] for t in x]for x in user_x])
        user_x = np.asarray([[np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0, 0)) for t in x] for x in user_x],
                            dtype=np.int32)
        user_x = user_x.reshape(user_x.shape[0], sequence_len * maxlen)

        mentioned_id = np.asarray([user_vocabulary[id] for id in mentioned_id], dtype=np.int32)

        mentioned_x = np.asarray(
            [[[vocabulary[word] for word in t if word in vocabulary] for t in x] for x in mentioned_x])
        mentioned_x = np.asarray(
            [[np.pad(t, (0, maxlen - len(t)), mode='constant', constant_values=(0, 0)) for t in x] for x in mentioned_x],
            dtype=np.int32)
        mentioned_x = mentioned_x.reshape(mentioned_x.shape[0], sequence_len * maxlen)

        y = np.asarray(y, dtype=np.int32)

        valid_data.append((uid_x, tweet_x, tweet_i, user_x, user_i, mentioned_id, mentioned_x, mentioned_i, y))

    return valid_data

def build_test_data(dataset, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned):
    test_data = []
    for uid, userdata in dataset.items():
        split_index = int(len(userdata)*0.8)
        train_data = userdata[:split_index]
        valid_data = userdata[split_index:]

        train_dataset = build_train_data(train_data, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned)
        valid_dataset = build_valid_data(valid_data, vocabulary, user_vocabulary, sequence_len, maxlen, maxmentioned)
        test_data.append((train_dataset, valid_dataset))
    return test_data

def load_image(sequence_len, ti, ui, mi):
    tweet_i_reader = h5py.File('../Dataset/tweet/tweet_img_vgg_feature_8275_224.h5', 'r')
    user_i_reader = h5py.File('../Dataset/author/user_img_vgg_feature_8275_224.h5', 'r')
    mentioned_i_reader = h5py.File('../Dataset/mentioned_user/mentioned_img_vgg_feature_8275_224.h5', 'r')
    feat_dim = 512

    tweet_i = []
    for i in ti:
        try:
            t = tweet_i_reader[i][:]
        except KeyError:
            t = np.zeros(shape=(1, feat_dim, 7, 7), dtype=np.float32)
        tweet_i.append(t)
    tweet_i = np.asarray(tweet_i, dtype=np.float32)

    user_i = []
    for ii in ui:
        temp = []
        for i in ii:
            try:
                t = user_i_reader[i][:]
                t = np.asarray(t, dtype=np.float32).reshape((feat_dim, 7, 7))
            except KeyError:
                t = np.zeros(shape=(feat_dim, 7, 7), dtype=np.float32)
            temp.append(t)
        temp = np.asarray(temp, dtype=np.float32)
        user_i.append(temp)
    user_i = np.asarray(user_i, dtype=np.float32)

    mentioned_i = []
    for ii in mi:
        temp = []
        for i in ii:
            try:
                t = mentioned_i_reader[i][:]
                t = np.asarray(t, dtype=np.float32).reshape((feat_dim, 7, 7))
            except KeyError:
                t = np.zeros(shape=(feat_dim, 7, 7), dtype=np.float32)
            temp.append(t)
        temp = np.asarray(temp, dtype=np.float32)
        mentioned_i.append(temp)
    mentioned_i = np.asarray(mentioned_i, dtype=np.float32)

    tweet_i_reader.close()
    user_i_reader.close()
    mentioned_i_reader.close()
    return tweet_i, user_i, mentioned_i

def generate_img(train_tx, train_ti, train_ux, train_ui, train_mx, train_mi, train_y, batch_size):
    tweet_i_reader = h5py.File('../Dataset/tweet/tweet_img_vgg_feature_8275_224.h5', 'r')
    user_i_reader = h5py.File('../Dataset/author/user_img_vgg_feature_8275_224.h5', 'r')
    mentioned_i_reader = h5py.File('../Dataset/mentioned_user/mentioned_img_vgg_feature_8275_224.h5', 'r')
    total_data_size = train_tx.shape[0]
    np.random.seed()
    indices = np.random.permutation(np.arange(total_data_size))
    feat_dim = 512
    while True:
        ttx = []
        tti = []
        tux = []
        tui = []
        tmx = []
        tmi = []
        tty = []
        out = 0
        for i in indices:
            out += 1
            try:
                ti = tweet_i_reader[train_ti[i]][:]
            except KeyError:
                ti = np.zeros(shape=(1, feat_dim, 7, 7), dtype=np.float32)
            user_i = []
            for ii in train_ui[i]:
                try:
                    ui = user_i_reader[ii][:]
                    ui = np.asarray(ui, dtype=np.float32).reshape((feat_dim, 7, 7))
                except KeyError:
                    ui = np.zeros(shape=(feat_dim, 7, 7), dtype=np.float32)
                user_i.append(ui)
            user_i = np.asarray(user_i, dtype=np.float32)
            mentioned_i = []
            for ii in train_mi[i]:
                try:
                    mi = mentioned_i_reader[ii][:]
                    mi = np.asarray(mi, dtype=np.float32).reshape((feat_dim, 7, 7))
                except KeyError:
                    mi = np.zeros(shape=(feat_dim, 7, 7), dtype=np.float32)
                mentioned_i.append(mi)
            mentioned_i = np.asarray(mentioned_i, dtype=np.float32)
            if out % batch_size == 0:
                ttx.append(train_tx[i])
                tti.append(ti)
                tux.append(train_ux[i])
                tui.append(user_i)
                tmx.append(train_mx[i])
                tmi.append(mentioned_i)
                tty.append(train_y[i])
                ttx = np.asarray(ttx, dtype=np.int32)
                tti = np.asarray(tti, dtype=np.float32)
                tux = np.asarray(tux, dtype=np.int32)
                tui = np.asarray(tui, dtype=np.float32)
                tmx = np.asarray(tmx, dtype=np.int32)
                tmi = np.asarray(tmi, dtype=np.float32)
                tty = np.asarray(tty, dtype=np.int32)
                yield([ttx, tti, tux, tui, tmx, tmi], tty)
                ttx = []
                tti = []
                tux = []
                tui = []
                tmx = []
                tmi = []
                tty = []
            else:
                ttx.append(train_tx[i])
                tti.append(ti)
                tux.append(train_ux[i])
                tui.append(user_i)
                tmx.append(train_mx[i])
                tmi.append(mentioned_i)
                tty.append(train_y[i])
        if out % batch_size != 0:
                ttx = np.asarray(ttx, dtype=np.int32)
                tti = np.asarray(tti, dtype=np.float32)
                tux = np.asarray(tux, dtype=np.int32)
                tui = np.asarray(tui, dtype=np.float32)
                tmx = np.asarray(tmx, dtype=np.int32)
                tmi = np.asarray(tmi, dtype=np.float32)
                tty = np.asarray(tty, dtype=np.int32)
                yield([ttx, tti, tux, tui, tmx, tmi], tty)
    tweet_i_reader.close()
    user_i_reader.close()
    mentioned_i_reader.close()

    mentioned_i_reader.close()

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, vocabulary, vocabulary_inv, user_vocabulary, user_vocabulary_inv, sequnce_len, maxlen, maxmentioned = load_data()

    uid_x, tweet_x, tweet_i, user_x, user_i, mentioned_id, mentioned_x, mentioned_i, y = train_dataset

    print 'vocabulary size: %d' % len(vocabulary_inv)
    print 'user vocabulary size: %d' % len(user_vocabulary_inv)

    print 'tweet shape:' , tweet_x.shape
    print 'user history shape:' , user_x.shape
    print 'mentioned history shape:', mentioned_x.shape
    print 'target shape:', y.shape

    test_train_dataset, test_valid_dataset = test_dataset[0]
    test_uid_x, test_tweet_x, test_tweet_i, test_user_x, test_user_i, test_mentioned_id, test_mentioned_x, test_mentioned_i, test_y = test_train_dataset

    print '-------------sample-----------------'
    print test_tweet_x
    print test_tweet_i
    print 'test_tweet_i shape:', test_tweet_i[0].shape
    print test_user_x
    print test_user_i
    print 'test_user_i shape:', test_user_i[0].shape
    print test_mentioned_x
    print test_mentioned_i
    print test_y

