from keras.models import Model
from keras.layers import Input, Dense, Embedding, merge, Dropout, LSTM, AveragePooling1D, \
    TimeDistributed, Multiply, Dot, Concatenate, Add
from keras.layers.core import Activation, Dense, Permute, Flatten, Dropout, Reshape, Layer, \
    ActivityRegularization, RepeatVector, Lambda
from keras.utils import plot_model
from keras.callbacks import History
from keras import backend as K
from keras.layers.wrappers import Bidirectional
from data_helper import load_data, load_image, generate_img, test_img
from result_calculator import *
import time
from keras.optimizers import RMSprop, Adamax
from keras.initializers import TruncatedNormal

if __name__ == '__main__':
    print ('loading data...')
    train_dataset, valid_dataset, test_dataset, vocabulary, vocabulary_inv, user_vocabulary, user_vocabulary_inv, seqlen, maxlen, maxmentioned = load_data()

    uid, tx, ti, ux, ui, meid, mx, mi, y = train_dataset

    vocab_size = len(vocabulary_inv) + 1
    user_vocab_size = len(user_vocabulary_inv) + 1

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('User size:', user_vocab_size, 'unique users')
    print('Max length:', maxlen, 'words')
    print('Max mentioned:', maxmentioned, 'users')
    print('-')
    print('Here\'s what a "mention" tuple looks like (tweet_x, user_x, mentioned_x, label):')
    print(tx[0], ux[0], mx[0], y[0])
    print('-')
    print('-')
    print('input_mention_tweet: integer tensor of shape (samples, max_length)')
    print('shape:', tx.shape)
    print('-')
    print('-')
    print('input_user_document: integer tensor of shape (samples, sequence_len*max_length)')
    print('shape:', ux.shape)
    print('-')
    print('-')
    print('input_mentioned_document: integer tensor of shape (samples, maxmentioned)')
    print('shape', mx.shape)
    print('-')
    print('-')
    print('input_label: integer tensor of shape (samples,)')
    print('shape:', y.shape)
    print('-')

    embedding_dim = 300
    feat_dim = 512
    w = 7
    layers = 7
    sequence_len = 5

    train_image_batch_size = 2000

    print 'embedding dim: %s' % embedding_dim
    print 'layer num: %s' % layers

    # build model
    print "Build model..."

    tweet_x = Input(shape=(maxlen,), dtype='int32')
    tweet_i = Input(shape=(1, feat_dim, w, w), dtype='float32')

    user_x = Input(shape=(sequence_len * maxlen,), dtype='int32')
    user_i = Input(shape=(sequence_len, feat_dim, w, w), dtype='float32')

    mentioned_x = Input(shape=(sequence_len * maxlen,), dtype='int32')
    mentioned_i = Input(shape=(sequence_len, feat_dim, w, w), dtype='float32')

    '''Query tweet part'''

    tweet_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer='uniform', \
                                mask_zero=False, input_length=maxlen)(tweet_x)
    tweet_lstm = LSTM(embedding_dim, return_sequences=True, input_shape=(maxlen, embedding_dim))(tweet_embedding)
    tweet_lstm = Dropout(0.5)(tweet_lstm)
    tweet_avg = AveragePooling1D(pool_size=maxlen)(tweet_lstm)
    tweet_avg = Flatten()(tweet_avg)

    tweet_i_reshape = Reshape(input_shape=(1, feat_dim, w, w), target_shape=(w * w, feat_dim))(tweet_i)

    # tweet->image
    tweet_avg_dense = Dense(embedding_dim)(tweet_avg)
    tweet_repeat = RepeatVector(w * w)(tweet_avg_dense)
    img_dense = TimeDistributed(Dense(embedding_dim))(tweet_i_reshape)
    att_1 = Multiply()([tweet_repeat, img_dense])
    att_1 = Activation('tanh')(att_1)
    att_1 = TimeDistributed(Dense(1))(att_1)
    att_1 = Activation('softmax')(att_1)

    ti_new = Dot(axes=(1, 1))([att_1, img_dense])
    ti_new = Permute((2, 1))(ti_new)
    ti_new = Flatten()(ti_new)

    # img->text
    img_new_dense = Dense(embedding_dim)(ti_new)
    img_new_repeat = RepeatVector(maxlen)(img_new_dense)
    tweet_dense = TimeDistributed((Dense(embedding_dim)))(tweet_lstm)
    att_2 = Multiply()([img_new_repeat, tweet_dense])
    att_2 = Activation('tanh')(att_2)
    att_2 = TimeDistributed(Dense(1))(att_2)
    att_2 = Activation('softmax')(att_2)

    tweet_new = Dot(axes=(1, 1))([att_2, tweet_lstm])
    tweet_new = Permute((2, 1))(tweet_new)
    tweet_new = Flatten()(tweet_new)

    ohi = Dense(embedding_dim, activation='tanh')(ti_new)
    ohx = Lambda(function=lambda x: K.sum(x, axis=1),
                 output_shape=lambda shape: (shape[0],) + shape[2:])(tweet_embedding)
    ohx = Add()([tweet_new, ohx])

    oh = Multiply()([ohi, ohx])

    '''Author history part'''
    ux_embedding_A = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer='uniform', \
                               mask_zero=False, input_length=sequence_len * maxlen)(user_x)
    ux_embedding_A = Dropout(0.5)(ux_embedding_A)
    ux_embedding_A_reshape = Reshape(target_shape=(seqlen, maxlen, embedding_dim))(ux_embedding_A)
    ux_embedding_B = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer='uniform', \
                               mask_zero=False, input_length=sequence_len * maxlen)(user_x)
    ux_embedding_B = Dropout(0.5)(ux_embedding_B)
    ux_embedding_B_reshape = Reshape(target_shape=(seqlen, maxlen, embedding_dim))(ux_embedding_B)

    ui_reshape = Reshape(input_shape=(sequence_len, feat_dim, w, w), target_shape=(sequence_len * w * w, feat_dim))(
        user_i)
    ui_bi_lstm = LSTM(embedding_dim, return_sequences=True, dropout=0.5, input_shape=(sequence_len * w * w, feat_dim))(
        ui_reshape)
    ui_bi_reshape = Reshape(input_shape=(sequence_len * w * w, embedding_dim),
                            target_shape=(sequence_len, w * w, embedding_dim))(ui_bi_lstm)

    '''Candidate user history part'''
    mx_embedding_A = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer='uniform', \
                               mask_zero=False, input_length=sequence_len * maxlen)(mentioned_x)
    mx_embedding_A = Dropout(0.5)(mx_embedding_A)
    mx_embedding_A_reshape = Reshape(target_shape=(seqlen, maxlen, embedding_dim))(mx_embedding_A)

    mx_embedding_B = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer='uniform', \
                               mask_zero=False, input_length=sequence_len * maxlen)(mentioned_x)
    mx_embedding_B = Dropout(0.5)(mx_embedding_B)
    mx_embedding_B_reshape = Reshape(target_shape=(seqlen, maxlen, embedding_dim))(mx_embedding_B)

    mi_reshape = Reshape(input_shape=(sequence_len, feat_dim, w, w), target_shape=(sequence_len * w * w, feat_dim))(
        mentioned_i)
    mi_bi_lstm = LSTM(embedding_dim, return_sequences=True, dropout=0.5, input_shape=(sequence_len * w * w, feat_dim))(
        mi_reshape)
    mi_bi_reshape = Reshape(input_shape=(sequence_len * w * w, embedding_dim),
                            target_shape=(sequence_len, w * w, embedding_dim))(mi_bi_lstm)

    for i in xrange(layers):
        ux_word_att = Dot(axes=(1, 3))([ohx, ux_embedding_A_reshape])
        ux_word_att = Reshape(target_shape=(sequence_len, maxlen))(ux_word_att)
        ux_word_att = TimeDistributed(Activation('softmax'))(ux_word_att)
        ux_word_att = Lambda(function=lambda x: K.repeat_elements(x, embedding_dim, axis=2),
                             output_shape=lambda shape: (shape[0], shape[1], shape[2] * embedding_dim))(ux_word_att)

        ux_word_att = Reshape(target_shape=(sequence_len, maxlen, embedding_dim))(ux_word_att)
        ux_tpre = Multiply()([ux_word_att, ux_embedding_B_reshape])
        # shape(None, sequence_len, maxlen, embedding_dim)
        ux_tpre = Lambda(function=lambda x: K.sum(x, axis=2),
                         output_shape=lambda shape: (shape[0], shape[1]) + shape[3:])(ux_tpre)
        # shape(None, sequence_len, embedding_dim)
        ohx_s_dense = Dense(embedding_dim)(ohx)
        ohx_s_repeat = RepeatVector(sequence_len)(ohx_s_dense)

        ux_tpre_dense = TimeDistributed(Dense(embedding_dim))(ux_tpre)

        ux_mpre = Add()([ux_tpre_dense, ohx_s_repeat])
        ux_mpre = TimeDistributed(Activation('tanh'))(ux_mpre)
        # shape(None, sequence_len, embedding_dim)
        ux_m_att = TimeDistributed(Dense(1))(ux_mpre)
        ux_m_att = Reshape(target_shape=(sequence_len,))(ux_m_att)
        ux_m_att = Activation('softmax')(ux_m_att)

        ux_mpre = Dot(axes=(1, 1))([ux_m_att, ux_tpre])
        ux_mpre = Reshape(target_shape=(embedding_dim,))(ux_mpre)

        # shape(None, embedding)

        # hop->image
        ui_bi = TimeDistributed(Dense(embedding_dim))(ui_bi_reshape)
        ui_bi = TimeDistributed(Activation('tanh'))(ui_bi)
        ohi_ui_dense = Dense(embedding_dim)(ohi)
        ohi_ui_dense = Activation('tanh')(ohi_ui_dense)
        ohi_ui_repeat = RepeatVector(sequence_len * w * w)(ohi_ui_dense)
        ohi_ui = Reshape(target_shape=(sequence_len, w * w, embedding_dim))(ohi_ui_repeat)
        ui_att_h2i = Multiply()([ohi_ui, ui_bi])
        ui_att_h2i = TimeDistributed(Dense(1))(ui_att_h2i)
        ui_att_h2i = Reshape(input_shape=(sequence_len * w * w, 1), target_shape=(sequence_len, w * w))(ui_att_h2i)
        ui_att_h2i = TimeDistributed(Activation('softmax'))(ui_att_h2i)
        # shape(None, sequence, w*w)
        ui_att_h2i = Lambda(function=lambda x: K.repeat_elements(x, embedding_dim, axis=2),
                            output_shape=lambda shape: (shape[0], shape[1], shape[2] * embedding_dim))(ui_att_h2i)
        ui_att_h2i = Reshape(target_shape=(sequence_len, w * w, embedding_dim))(ui_att_h2i)
        # shape(None, sequence, w*w, embedding_dim)

        ui_s_pre = Multiply()([ui_att_h2i, ui_bi_reshape])
        # shape(sequence_len, w*w, embedding_dim)
        ui_s_pre = Lambda(function=lambda x: K.sum(x, axis=2),
                          output_shape=lambda shape: (shape[0], shape[1]) + shape[3:])(ui_s_pre)
        # shape(None, sequence_len, embedding_dim)

        ohi_uis_dense = Dense(embedding_dim)(ohi)
        ohi_uis_dense = Activation('tanh')(ohi_uis_dense)
        ohi_uis_repeat = RepeatVector(sequence_len)(ohi_uis_dense)

        ui_s_pre_dense = TimeDistributed(Dense(embedding_dim))(ui_s_pre)
        ui_s_pre_dense = TimeDistributed(Activation('tanh'))(ui_s_pre_dense)

        ui_m_pre = Multiply()([ui_s_pre_dense, ohi_uis_repeat])

        ui_m_att = TimeDistributed(Dense(1))(ui_m_pre)
        ui_m_att = Reshape(target_shape=(sequence_len,))(ui_m_att)
        ui_m_att = Activation('softmax')(ui_m_att)
        ui_m_att = Dropout(0.5)(ui_m_att)

        ui_mpre = Dot(axes=(1, 1))([ui_m_att, ui_s_pre])
        ui_mpre = Reshape(target_shape=(embedding_dim,))(ui_mpre)
        # shape(None, embedding_dim)


        mx_word_att = Dot(axes=(1, 3))([ohx, mx_embedding_A_reshape])
        mx_word_att = Reshape(target_shape=(sequence_len, maxlen))(mx_word_att)
        mx_word_att = TimeDistributed(Activation('softmax'))(mx_word_att)
        mx_word_att = Lambda(function=lambda x: K.repeat_elements(x, embedding_dim, axis=2),
                             output_shape=lambda shape: (shape[0], shape[1], shape[2] * embedding_dim))(mx_word_att)

        mx_word_att = Reshape(target_shape=(sequence_len, maxlen, embedding_dim))(mx_word_att)
        mx_tpre = Multiply()([mx_word_att, mx_embedding_B_reshape])
        # shape(None, sequence_len, maxlen, embedding_dim)
        mx_tpre = Lambda(function=lambda x: K.sum(x, axis=2),
                         output_shape=lambda shape: (shape[0], shape[1]) + shape[3:])(mx_tpre)
        # shape(None, sequence_len, embedding_dim)
        ohx_s_dense = Dense(embedding_dim)(ohx)
        ohx_s_repeat = RepeatVector(sequence_len)(ohx_s_dense)

        mx_tpre_dense = TimeDistributed(Dense(embedding_dim))(mx_tpre)

        mx_mpre = Add()([mx_tpre_dense, ohx_s_repeat])
        mx_mpre = TimeDistributed(Activation('tanh'))(mx_mpre)
        # shape(None, sequence_len, embedding_dim)
        mx_m_att = TimeDistributed(Dense(1))(mx_mpre)
        mx_m_att = Reshape(target_shape=(sequence_len,))(mx_m_att)
        mx_m_att = Activation('softmax')(mx_m_att)

        mx_mpre = Dot(axes=(1, 1))([mx_m_att, mx_tpre])
        mx_mpre = Reshape(target_shape=(embedding_dim,))(mx_mpre)
        # shape(None, embedding)

        # hop->image

        mi_bi = TimeDistributed(Dense(embedding_dim))(mi_bi_reshape)
        mi_bi = TimeDistributed(Activation('tanh'))(mi_bi)
        ohi_mi_dense = Dense(embedding_dim)(ohi)
        ohi_mi_dense = Activation('tanh')(ohi_mi_dense)
        ohi_mi_repeat = RepeatVector(sequence_len * w * w)(ohi_mi_dense)
        ohi_mi = Reshape(target_shape=(sequence_len, w * w, embedding_dim))(ohi_mi_repeat)
        mi_att_h2i = Multiply()([ohi_mi, mi_bi])
        mi_att_h2i = TimeDistributed(Dense(1))(mi_att_h2i)
        mi_att_h2i = Reshape(input_shape=(sequence_len * w * w, 1), target_shape=(sequence_len, w * w))(mi_att_h2i)
        mi_att_h2i = TimeDistributed(Activation('softmax'))(mi_att_h2i)
        # shape(None, sequence, w*w)
        mi_att_h2i = Lambda(function=lambda x: K.repeat_elements(x, embedding_dim, axis=2),
                            output_shape=lambda shape: (shape[0], shape[1], shape[2] * embedding_dim))(mi_att_h2i)
        mi_att_h2i = Reshape(target_shape=(sequence_len, w * w, embedding_dim))(mi_att_h2i)
        # shape(None, sequence, w*w, embedding_dim)

        mi_s_pre = Multiply()([mi_att_h2i, mi_bi_reshape])
        # shape(sequence_len, w*w, embedding_dim)
        mi_s_pre = Lambda(function=lambda x: K.sum(x, axis=2),
                          output_shape=lambda shape: (shape[0], shape[1]) + shape[3:])(mi_s_pre)
        # shape(None, sequence_len, embedding_dim)

        ohi_mis_dense = Dense(embedding_dim)(ohi)
        ohi_mis_dense = Activation('tanh')(ohi_mis_dense)
        ohi_mis_repeat = RepeatVector(sequence_len)(ohi_mis_dense)

        mi_s_pre_dense = TimeDistributed(Dense(embedding_dim))(mi_s_pre)
        mi_s_pre_dense = TimeDistributed(Activation('tanh'))(mi_s_pre_dense)

        mi_m_pre = Multiply()([mi_s_pre_dense, ohi_mis_repeat])

        mi_m_att = TimeDistributed(Dense(1))(mi_m_pre)
        mi_m_att = Reshape(target_shape=(sequence_len,))(mi_m_att)
        mi_m_att = Activation('softmax')(mi_m_att)
        mi_m_att = Dropout(0.5)(mi_m_att)

        mi_mpre = Dot(axes=(1, 1))([mi_m_att, mi_s_pre])
        mi_mpre = Reshape(target_shape=(embedding_dim,))(mi_mpre)

        ohx = Add()([mx_mpre, ohx, ux_mpre])
        ohi = Add()([mi_mpre, ohi, ui_mpre])
        ui_mpre = Dense(embedding_dim, activation='tanh')(ui_mpre)
        uxi = Multiply()([ux_mpre, ui_mpre])
        uxi = Dropout(0.5)(uxi)
        mi_mpre = Dense(embedding_dim, activation='tanh')(mi_mpre)
        mxi = Multiply()([mx_mpre, mi_mpre])
        mxi = Dropout(0.5)(mxi)
        oh = Add()([uxi, oh, mxi])

    output = Dense(1, activation='sigmoid')(oh)

    model = Model(input=[tweet_x, tweet_i, user_x, user_i, mentioned_x, mentioned_i], output=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

    # If you want to draw the model graph, you can use following code
    # plot_model(model, to_file='model.png', show_shapes='True')
    # plot(model, show_shapes=True, to_file='new_model.png')

    print model.summary()
    print "finished building model"


    print "starts training"
    best_f1 = 0
    topK = [1, 2, 3, 4, 5]
    y_pred = []
    test_y = []
    pred_time = []
    for i in xrange(len(test_dataset)):
        model.load_weights("parameters/CoA_CAMN.h5")
        train_dataset, test_data = test_dataset[i]
        start_clock = time.clock()
        uid, tx, ti, ux, ui, meid, mx, mi, y = train_dataset
        ti, ui, mi = load_image(sequence_len, ti, ui, mi)
        try:
            model.fit([tx, ti, ux, ui, mx, mi], y, batch_size=200, epochs=6, verbose=0)
        except:
            continue

        for data in test_data:
            test_uid_x, test_tweet_x, test_tweet_i, test_user_x, test_user_i, test_mentioned_id, test_mentioned_x, test_mentioned_i, ty = data
            test_tweet_id = test_tweet_i
            test_tweet_i, test_user_i, test_mentioned_i = load_image(sequence_len, test_tweet_i, test_user_i,
                                                                     test_mentioned_i)
            p = model.predict(
                [test_tweet_x, test_tweet_i, test_user_x, test_user_i, test_mentioned_x, test_mentioned_i], \
                batch_size=test_tweet_x.shape[0], verbose=0)
            py = np.argsort(p.flatten())
            ty = np.where(ty == 1)[0]
            y_pred.append(py)
            test_y.append(ty)

        finish_clock = time.clock()
        pred_time.append(finish_clock - start_clock)

    print "Mean time per tweet: %f s" % np.mean(pred_time)
    mrr = mrr_score(test_y, y_pred)
    bp = bpref(test_y, y_pred)
    print "MRR: ", mrr, "Bpref: ", bp
    for k in topK:
        precision = precision_score(test_y, y_pred, k=k)
        recall = recall_score(test_y, y_pred, k=k)
        hscore = hits_score(test_y, y_pred, k=k)
        hits3 = hits_score(test_y, y_pred, k=3)
        hits5 = hits_score(test_y, y_pred, k=5)
        F1 = 2 * (precision * recall) / (precision + recall)
        print "\t", "top:", k, "Test: precision:", precision, "recall:", recall, "f1 score:", F1, "hits score:", hscore, "hits@3 score:", hits3, "hits@5 score:", hits5




