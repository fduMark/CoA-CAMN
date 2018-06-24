import numpy as np

def precision_score(test_y, pred_y, k=1):
    p_score = []
    for i in xrange(len(test_y)):
        result_at_topk = pred_y[i][-k:]
        count = 0
        for j in result_at_topk:
            if j in test_y[i]:
                count += 1

        p_score.append(float(count) / float(k))

    return np.mean(p_score)

def recall_score(test_y, pred_y, k=1):
    r_score = []
    for i in xrange(len(test_y)):
        result_at_topk = pred_y[i][-k:]
        count = 0
        for j in result_at_topk:
            if j in test_y[i]:
                count += 1
        r_score.append(float(count) / float(len(test_y[i])))

    return np.mean(r_score)

def hits_score(test_y, pred_y, k=1):
    h_score = []
    for i in xrange(len(test_y)):
        result_at_topk = pred_y[i][-k:]
        count = 0
        for j in result_at_topk:
            if j in test_y[i]:
                count += 1
        h_score.append(1 if count > 0 else 0)

    return np.mean(h_score)

def mrr_score(test_y, pred_y):
    m_score = []
    for i in xrange(len(test_y)):
        for j in xrange(len(pred_y[i])):
            if pred_y[i][-(j+1)] in test_y[i]:
                m_score.append(1.0 / float(j+1))
                break

    return np.mean(m_score)

def bpref(test_y, pred_y):
    b_score = []
    for i in xrange(len(test_y)):
        index = 0
        for j in xrange(len(pred_y[i])):
            if pred_y[i][-(j+1)] in test_y[i]:
                index = j+1
        b_score.append(1.0 - float(index - len(test_y[i]))/float(len(pred_y[i])))
    return np.mean(b_score)
