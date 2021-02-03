import numpy as np
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import mean_squared_error
from skll.metrics import kappa as kpa

def auc(actual, predicted, average_over_labels=True, partition=1024.):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual),-1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted),-1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    ac = ac[na]
    pr = pr[na]

    label_auc = []
    for i in range(ac.shape[-1]):
        a = np.array(ac[:,i])
        p = np.array(pr[:,i])

        val = np.unique(a)

        # if len(val) > 2:
        #     print('AUC Warning - Number of distinct values in label set {} is greater than 2, '
        #           'using median split of distinct values...'.format(i))
        if len(val) == 1:
            # print('AUC Warning - There is only 1 distinct value in label set {}, unable to calculate AUC'.format(i))
            label_auc.append(np.nan)
            continue

        pos = np.argwhere(a[:] >= np.median(val))
        neg = np.argwhere(a[:] < np.median(val))

        # print(pos)
        # print(neg)

        p_div = int(np.ceil(len(pos)/partition))
        n_div = int(np.ceil(len(neg)/partition))

        # print(len(pos), p_div)
        # print(len(neg), n_div)

        div = 0
        for j in range(int(p_div)):
            p_range = list(range(int(j * partition), int(np.minimum(int((j + 1) * partition), len(pos)))))
            for k in range(n_div):
                n_range = list(range(int(k * partition), int(np.minimum(int((k + 1) * partition), len(neg)))))


                eq = np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[pos[p_range]].T == np.ones(
                    (np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[neg[n_range]]

                geq = np.array(np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) *
                               p[pos[p_range]].T >= np.ones((np.alen(neg[n_range]),
                                                             np.alen(pos[p_range]))) * p[neg[n_range]],
                               dtype=np.float32)
                geq[eq[:, :] == True] = 0.5

                # print(geq)
                div += np.sum(geq)
                # print(np.sum(geq))
                # exit(1)

        label_auc.append(div / (np.alen(pos)*np.alen(neg)))
        # print(label_auc)

    if average_over_labels:
        return np.nanmean(label_auc)
    else:
        return label_auc


def f1(actual, predicted):
    return f1_score(np.array(actual), np.round(predicted))


def rmse(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    label_rmse = []
    for i in range(ac.shape[-1]):
        dif = np.array(ac[:, i]) - np.array(pr[:, i])
        sqdif = dif**2
        mse = np.nanmean(sqdif)
        label_rmse.append(np.sqrt(mse))


    if average_over_labels:
        return np.nanmean(label_rmse)
    else:
        return label_rmse


def cohen_kappa(actual, predicted, split=0.5, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual,dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted,dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = np.array(np.array(ac[na]) > split, dtype=np.int32)
    pr = np.array(np.array(pr[na]) > split, dtype=np.int32)

    label_kpa = []
    if hasattr(split, '__iter__'):
        assert len(split) == ac.shape[-1]
    else:
        split = np.ones(ac.shape[1]) * split

    for i in range(ac.shape[-1]):
        label_kpa.append(kpa(np.array(np.array(ac[:, i]) > split[i], dtype=np.int32),
                np.array(np.array(pr[:, i]) > split[i], dtype=np.int32)))

    if average_over_labels:
        return np.nanmean(label_kpa)
    else:
        return label_kpa


def cohen_kappa_multiclass(actual, predicted):
    assert len(actual) == len(predicted)

    ac = np.array(actual,dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted,dtype=np.float32).reshape((len(predicted), -1))

    try:
        na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
    except:
        for i in ac:
            print(i)

        for i in ac:
            print(np.any(np.isnan(i)))

    if len(na) == 0:
        return np.nan

    aci = np.argmax(np.array(np.array(ac[na]), dtype=np.int32), axis=1)
    pri = np.argmax(np.array(np.array(pr[na]), dtype=np.float32), axis=1)

    # for i in range(len(aci)):
    #     print(aci[i],'--',pri[i],':',np.array(pr[na])[i])

    return kpa(aci,pri)

# def kappa(actual, predicted, split=0.5):
#     # pred = normalize(list(predicted), method='uniform')
#     return kpa(actual, [p > split for p in predicted])

def rsquared(actual, predicted):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    return r2_score(ac[na], pr[na])


if __name__ == "__main__":

    import time

    n = 100

    partition = [64., 128., 256., 512., 1024., 2048., 4096.]
    h = '{:^20}'.format('Samples')
    for p in partition:
        h += '{:^15}'.format('AUC ({})'.format(int(p)))
    print(h)
    print(('{:=<' + '{}'.format(len(h)) + '}').format(''))
    for i in range(13):
        print('{:<20}'.format(n), end='')

        y = np.random.randint(0,2,n)

        y_hat = np.random.rand(n)
        y_hat[np.argwhere(y == 1).ravel()] += 0.2

        for p in partition:
            start = time.time()
            try:
                a = auc(y,y_hat, p)
                t = time.time()-start
                print('{:^15}'.format('{:<.3f}s'.format(t)), end='')
            except:
                print('{:^15}'.format('Failed'), end='')
        print('')

        n *= 2

