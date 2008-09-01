#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:

from mvpa.suite import *

verbose.level = 4

# set MVPA_PATHS_DATA_ROOT accordingly!
datapath = cfg.get('paths', 'data root', default='data/meg.rieger')
verbose(1, 'Datapath is %s' % datapath)

conditions = {1: 'cosu', 2: 'fausu'}
subj = 'vp02'
# sampling rate after preprocessing in Hz
target_samplingrate = 80


def loadData(subj):
    datasets = []

    for cond_id, cond in conditions.iteritems():
        verbose(1, 'Loading data for condition %s' % cond)
        # for now just with tiny dataset
        meg = TuebingenMEG(os.path.join(datapath,
                                        subj + 'cf-3f' + cond + '.dat.gz'))

        # just select MEG channels
        data = meg.data[:, [i for i, v in enumerate(meg.channelids)
                                            if v.startswith('M')]]
        # keep list of corresponding channel ids
        channelids = [i for i in meg.channelids if i.startswith('M')]

        verbose(2, 'Selected %i channels' % data.shape[1])

        datasets.append(ChannelDataset(samples=data, labels=cond_id,
                            channelids=channelids, dt=1./meg.samplingrate,
                            t0=meg.timepoints[0]))

    # merge all datasets
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d
    # set uniq chunk id per each sample
    dataset.chunks = N.arange(dataset.nsamples)

    dataset = dataset.resample(sr=target_samplingrate)
    verbose(2, 'Downsampled data to %.1f Hz' % dataset.samplingrate)

    # substract the baseline from the data; uses t0 to determine the length
    # of the baseline window
    # XXX: shouldn't this be done per chunk?
    dataset.substractBaseline()
    verbose(2, 'Substracted %f sec baseline' % N.abs(dataset.t0))

    # select time window of interest: from onset to 600 ms after onset
    mask = dataset.mapper.getMask()
    # deselect timespoints prior to onset
    mask[:, :int(N.round(-dataset.t0 * dataset.samplingrate))] = False
    # deselect timepoints after 600 ms after onset
    mask[:, int(N.round((-dataset.t0 + 0.6) * dataset.samplingrate)):] = False
    # finally transform into feature selection list
    mask = dataset.mapForward(mask).nonzero()[0]

    # and apply selectio
    dataset = dataset.selectFeatures(mask)
    verbose(2, 'Applied a-priori feature selection, ' \
               'leaving %i timepoints per channel' % dataset.mapper.dsshape[1])

    # We might want to rechunk differently and coarsen the chunks
    if True:
        # arbitrarily group the sample into chunks
        # done to make the cross-validation a bit quicker; using 10 chunks
        # yields about the same performance as a full 'leave-really-only-a-
        # single-sample-out cross-validation'
        # samples are distributed, so that each chunk contains at least one
        # of each condition
        #nchunks = min(dataset.samplesperlabel.values())

        # hmm, the code below does a better job than coarsenChunks, wrt to
        # equalized distribution of samples....
        nchunks = 10
        dataset.chunks[dataset.labels==1] = \
            N.arange(N.sum(dataset.labels == 1)) % nchunks
        dataset.chunks[dataset.labels==2] = \
            N.arange(N.sum(dataset.labels == 2)) % nchunks

    return dataset


if __name__ == '__main__':
    # load the only subject that we have
    verbose(1, 'Loading data for subject: ' + subj)
    ds = loadData(subj)

    print ds.summary()

    verbose(1, 'Precondition data')
    doZScore = True
    if doZScore == True:
        zscore(ds, perchunk=True)
    else:
        # Just divide by max value among first 33 sensors (34 and 91 seems to
        # be bad, thus we need to exclude them)
        ds.samples *= 1.0/N.max(N.abs(ds.O[:,:33,:]))

    print ds.summary()

    best = {}
    for clf in clfs['linear', '!lars']:#[:1]:#[::-1]:
        try:# since some evil nuSVMs might puke on infeasible default nu
            # to get results from Figure2A
            cv2A = CrossValidatedTransferError(
                      TransferError(clf),
                      NFoldSplitter(),
                      enable_states=['confusion', 'training_confusion'])

            # to get results from Figure2B
            cv2B = CrossValidatedTransferError(
                      TransferError(clf),
                      NFoldSplitter(nperlabel='equal',
                                    # increase to reasonable number
                                    nrunspersplit=4),
                      enable_states=['confusion', 'training_confusion'])


            verbose(1, "Running cross-validation on %s" % clf.descr)
            error2A = cv2A(ds)
            verbose(2, "Figure 2A LOO performance:\n%s" % cv2A.confusion)
            if best.get('2A', (100, None, None))[0] > error2A:
                best['2A'] = (error2A, cv2A.confusion, clf.descr)

            error2B = cv2B(ds)
            verbose(2, "Figure 2B LOO performance:\n%s" % cv2B.confusion)
            if best.get('2B', (100, None, None))[0] > error2B:
                best['2B'] = (error2B, cv2B.confusion, clf.descr)
        except:
            pass

    verbose(1, "Best result for 2A was %g achieved on %s, and for 2B " \
               "was %g achieved using %s" %
            (best['2A'][0], best['2A'][2],
             best['2B'][0], best['2B'][2]))
