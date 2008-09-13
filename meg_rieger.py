#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

__docformat__ = 'restructuredtext'

from mvpa.suite import *
from warehouse import doSensitivityAnalysis

if not locals().has_key('__IP'):
    # if not ran within IPython,
    # parse cmdline options if given
    parser.add_options([opt.zscore])
    parser.option_groups = [opts.common, opts.wavelet]
    (options, files) = parser.parse_args()
else:
    class O(object): pass
    options = O()
    options.wavelet_family = None # 'db1'
    options.wavelet_decomposition = 'dwt' # 'dwp'
    options.zscore = True

datapath = cfg.get('paths', 'data root', default='data/meg.rieger')
verbose(1, 'Datapath is %s' % datapath)

all_conditions = (('Correct Sure', 'cosu'),  ('Correct Unsure', 'cousu'),
                  ('False Unsure', 'fausu'), ('False Sure', 'fasu'))
conditions = dict([all_conditions[0], all_conditions[2]])
conditions_name = '-'.join(conditions.values())

# subject to process
subj = 'vp02'

# sampling rate after preprocessing in Hz
target_samplingrate = 80
post_duration = 0.6

def loadData(subj):
    datasets = []

    for cond_id, cond in conditions.iteritems():
        verbose(1, 'Loading data for condition %s:%s' % (cond, cond_id))
        # for now just with tiny dataset
        meg = TuebingenMEG(os.path.join(datapath,
                                        subj + 'cf-3f' + cond + '.dat.gz'))

        # just select MEG channels
        data = meg.data[:, [i for i, v in enumerate(meg.channelids)
                                            if v.startswith('M')]]
        # keep list of corresponding channel ids
        channelids = [i for i in meg.channelids if i.startswith('M')]

        verbose(2, 'Selected %i channels' % data.shape[1])

        datasets.append(ChannelDataset(samples=data,
                            labels=[cond_id] * len(data),
                            labels_map=True,
                            channelids=channelids, dt=1./meg.samplingrate,
                            t0=meg.timepoints[0]))

    # merge all datasets
    dataset = reduce(lambda x,y: x+y, datasets)
    # set uniq chunk id per each sample
    dataset.chunks = N.arange(dataset.nsamples)

    dataset = dataset.resample(sr=target_samplingrate)
    verbose(2, 'Downsampled data to %.1f Hz' % dataset.samplingrate)

    # substract the baseline from the data; uses t0 to determine the length
    # of the baseline window
    dataset.substractBaseline()
    verbose(2, 'Substracted %f sec baseline' % N.abs(dataset.t0))

    # select time window of interest: from onset to 600 ms after onset
    mask = dataset.mapper.getMask()
    # deselect timespoints prior to onset
    mask[:, :int(N.round(-dataset.t0 * dataset.samplingrate))] = False
    # deselect timepoints after 600 ms after onset
    mask[:, int(N.round((-dataset.t0 + post_duration)
                        * dataset.samplingrate)):] = False
    # finally transform into feature selection list
    mask = dataset.mapForward(mask).nonzero()[0]
    # and apply selection
    dataset = dataset.selectFeatures(mask)
    verbose(2, 'Applied a-priori feature selection, ' \
               'leaving %i timepoints per channel' % dataset.mapper.dsshape[1])

    nchunks = 8
    for l in dataset.uniquelabels:
        dataset.chunks[dataset.labels==l] = \
            N.arange(N.sum(dataset.labels == l)) % nchunks

    return dataset


def preprocess(ds):
    """Additional preprocessing
    """
    if options.wavelet_family is not None:
        verbose(2, "Converting into wavelets family %s."
                % options.wavelet_family)
        ebdata = ds.mapper.reverse(ds.samples)
        kwargs = {'dim': 1, 'wavelet': options.wavelet_family}
        if options.wavelet_decomposition == 'dwt':
            verbose(3, "Doing DWT")
            WT = WaveletTransformationMapper(**kwargs)
        else:
            verbose(3, "Doing DWP")
            WT = WaveletPacketMapper(**kwargs)
        ds_orig = ds
        ebdata_wt = WT(ebdata)
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels, chunks=ds_orig.chunks)
        ds.labels_map = ds_orig.labels_map

    zscore(ds, perchunk=False)

    return ds


def analysis(ds):
    # Lets replicate published obtained resuls.  We can do slightly
    # better using RFEs and initial feature selection, but lets just
    # replicate for the purpose of the paper
    clf = sg.SVM(kernel_type='linear')
    C = -2.0 # our default scaling is too soft
    # Scale C according  to the number of samples per class
    spl = ds.samplesperlabel
    ratio = N.sqrt(float(spl[0])/spl[1])
    clf.C = (C/ratio, C*ratio)

    # If we were only to do classification, following snippet is sufficient.
    # But lets reuse doSensitivityAnalysis
    #
    # cv2A = CrossValidatedTransferError(
    #           TransferError(clf),
    #           NFoldSplitter(),
    #           enable_states=['confusion', 'training_confusion', 'splits'])
    #
    # verbose(1, "Running cross-validation on %s" % clf.descr)
    # error2A = cv2A(ds)
    # verbose(2, "Figure 2A LOO performance:\n%s" % cv2A.confusion)

    clfs = {
        # explicitly instruct SMLR just to fit a single set of weights for our
        # binary task
        'SMLR': SMLR(lm=1.0, fit_all_weights=False),
        'lCSVM': clf,
        }

    # define some pure sensitivities (or related measures)
    sensanas={'ANOVA': OneWayAnova()}

    # perform the analysis and get all sensitivities
    senses = doSensitivityAnalysis(ds, clfs, sensanas, NFoldSplitter())

    # assign original single C
    clf.C = C
    # get results from Figure2B with resampling of the samples to
    # ballance number of samples per label
    cv2B = CrossValidatedTransferError(
              TransferError(clf),
              NFoldSplitter(nperlabel='equal',
                            # increase to reasonable number
                            nrunspersplit=4),
              enable_states=['confusion', 'training_confusion'])

    error2B = cv2B(ds)

    verbose(2, "Figure 2B LOO performance:\n%s" % cv2B.confusion)

    return senses


def finalFigure(ds_pristine, ds, senses, channel,
                fig=None, nsx=1, nsy=2, serp=1, ssens=2):
    """Pretty much rip off the EEG script
    """
    SR = ds_pristine.samplingrate
    # data is already trials, this would correspond sec before onset
    pre_onset = N.abs(ds_pristine.t0)
    pre = 0.05
    # number of channels, samples per trial
    nchannels, spt = ds_pristine.mapper.mask.shape
    post = 0.4

    # index of the channel of interest
    ch_of_interest = ds_pristine.channelids.index(channel)

    if fig is None:
        fig = P.figure(facecolor='white', figsize=(12, 6))

    # plot ERPs
    ax = fig.add_subplot(nsy, nsx, serp, frame_on=False)

    plots = []
    colors = ('r', 'b', '0')
    responses = [ ds_pristine['labels', i].O[:, ch_of_interest, :] * 1e15
                  for i in [0, 1] ]

    # TODO: move inside dataset API
    labels_map_rev = dict([reversed(x) for x in ds.labels_map.iteritems()])

    for l in ds_pristine.UL:
        plots.append({'label': labels_map_rev[l].tostring(),
                      'data' : responses[l], 'color': colors[l]})

    plots.append(
        {'label': 'dwave', 'color': colors[2], 'pre_mean': 0,
         'data':  N.array(responses[0].mean(axis=0)
                          - responses[1].mean(axis=0), ndmin=2)})

    plotERPs( plots,
              pre=pre, pre_onset=pre_onset,
              pre_mean=pre, post=post, SR=SR, ax=ax, errtype=['std', 'ci95'],
              ylim=(-500, 300), ylabel='fT', ylformat='%.1f',
              xlabel=None, legend=True)

    P.title(channel)
    # plot sensitivities
    ax = fig.add_subplot(nsy, nsx, ssens, frame_on=False)

    sens_labels = []
    erp_cfgs = []
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']

    for i, sens_ in enumerate(senses[::-1]):
        (sens_id, sens) = sens_[:2]
        sens_labels.append(sens_id)
        # back-project
        backproj = ds.mapReverse(sens)

        # and normalize so that all non-zero weights sum up to 1
        # ATTN: need to norm sensitivities for each fold on their own --
        # who knows what's happening otherwise
        for f in xrange(backproj.shape[0]):
            backproj[f] = L2Normed(backproj[f])

        # take one channel: yields (nfolds x ntimepoints)
        ch_sens = backproj[:, ch_of_interest, :]

        # sign of sensitivities is up to classifier relabling of the
        # input classes.
        # TODO: make it explicit, for now judge by the mean and flip
        #       if needed
        if ch_sens.mean() < 0:
            ch_sens *= -1

        # charge plot definition
        erp_cfgs.append(
            {'label': sens_id, 'color': colors[i], 'data' : ch_sens})

    # just ci95 error here, due to the low number of folds not much different
    # from std; also do _not_ demean based on initial baseline as we want the
    # untransformed sensitivities
    plotERPs(erp_cfgs, pre=pre, pre_onset=pre_onset,
             post=post, SR=SR, ax=ax, errtype='ci95',
             ylim=(-0.05, 0.3),
             ylabel=None, xlabel=None, ylformat='%.2f', pre_mean=0)

    P.legend(sens_labels)

    return fig


if __name__ == '__main__':
    # load the only subject that we have
    verbose(1, 'Loading data for subject: ' + subj)
    ds = loadData(subj)
    ds_pristine = ds.copy()
    print ds.summary()
    ds = preprocess(ds)
    senses = analysis(ds)

    # Draw few interesting channels
    for c in ['MRO22', 'MRO32', 'MZO01']:
        fig = finalFigure(ds_pristine, ds, senses, c)
        fig.savefig('figs/meg_rieger-%s-%s.svg' % (conditions_name, c))
        fig.savefig('figs/meg_rieger-%s-%s.png' % (conditions_name, c), dpi=90)
        P.close(fig)

    # Draw combined for two interesting channels
    fig = P.figure(figsize=(10,5), facecolor='white')
    finalFigure(ds_pristine, ds, senses, 'MRO22', fig, 2, 2, 1, 3)
    finalFigure(ds_pristine, ds, senses, 'MZO01', fig, 2, 2, 2, 4)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, wspace=0.01)
    P.draw()
    fig.savefig('figs/meg_rieger-%s-MRO22+MZO01.svg' % (conditions_name))
    fig.savefig('figs/meg_rieger-%s-MRO22+MZO01.png' % (conditions_name), dpi=90)

