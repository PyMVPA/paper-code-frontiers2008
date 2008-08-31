#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simply functors that transform something."""

__docformat__ = 'restructuredtext'

from mvpa.suite import *
from mvpa.misc.plot.erp import _make_centeredaxis
import os.path
from scipy.signal import resample

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'eeg.fruend')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels
# XXX: only need id2label
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])

mode = 'color' # object, delayed
target_samplingrate = 200.0


# plotting helper function
def makeBarPlot(data, labels=None, title=None, ylim=None, ylabel=None,
               width=0.2, offset=0.2, color='0.6', distance=1.0):

    # determine location of bars
    xlocations = (N.arange(len(data)) * distance) + offset

    # work with arrays
    data = N.array(data)

    # plot bars
    plot = P.bar(xlocations,
                 data.mean(axis=1),
                 yerr=data.std(axis=1) / N.sqrt(data.shape[1]),
                 width=width,
                 color=color,
                 ecolor='black')

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations + width / 2, labels)

    if ylabel:
        P.ylabel(ylabel)

    # leave some space after last bar
    P.xlim(0, xlocations[-1] + width + offset)

    return plot


def labels2binlabels(ds, mode):
    try:
        filt = {'delayed': [1, 2, 3, 4],
                'color':   [1, 2, 5, 6],
                'object':  [2, 4, 6, 8]}[mode]
    except KeyError:
        raise ValueError, 'Unknown label recoding mode %s' % mode

    # XXX N.setmember1d should do smth like what we need but it does
    #     smth else ;-)
    # ACTUALLY: this should work:
    # N.logical_or.reduce(dataset.labels[:,None] == filt, axis=1).astype(int)
    # it seems not to be shorter though ;-) but more efficient! (may be ;-))

    ds.labels[:]=N.array([i in filt for i in ds.labels], dtype='int')

    # also we have now where, so smth like
    # l1 = ds.where(labels=[filt]); ds.labels[:] = 0; ds.labels[l1] = 1
    # should do


# TODO big -- make pymvpa working with string labels

def loadData(subj):
    ds = []                             # list of datasets

    verbose(1, "Loading EEG data from basepath %s" % datapath)
    for k, v in id2label.iteritems():
        filename = os.path.join(datapath, subj, v + '.bin')
        verbose(2, "Loading data '%s' with labels '%i'" % (v, k))

        ds += [EEPDataset(filename, labels=k)]

    d = reduce(lambda x,y: x+y, ds)     # combine into a single dataset

    verbose(1, 'Limit to binary problem: ' + mode)
    labels2binlabels(d, mode)

    # get data in original shape
    data = d.O
    ntimepoints = data.shape[2]

    # downsample data
    data = resample(data,
                    ntimepoints * target_samplingrate / d.samplingrate,
                    window='ham', axis=2)
    verbose(2, 'Downsampled data to %d Hz' % target_samplingrate)

    # new dt is total length by new timepoints
    new_dt = ntimepoints * d.dt / data.shape[2]

    # wrap data into new dataset
    d_new = ChannelDataset(samples=data, labels=d.labels, chunks=d.chunks,
                           t0=d.t0, dt=new_dt, channelids=d.channelids)

    return d_new

#
# Just a simple example of ERP plotting
#
# XXX: might vanish now
#
def plot_ERP(ds, addon=None):
    # sampling rate
    SR = ds.samplingrate
    # data is already trials, this would correspond sec before onset
    pre = -ds.t0
    # number of channels, samples per trial
    nchannels, spt = ds.mapper.mask.shape
    post = spt * 1.0/ SR - pre # compute seconds in trials after onset

    # map from channel name to index
#    ch_map = dict(zip(ds.channelids, xrange(nchannels)))
#    ch_of_interest = ch_map['Pz']

    # Nice to see the truth behind the bars ;-)
#    for errtype in ['std', 'ste']:
    errtype='std'

    fig = P.figure(facecolor='white', figsize=(8,4))
#    fig.clf()
    # tricks to get title above all subplots
#    ax = fig.add_subplot(1, 1, 1, frame_on=False);

#    P.title("Mimique of Figure 5 with error being %s " % errtype)
    # Lets plot few ERPs. Attempt to replicate Figure 5
    # Misfits:
    #  * range of results is completely different, so what scaling of the data
    #  * ERPs of figures are reported to diverge into positive side although from
    #    our plots we see that first we get N-peak
    #  * ERPs in the article are nice and smooth ;-) here we see the reality. May be ERPs were
    #    computed at different samples rate, rereferenced?
#        diff = ds.selectSamples(ds.labels == 0).samples \
#               - ds.selectSamples(ds.labels == 1).samples
    # for a nice selection
#    channels_oi = ['P7', 'P3', 'Pz', 'O1', 'O2', 'CP1']
    for nchannel, channel in enumerate(ds.channelids):
    #for nchannel, channel in enumerate(channels_oi):
        ch_of_interest = ds.channelids.index(channel)
        ax = fig.add_subplot(6, 6, nchannel+1, frame_on=False)
#        ax = fig.add_subplot(2, 3, nchannel+1, frame_on=False)
        P.title(channel)
        ax.axison = False
        t1 = plotERP(ds.mapper.reverse(
                     ds.selectSamples(
                         ds.labels == 0).samples)[:, ch_of_interest, :],
                     color='red',
                     pre=pre, post=post, SR=SR, ax=ax, errtype=errtype)
        t2 = plotERP(ds.mapper.reverse(
                     ds.selectSamples(
                         ds.labels == 1).samples)[:, ch_of_interest, :],
                     color='blue',
                     pre=pre, post=post, SR=SR, ax=ax, errtype=errtype)
        dwave = N.array(t1 - t2, ndmin=2)
        t2 = plotERP(dwave, color='black',
                     pre=pre, post=post, SR=SR, ax=ax, errtype='none')
        if not addon is None:
            addon_oi = addon[:, ch_of_interest, :]
            print dwave.max(),addon_oi.max()
            # scale to same max as dwave
            plotERP(dwave.max()/addon_oi.max() * addon_oi, color='green',
                         pre=pre, post=post, SR=SR, ax=ax, errtype=errtype)

        P.axhline(y=0, color='gray')
#        props = dict(color='gray', linewidth=2, markeredgewidth=2, zorder=1)
        # should become public
#        _make_centeredaxis(ax, 0, offset=0.3, ai=1, mult=-1.0, **props)


    # XXX yeah... the world is not perfect... with ylim to center them at
    # the same location we get some problems, thus manual tuning
    # remove ylim in plot_erps to get somewhat different result without need for this adjust
    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.65)
    #P.show()


def finalFigure(origds, mldataset, sens, channel):
    # sampling rate
    SR = origds.samplingrate
    # data is already trials, this would correspond sec before onset
    pre = -origds.t0
    # number of channels, samples per trial
    nchannels, spt = origds.mapper.mask.shape
    # compute seconds in trials after onset
    post = spt * 1.0/ SR - pre

    # index of the channel of interest
    ch_of_interest = origds.channelids.index(channel)

    # error type to use in all plots
    errtype='std'

    fig = P.figure(facecolor='white', figsize=(8,4))

    # plot ERPs
    ax = fig.add_subplot(2, 1, 1, frame_on=False)

    responses = [ origds['labels', i].O[:, ch_of_interest, :]
                  for i in [0, 1] ]
    dwave = N.array(responses[0].mean(axis=0) - responses[1].mean(axis=0),
                    ndmin=2)
    plotERPs( [{'label':'lineart', 'color':'r', 'data':responses[0]},
               {'label':'picture', 'color':'b', 'data':responses[1]},
               {'label':'dwave',   'color':'0', 'data':dwave, 'pre_mean':0}],
               pre=pre, pre_mean=pre, post=post, SR=SR, ax=ax, errtype=errtype,
               xlabel=None)

    # plot sensitivities
    ax = fig.add_subplot(2, 1, 2, frame_on=False)

    sens_labels = []
    erp_cfgs = []
    colors = ['red', 'green', 'blue', 'cyan']

    for i, (sid, s) in enumerate(sens[::-1]):
        sens_labels.append(sid)
        # back-project
        backproj = mldataset.mapReverse(s)
        # and normalize so that all non-zero weights sum up to 1
        # and scale into digestable range
        # finally, all sensitivities as absolute values
        normed_soi = Absolute(L2Normed(backproj)[:, ch_of_interest, :] * 1000)

        erp_cfgs.append(
            {'label': sid,
             'color': colors[i],
             'data' : normed_soi})

    plotERPs(erp_cfgs, pre=pre, post=post, SR=SR, ax=ax, errtype=errtype,
             ylabel=None)

    P.legend(sens_labels)

#    # per-channel barplot
#    ax = fig.add_subplot(3, 1, 3, frame_on=False)
#
#    barwidth=0.2
#    ps = []
#    for i, (sid, s) in enumerate(sens):
#        # back-project
#        backproj = mldataset.mapReverse(s)
#        # and normalize so that all non-zero weights sum up to 1
#        s_orig = L2Normed(backproj)#, norm=N.mean(backproj > 0))
#
#        # compute per channel scores (yields nchannels x nchunks)
#        scores = N.sum(s_orig, axis=2).T
#        p = makeBarPlot(scores[:-3], labels=origds.channelids[:-3],
#                    width=barwidth, offset=barwidth*i, color=colors[i])
#        ps.append(p)
#
#    P.legend( [p[0] for p in ps], sens_labels, loc=2)
#
    P.show()



if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData('ga14')

    # artificially group into chunks
    nchunks = 6
    verbose(1, 'Group data into %i handy chunks' % nchunks)
    coarsenChunks(ds, nchunks)

    # Re-reference the data relative to avg reference... not sure if
    # that would give any result
    do_avgref = False
    if do_avgref:
        verbose(1, 'Rereferencing data')
        ebdata = ds.mapper.reverse(ds.samples)
        ebdata_orig = ebdata
        avg = N.mean(ebdata[:,:-3,:], axis=1)
        ebdata_ = ebdata.swapaxes(1,2)
        ebdata_[:,:,:-3] -= avg[:,:,N.newaxis]
        ebdata = ebdata_.swapaxes(1,2)
        ds.samples = ds.mapper.forward(ebdata)

    verbose(1, 'A-priori feature selection')
    # a-priori feature selection
    mask = ds.mapper.getMask()
    # throw away EOG channels
    mask[-3:] = False
    # throw away timepoints prior onset
#    mask[:, :int(-ds.t0 * ds.samplingrate)] = False

    print ds.summary()
    # apply selection
    ds = ds.selectFeatures(ds.mapForward(mask).nonzero()[0])
    print ds.summary()

    do_zscore = True
    if do_zscore:
        verbose(1, 'Z-scoring')
        zscore(ds, perchunk=True)
    print ds.summary()

    # eats all sensitivities
    senses = []

    # splitter to use for all analyses
    splttr = NFoldSplitter()

    # some classifiers to test
    clfs = {
            'SMLR': SMLR(lm=0.1),
            'lCSVM': LinearCSVMC(),
            'lGPR': GPR(kernel=KernelLinear()),
           }

    # run classifiers in cross-validation
    for label, clf in clfs.iteritems():
        cv = \
          CrossValidatedTransferError(
            TransferError(clf),
            splttr,
            harvest_attribs=\
              ['transerror.clf.getSensitivityAnalyzer(force_training=False)()'],
            enable_states=['confusion', 'training_confusion'])

        verbose(1, 'Doing cross-validation with ' + label)
        # run cross-validation
        merror = cv(ds)
        verbose(1, 'Accumulated confusion matrix for out-of-sample tests')
        print cv.confusion

        # get harvested sensitivities for all splits
        sensitivities = N.array(cv.harvested.values()[0])
        # and store
        senses.append(
            (label + ' (%.2f%% corr.) weights' \
                % cv.confusion.stats['ACC'],
             sensitivities))
        # XXX: Do I really need to go through the valley of pain to get the
        #      accuracy?

    verbose(1, 'Computing additional sensitvities')
    # define some pure sensitivities (or related measures)
    sensanas={
              'ANOVA': OneWayAnova(),
              # no I-RELIEF for now -- takes too long
              #'I-RELIEF': IterativeReliefOnline(transformer=N.abs),
              # gimme more !!
             }

    # wrapper everything into SplitFeaturewiseMeasure
    # to get sense of variance across our artificial splits
    # compute additional sensitivities
    for k, v in sensanas.iteritems():
        verbose(2, 'Computing: ' + k)
        sa = SplitFeaturewiseMeasure(v, splttr,
                                     enable_states=['maps'])
        # compute sensitivities
        sa(ds)
        # and grab them for all splits
        senses.append((k, sa.maps))

    # (re)get pristine dataset for plotting of ERPs
    ds_pristine=loadData('ga14')

    # and finally plot figure for channel of choice
    finalFigure(ds_pristine, ds, senses, 'Pz')
