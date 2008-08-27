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

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'eeg.fruend')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])
cond_prefixes = ( 'sm', 'sf', 'dm', 'df' )


# plotting helper function
def makeBarPlot(data, labels=None, title=None, ylim=None, ylabel=None):
    xlocations = N.array(range(len(data))) + 0.5
    width = 0.5

    # work with arrays
    data = N.array(data)

    # plot bars
    plot = P.bar(xlocations,
                 data.mean(axis=1),
                 yerr=data.std(axis=1) / N.sqrt(data.shape[1]),
                 width=width,
                 color='0.6',
                 ecolor='black')

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations+ width/2, labels)

    if ylabel:
        P.ylabel(ylabel)

    P.xlim(0, xlocations[-1]+width*2)




# TODO big -- make pymvpa working with string labels

def loadData(subj):
    d = None

    verbose(1, "Loading EEG data from basepath %s" % datapath)
    for k, v in id2label.iteritems():
        filename = os.path.join(datapath, subj, v + '.bin')
        verbose(2, "Loading data '%s' with labels '%i'" % (v, k))

        t = EEPDataset(filename, labels=k)

        # XXX equalize number of samples (TODO)
        # YYY why do we actually want to equalize at this level?
        #t = t.selectSamples(t.chunks < 105)

        if d is None:
            d = t
        else:
            d += t

    return d

#
# Just a simple example of ERP plotting
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
    errtype='ste'

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


def clfEEG_dummy(ds):
    #
    # Simple classification. Silly one for now
    #
    verbose(1, "Sweeping through classifiers with odd/even splitter for generalization")
    for clf in clfs['binary', '!lars', 'has_sensitivity', 'does_feature_selection']: # lars is too slow
        cv = CrossValidatedTransferError(
            TransferError(clf),
            OddEvenSplitter(nperlabel='equal'),
            enable_states=['confusion', 'training_confusion'])
        verbose(2, "Classifier " + clf.descr)
        error = cv(ds)
        tstats = cv.training_confusion.stats
        stats = cv.confusion.stats
        verbose(3, "Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                (tstats['ACC'], tstats['MCC'][0],
                 stats['ACC'], stats['MCC'][0]))


def scoreChannels(ds):
    clf = SMLR()
    cv=CrossValidatedTransferError(TransferError(LinearCSVMC()), OddEvenSplitter(), enable_states=['transerrors'])
    clf.train(ds)

    sa = clf.getSensitivityAnalyzer(force_training=False, transformer=L2Normed)

    return sa()


def runCV(ds):
    cv = CrossValidatedTransferError(
          TransferError(LinearCSVMC()),
#          OddEvenSplitter(),
          NFoldSplitter(),
          harvest_attribs=\
           ['transerror.clf.getSensitivityAnalyzer(force_training=False)()'],
          enable_states=['confusion', 'training_confusion'])
    verbose(1, 'Doing cross-validation')
    merror = cv(ds)

    return cv


def labels2binlabels(ds, mode):
    if mode == 'delayed':
        filt = [1, 2, 3, 4]
    elif mode == 'color':
        filt = [1, 2, 5, 6]
    elif mode == 'object':
        filt = [2, 4, 6, 8]
    else:
        raise ValueError, 'Unknown label recoding mode'

    ds.labels[:]=N.array([i in filt for i in ds.labels], dtype='int')


if __name__ == '__main__':
    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess
    ds=loadData('ga14')

    mode = 'color' # object, delayed

    # limit to binary problem
    verbose(1, 'Limit to binary problem: ' + mode)
    labels2binlabels(ds, mode)

    # artificially group into chunks
    coarsenChunks(ds, 6)

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

#    plot_ERP(ds)

    verbose(1, 'A-priori feature selection')
    # a-priori feature selection
    mask = ds.mapper.getMask()
    # throw away EOG channels
    mask[-3:] = False
    # throw away timepoints prior onset
    mask[:, :int(-ds.t0 * ds.samplingrate)] = False

    print ds.summary()
    # apply selection
    ds = ds.selectFeatures(ds.mapForward(mask).nonzero()[0])
    print ds.summary()


    #do_wavelets = False
    #if do_wavelets:
    #    verbose(1, 'Applying wavelet mapper')
    #    ebdata = ds.mapper.reverse(ds.samples)
    #    WT = WaveletTransformationMapper(dim=2)
    #    ds_orig = ds
    #    ebdata_wt = WT(ebdata)
    #    ds = MaskedDataset(samples=ebdata_wt,
    #                       labels=ds_orig.labels,
    #                       chunks=ds_orig.chunks)


    do_zscore = True
    if do_zscore:
        verbose(1, 'Zscoring')
        zscore(ds, perchunk=True)

    #clfEEG_dummy(ds)
    #r = scoreChannels(ds)
    cv = runCV(ds)
    sensitivities = N.array(cv.harvested.values()[0])

    # back-project
    s_orig = ds.mapReverse(sensitivities)

    # get pristine dataset
    ds_pristine=loadData('ga14')
    labels2binlabels(ds_pristine, mode)

    plot_ERP(ds_pristine, s_orig)
