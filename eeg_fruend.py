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
import os.path

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='../data'),
                        'eeg.fruend/ga14/')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])
cond_prefixes = ( 'sm', 'sf', 'dm', 'df' )


# TODO big -- make pymvpa working with string labels

def loadData():
    d = None

    verbose(1, "Loading EEG data from basepath %s" % datapath)
    for k, v in id2label.iteritems():
        filename = datapath + v + '.bin'
        verbose(2, "Loading data '%s' with labels '%i'" % (v, k))

        t = EEPDataset(filename, labels=k)

        # XXX equalize number of samples (TODO)
        # YYY why do we actually want to equalize at this level?
        t = t.selectSamples(t.chunks < 105)

        if d is None:
            d = t
        else:
            d += t

    return d

#
# Just a simple example of ERP plotting
#
def plot_ERP(ds):
    # sampling rate
    SR = ds.samplingrate
    # data is already trials, this would correspond sec before onset
    pre = -ds.t0
    # number of channels, samples per trial
    nchannels, spt = ds.mapper.mask.shape
    post = spt * 1.0/ SR - pre # compute seconds in trials after onset
    # map from channel name to index
    ch_map = dict(zip(ds.channelids, xrange(nchannels)))
    ch_of_interest = ch_map['Pz']

    # Nice to see the truth behind the bars ;-)
    for errtype in ['std', 'ste']:
        fig = P.figure(facecolor='white', figsize=(8,8))
        fig.clf()
        # tricks to get title above all subplots
        ax = fig.add_subplot(1, 1, 1, frame_on=False);
        ax.axison = False

        P.title("Mimique of Figure 5 with error being %s " % errtype)
        # Lets plot few ERPs. Attempt to replicate Figure 5
        # Misfits:
        #  * range of results is completely different, so what scaling of the data
        #  * ERPs of figures are reported to diverge into positive side although from
        #    our plots we see that first we get N-peak
        #  * ERPs in the article are nice and smooth ;-) here we see the reality. May be ERPs were
        #    computed at different samples rate, rereferenced?
        for i, cond_prefix in enumerate(cond_prefixes):
            l1, l2 = cond_prefix + 'o', cond_prefix + 'n'
            ds_1 = ds.selectSamples(ds.idsbylabels([label2id[l1]]))
            ds_2 = ds.selectSamples(ds.idsbylabels([label2id[l2]]))

            ax = fig.add_subplot(2, 2, i+1, frame_on=False)

            fig = plotERPs(
                ({'label': l1,
                  'data': ds.mapper.reverse(ds_1.samples)[:, ch_of_interest, :],
                  'color': 'r'},
                 {'label': l2,
                  'data': ds.mapper.reverse(ds_2.samples)[:, ch_of_interest, :],
                  'color': 'b'}
                 ), pre=pre, post=post, SR=SR, ax=ax, errtype=errtype,
                #ylim=(-30, 30)
                )

    # XXX yeah... the world is not perfect... with ylim to center them at
    # the same location we get some problems, thus manual tuning
    # remove ylim in plot_erps to get somewhat different result without need for this adjust
    #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.65)
    P.show()

def clfEEG_dummy(ds):
    #
    # Simple classification. Silly one for now
    #
    verbose(1, "Sweeping through classifiers with odd/even splitter for generalization")
    for clf in clfs['binary', '!lars', 'has_sensitivity', 'does_feature_selection']: # lars is too slow
        cv = CrossValidatedTransferError(
            TransferError(clf),
            OddEvenSplitter(),
            enable_states=['confusion', 'training_confusion'])
        verbose(2, "Classifier " + clf.descr)
        for cond_prefix in cond_prefixes:
            l1, l2 = cond_prefix + 'o', cond_prefix + 'n'
            ds_ = ds.selectSamples(ds.idsbylabels([label2id[l1], label2id[l2]]))
            error = cv(ds_)
            tstats = cv.training_confusion.stats
            stats = cv.confusion.stats
            verbose(3, "%s vs %s: Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                    (l1, l2, tstats['ACC'], tstats['MCC'][0],
                     stats['ACC'], stats['MCC'][0]))

def main():
    # TODO we need to make EEPBin available from the EEPDataset
    # DONE some basic assignment of attributes to dsattr

    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess
    ds = loadData()

    # Re-reference the data relative to avg reference... not sure if
    # that would give any result
    do_avgref = True
    if do_avgref:
        ebdata = ds.mapper.reverse(ds.samples)
        ebdata_orig = ebdata
        avg = N.mean(ebdata[:,:-3,:], axis=1)
        ebdata_ = ebdata.swapaxes(1,2)
        ebdata_[:,:,:-3] -= avg[:,:,N.newaxis]
        ebdata = ebdata_.swapaxes(1,2)
        ds.samples = ds.mapper.forward(ebdata)

    do_wavelets = True
    if do_wavelets:
        ebdata = ds.mapper.reverse(ds.samples)
        WT = WaveletTransformationMapper(dim=2)
        ds_orig = ds
        ebdata_wt = WT(ebdata)
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels, chunks=ds_orig.chunks)


    #plot_ERP(ds)

    do_zscore = True
    if do_zscore:
        zscore(ds, perchunk=False)

    clfEEG_dummy(ds)


if __name__ == '__main__':
    main()
