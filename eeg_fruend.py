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
#from mvpa.base import verbose
#from mvpa.datasets.eep import EEPDataset
##from mvpa.misc.eepbin import EEPBin
#from mvpa.misc.plot.erp import *


verbose.level = 4

datapath = '../data/eeg.fruend/ga14/'

# Code our poor labels
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])

# TODO big -- make pymvpa working with string labels

def loadData():
    d = None

    verbose(1, "Loading EEG data from basepath %s" % datapath)
    for k, v in id2label.iteritems():
        t = EEPDataset(datapath + v + '.bin', labels=k)
        verbose(2, "Loaded data '%s' with labels '%i'" % (v, k))

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
def plotERP():
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

            fig = plot_erps(
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

def clfEEG_dummy():
    #
    # Simple classification. Silly one for now
    #
    verbose(1, "Sweeping through classifiers with odd/even splitter for generalization")
    for clf in clfs['binary']:
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


if __name__ == '__main__':
    # TODO we need to make EEPBin available from the EEPDataset
    # DONE some basic assignment of attributes to dsattr

    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess
    ds = loadData()

    SR = 1.0/ds._dsattr['eb_dt']        # sampling rate
    pre = -ds._dsattr['eb_t0']          # data is already trials, this would correspond sec before onset
    nchannels, spt = ds.mapper.mask.shape      # number of channels, samples per trial
    post = spt * 1.0/ SR - pre # compute seconds in trials after onset
    ch_map = dict(zip(ds._dsattr['eb_channels'], xrange(nchannels))) # map from channel name to index

    ch_of_interest = ch_map['Pz']
    #ch_mask = N.zeros(ds.mapper.mask.shape, dtype='b')

    cond_prefixes = ( 'sm', 'sf', 'dm', 'df' )

    #plotERP()
    clfEEG_dummy()
