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
from scipy.io import loadmat

import os.path

if not locals().has_key('__IP'):
    opt.do_lfp = \
                 Option("--lfp",
                        action="store_true", dest="do_lfp",
                        default=False,
                        help="Either to process LFP instead of spike counts")

    opt.verbose.default=2                    # for now
    parser.add_options([opt.zscore, opt.do_lfp])
    parser.option_groups = [opts.common, opts.wavelet]
    (options, files) = parser.parse_args()
else:
    class O(object): pass
    options = O()
    options.wavelet_family = None#'db2'
    options.wavelet_decomposition = 'dwt'
    options.zscore = False
    options.do_lfp = False
verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='../data'),
                        'cell.luczak/')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels

def loadData():
    filepath = datapath + 'AL22_psth400.mat'
    verbose(1, "Loading Spike counts data from %s" % filepath)
    cell_mat = loadmat(filepath)
    samples =  cell_mat['tc_spk']
    labels = cell_mat['tc_stim']

    filepath = datapath + 'tc_eeg_AL22.mat'
    verbose(1, "Loading LFP data from %s" % filepath)
    lfp_mat = loadmat(filepath)
    tc_eeg = lfp_mat['tc_eeg']

    d = MaskedDataset(samples=samples, labels=labels)
    d_lfp = MaskedDataset(samples=tc_eeg, labels=labels)
    coarsenChunks(d, nchunks=4)         # lets split into 4 chunks
    coarsenChunks(d_lfp, nchunks=4)
    return d, d_lfp


def clf_dummy(ds):
    #
    # Simple classification. Silly one for now
    #
    verbose(1, "Sweeping through classifiers with NFold splitter for generalization")

    dsc = removeInvariantFeatures(ds)
    verbose(2, "Removed invariant features. Got %d out of %d features" % (dsc.nfeatures, ds.nfeatures))
    best_ACC = 0
    best_MCC = -1
    for clf_ in clfs['smlr', 'multiclass']:
      #clfs['sg', 'svm', 'multiclass'] + clfs['gpr', 'multiclass'] + clfs['smlr', 'multiclass']:
      for clf in [ FeatureSelectionClassifier(
                     clf_,
                     SensitivityBasedFeatureSelection(
                       OneWayAnova(),
                       FractionTailSelector(0.010, mode='select', tail='upper')),
                     descr="%s on 1%%(ANOVA)" % clf_.descr),
                   FeatureSelectionClassifier(
                     clf_,
                     SensitivityBasedFeatureSelection(
                       OneWayAnova(),
                       FractionTailSelector(0.05, mode='select', tail='upper')),
                     descr="%s on 5%%(ANOVA)" % clf_.descr),
                   #FeatureSelectionClassifier(
                   #  clf_,
                   #  SensitivityBasedFeatureSelection(
                   #    OneWayAnova(),
                   #    FractionTailSelector(0.10, mode='select', tail='upper')),
                   #  descr="%s on 10%%(ANOVA)" % clf_.descr),
                   #clf_
                  ]:
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        verbose(2, "Classifier " + clf.descr, lf=False)
        error = cv(dsc)
        tstats = cv.training_confusion.stats
        stats = cv.confusion.stats
        mMCC = N.mean(stats['MCC'])
        if stats['ACC'] > best_ACC:
            best_ACC = stats['ACC']
        if mMCC > best_MCC:
            best_MCC = mMCC
        verbose(3, " Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                (tstats['ACC'], N.mean(tstats['MCC']),
                 stats['ACC'], mMCC))
        if verbose.level > 3:
            print str(cv.confusion)
    verbose(1, "Best results were ACC=%.2g MCC=%.2g" % (best_ACC, best_MCC))

def main():
    # TODO we need to make EEPBin available from the EEPDataset
    # DONE some basic assignment of attributes to dsattr

    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess
    ds, ds_lfp = loadData()

    if options.do_lfp:
        verbose(1, "Working on LFP data instead of spike counts")
        # lets work on LFPs
        ds = ds_lfp

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

    if options.zscore:
        verbose(2, "Z-scoring full dataset")
        zscore(ds, perchunk=True)

    clf_dummy(ds)


if __name__ == '__main__':
    main()
