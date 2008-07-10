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

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='../data'),
                        'cell.luczak/')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels

def loadData():
    filepath = datapath + 'AL22_psth400.mat'
    verbose(1, "Loading Cell data from %s" % filepath)
    cell_mat = loadmat(filepath)
    samples =  cell_mat['tc_spk']
    labels = cell_mat['tc_stim']
    d = MaskedDataset(samples=samples, labels=labels)
    coarsenChunks(d, nchunks=4)         # lets split into 4 chunks
    return d


def clf_dummy(ds):
    #
    # Simple classification. Silly one for now
    #
    verbose(1, "Sweeping through classifiers with NFold splitter for generalization")

    dsc = removeInvariantFeatures(ds)
    verbose(2, "Removed invariant features. Got %d out of %d features" % (dsc.nfeatures, ds.nfeatures))

    for clf_ in clfs['sg', 'svm', 'multiclass'] + clfs['gpr', 'multiclass'] + clfs['smlr', 'multiclass']:
      for clf in [FeatureSelectionClassifier(
                     clf_,
                     SensitivityBasedFeatureSelection(
                       OneWayAnova(),
                       FractionTailSelector(0.10, mode='select', tail='upper')),
                     descr="%s on 10%%(ANOVA)" % clf_.descr),
#                  clf_
                  ]:
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        verbose(2, "Classifier " + clf.descr, lf=False)
        error = cv(dsc)
        tstats = cv.training_confusion.stats
        stats = cv.confusion.stats
        verbose(3, " Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                (tstats['ACC'], N.mean(tstats['MCC']),
                 stats['ACC'], N.mean(stats['MCC'])))
        if verbose.level > 3:
            print str(cv.confusion)

def main():
    # TODO we need to make EEPBin available from the EEPDataset
    # DONE some basic assignment of attributes to dsattr

    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess
    ds = loadData()

    do_wavelets = False                 # although it might come handy
    if do_wavelets:
        verbose(2, "Converting into wavelets")
        ebdata = ds.mapper.reverse(ds.samples)
        WT = WaveletTransformationMapper(dim=2, wavelet='db1')
        ds_orig = ds
        ebdata_wt = WT(ebdata)
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels, chunks=ds_orig.chunks)

    do_zscore = False
    if do_zscore:
        verbose(2, "Z-scoring full dataset")
        zscore(ds, perchunk=False)

    clf_dummy(ds)


if __name__ == '__main__':
    main()
