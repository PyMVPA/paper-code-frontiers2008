#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from mvpa.suite import *

def doSensitivityAnalysis(ds, clfs, sensanas, splitter):
    """Generic function to perform sensitivity analysis (along classification)

    To be used from EEG and MEG scripts.

    :Parameters:
      ds : Dataset
        Dataset to perform analysis on
      clfs : list of Classfier
        Classifiers to take sensitivities (default parameters) of
      sensanas : list of DatasetMeasure
        Additional measures to be computed
      splitter : Splitter
        Splitter to be used for cross-validation
    """
    # absorb all sensitivities
    senses = []

    # run classifiers in cross-validation
    for label, clf in clfs.iteritems():
        cv = \
          CrossValidatedTransferError(
            TransferError(clf),
            splitter,
            harvest_attribs=\
              ['transerror.clf.getSensitivityAnalyzer(force_training=False,' \
               'transformer=None)()'],
            enable_states=['confusion', 'training_confusion'])
        # XXX we might need to device some 'CommonSide' transformer so all
        # sensitivities point to the 'approx' the same side. Now we have them
        # flipped -- SVM vs GPR/SMLR
        # MH: Added crude 'fix' to plotting, as we now where they should point
        #     too

        verbose(1, 'Doing cross-validation with ' + label)
        # run cross-validation
        merror = cv(ds)
        verbose(1, 'Accumulated confusion matrix for out-of-sample tests:\n' +
                str(cv.confusion))

        # get harvested sensitivities for all splits
        sensitivities = N.array(cv.harvested.values()[0])
        # and store
        senses.append(
            (label + ' (%.1f%% corr.) weights' % cv.confusion.stats['ACC%'],
             sensitivities))

    verbose(1, 'Computing additional sensitvities')

    # wrap everything into SplitFeaturewiseMeasure
    # to get sense of variance across our artificial splits
    # compute additional sensitivities
    for k, v in sensanas.iteritems():
        verbose(2, 'Computing: ' + k)
        sa = SplitFeaturewiseMeasure(v, splitter, enable_states=['maps'])
        # compute sensitivities
        sa(ds)
        # and grab them for all splits
        senses.append((k, sa.maps))

    return senses
