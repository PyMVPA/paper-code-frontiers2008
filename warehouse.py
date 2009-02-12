#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

__docformat__ = 'restructuredtext'

# Instruct matplotlib to use sans-serif fonts. DejaVu Sans is our
# choice.  To get an effect - warehouse needs to be imported prior
# pylab, thus prior mvpa.suite.
#
from matplotlib import rc as rcmpl
rcmpl('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})

# import the full PyMVPA suite
from mvpa.suite import *

# efficiently dump and load intermediate results
import cPickle

# report everything
verbose.level = 100


#snippet_start doSensitivityAnalysis
def doSensitivityAnalysis(ds, clfs, sensanas, splitter, sa_args=""):
    """Generic function to perform sensitivity analysis (along classification)

    :Parameters:
      ds : Dataset
        Dataset to perform analysis on
      clfs : list of Classfier
        Classifiers to take sensitivities (default parameters) of
      sensanas : list of DatasetMeasure
        Additional measures to be computed
      splitter : Splitter
        Splitter to be used for cross-validation
      sa_args : basestring
        Additional optional arguments to provide to getSensitivityAnalyzer
    """
    # to absorb all sensitivities
    senses = []

    # run classifiers in cross-validation
    for label, clf in clfs.iteritems():
        sclf = SplitClassifier(clf, splitter,
            enable_states=['confusion', 'training_confusion'])

        verbose(1, 'Doing cross-validation with ' + label)
        # Compute sensitivity, which in turn trains the sclf
        sensitivities = sclf.getSensitivityAnalyzer(
            # do not combine sensitivities across splits, nor across classes
            combiner=None, slave_combiner=None)(ds)

        verbose(1, 'Accumulated confusion matrix for out-of-sample tests:\n' +
                str(sclf.confusion))

        # and store
        senses.append(
            (label + ' (%.1f%% corr.) weights' % sclf.confusion.stats['ACC%'],
             sensitivities, sclf.confusion, sclf.training_confusion))

    verbose(1, 'Computing additional sensitivities')

    # wrap everything into SplitFeaturewiseMeasure
    # to get sense of variance across our artificial splits
    # compute additional sensitivities
    for k, v in sensanas.iteritems():
        verbose(2, 'Computing: ' + k)
        sa = SplitFeaturewiseMeasure(v, splitter, enable_states=['maps'])
        # compute sensitivities
        sa(ds)
        # and grab them for all splits
        senses.append((k, sa.maps, None, None))

    return senses
#snippet_end doSensitivityAnalysis


def Pioff():
    """Turn off pylab redrawing
    """
    # should compare by backend?
    if P.matplotlib.get_backend() == 'TkAgg':
        P.ioff()

def Pion():
    """Turn on pylab redrawing
    """
    if P.matplotlib.get_backend() == 'TkAgg':
        P.draw()
        P.ion()
