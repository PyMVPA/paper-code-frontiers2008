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
    # eats all sensitivities
    senses = []

    # splitter to use for all analyses
    splttr = splitter

    # run classifiers in cross-validation
    for label, clf in clfs.iteritems():
        cv = \
          CrossValidatedTransferError(
            TransferError(clf),
            splttr,
            harvest_attribs=\
              ['transerror.clf.getSensitivityAnalyzer(force_training=False,' \
               'transformer=None)()'],
            enable_states=['confusion', 'training_confusion'])
        # we might need to device some 'CommonSide' transformer so all
        # sensitivities point to the 'approx' the same side. Now we have them
        # flipped -- SVM vs GPR/SMLR
        # MH: Added crude 'fix' to plotting, as we now where they should point
        #     too

        verbose(1, 'Doing cross-validation with ' + label)
        # run cross-validation
        merror = cv(ds)
        verbose(1, 'Accumulated confusion matrix for out-of-sample tests')
        print cv.confusion

        # get harvested sensitivities for all splits
        sensitivities = N.array(cv.harvested.values()[0])
        # and store
        senses.append(
            (label + ' (%.1f%% corr.) weights' \
                % cv.confusion.stats['ACC%'],
             sensitivities))

    verbose(1, 'Computing additional sensitvities')
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

    return senses

#
# Some functions born while analyzing data and which might find its
# place in mainline PyMVPA
#

def plot_ds_perchunk(ds, clf_labels=None):
    """Quick plot to see chunk sctructure in dataset with 2 features

    if clf_labels is provided for the predicted labels, then
    incorrectly labeled samples will have 'x' in them
    """
    if ds.nfeatures != 2:
        raise ValueError, "Can plot only in 2D, ie for datasets with 2 features"
    if P.matplotlib.get_backend() == 'TkAgg':
        P.ioff()
    if clf_labels is not None and len(clf_labels) != ds.nsamples:
        clf_labels = None
    colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    labels = ds.uniquelabels
    labels_map = dict(zip(labels, colors[:len(labels)]))
    for chunk in ds.uniquechunks:
        chunk_text = str(chunk)
        ids = ds.where(chunks=chunk)
        ds_chunk = ds[ids]
        for i in xrange(ds_chunk.nsamples):
            s = ds_chunk.samples[i]
            l = ds_chunk.labels[i]
            format = ''
            if clf_labels != None:
                if clf_labels[i] != ds_chunk.labels[i]:
                    P.plot([s[0]], [s[1]], 'x' + labels_map[l])
            P.text(s[0], s[1], chunk_text, color=labels_map[l],
                   horizontalalignment='center',
                   verticalalignment='center',
                   )
    dss = ds.samples
    P.axis((1.1*N.min(dss[:,0]), 1.1*N.max(dss[:,1]), 1.1*N.max(dss[:,0]), 1.1*N.min(dss[:,1])))
    P.draw()
    P.ion()


def inverseCmap(cmap_name):
    """Create a new colormap from the named colormap, where it got reversed

    """
    import matplotlib._cm as _cm
    import matplotlib as mpl
    try:
        cmap_data = eval('_cm._%s_data' % cmap_name)
    except:
        raise ValueError, "Cannot obtain data for the colormap %s" % cmap_name
    new_data = dict( [(k, [(v[i][0], v[-(i+1)][1], v[-(i+1)][2])
                           for i in xrange(len(v))])
                      for k,v in cmap_data.iteritems()] )
    return mpl.colors.LinearSegmentedColormap('%s_rev' % cmap_name, new_data, _cm.LUTSIZE)


# Lets add few additional cmdline options

opt.do_sweep = \
             Option("--sweep",
                    action="store_true", dest="do_sweep",
                    default=False,
                    help="Either to only sweep through various classifiers")

