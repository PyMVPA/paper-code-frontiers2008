#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

# Useful and required pieces
import numpy as N

# if we are to change backend to something which doesn't need DISPLAY
#import matplotlib
#matplotlib.use('pdf')
#matplotlib.use('agg')
#matplotlib.use('svg')

import pylab as P
import sys
import os.path
from nifti import NiftiImage

#P.rc("axes", linewidth=1.1)
#P.rc("axes", labelsize=14)
#P.rc("xtick", labelsize=14)
#P.rc("ytick", labelsize=14)
#P.rc("lines", markeredgewidth=0.5)
#P.rc("lines", markersize=4)
#P.rc("lines", linewidth=1)  # this also grows the points/lines in the plot..

# Dataset handling
from mvpa.datasets.nifti import NiftiDataset
from mvpa.misc.iohelpers import SampleAttributes
from mvpa.datasets.misc import zscore

# Progress reports and debugging
from mvpa.base import verbose
from mvpa.base import debug
verbose.level = 100                         # report everything
# Splitters
from mvpa.datasets.splitter \
    import NFoldSplitter, HalfSplitter, OddEvenSplitter, NoneSplitter

# Classifiers
from mvpa.clfs.base import SplitClassifier, BoostedClassifier
from mvpa.clfs.plr import PLR
from mvpa.clfs.ridge import RidgeReg
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.svm import RbfCSVMC, LinearCSVMC
from mvpa.clfs.knn import kNN
from mvpa.clfs.smlr import SMLR

# Error calculation
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError
from mvpa.clfs.transerror import TransferError

# Feature sensitivity
### CHECK HERE!!!!! ################## from mvpa.algorithms.datameasure import selectAnalyzer, ClassifierBasedSensitivityAnalyzer
from mvpa.measures.anova import OneWayAnova
from mvpa.clfs.libsvm.svm import LinearSVMWeights

# Algorithms
from mvpa.measures.searchlight import Searchlight
from mvpa.featsel.rfe import RFE
from mvpa.featsel.helpers import FixedNElementTailSelector, \
     FractionTailSelector
#from mvpa.algorithms.optthreshold import OptimalOverlapThresholder
from mvpa.misc.transformers import FirstAxisMean



# plotting helper functions
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
    P.axhline(0.5, ls='--', color='0.4')

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations+ width/2, labels)

    if ylabel:
        P.ylabel(ylabel)

    P.xlim(0, xlocations[-1]+width*2)


def runDumbClfCrossvalidation(dataset):
    """
    """
    # enable debug output for searchlight call
    debug.active += ["CROSSC", "CLFFS"]

    # which classifiers to test
    #    clfs = {'Linear SVM': LinearCSVMC(),
    #            'RBF SVM': RbfCSVMC(),
    #            'k-Nearest-Neighbour': kNN(k=5)}
    #    clfs = {'SMLR': SMLR()}
    from mvpa.clfs.warehouse import clfs

    # collect data for plot
    clf_terr = []
    clf_trerr = []
    clf_ttime = []
    clf_trtime = []
    clf_nfeatures = []

    labels = []

    plot_clfs = []
#    for clf in clfs['SVM+RFE/oe']: # clfs["all"]:
    for clf in clfs["all"]:
#    for clf in []:
#    for clf in clfs["SMLR"]:
        clf_descr = clf.descr
        filename = '%s-dumb_clf_comp-%s.nii.gz' % (outprefix, clf.descr.replace(' ', '_'))
        plot_clfs.append( (clf_descr, filename) )
        if os.path.exists(filename):
            verbose(2, 'Skipping actual analysis for %s since we have file %s' % (clf_descr, filename))
            continue
        verbose(2, 'Running: %s' % clf_descr)
        clf.states.enable(['training_time', 'predicting_time', 'feature_ids'])
        # setup leave-one-out cross-validation for each classifier
        # using default error function 'MeanMismatchError'
        harvest_attribs=['transerror.confusion',
                         'transerror.clf.feature_ids',
                         'transerror.clf.training_confusion',
                         'transerror.clf.training_time',
                         'transerror.clf.predicting_time',
                         ]

        if isinstance(clf, BoostedClassifier):
            # harvest internal split/classifier used feature_ids
            clf.harvest_attribs = ['clf.feature_ids']
            harvest_attribs.append('transerror.clf.harvested')

        cv = CrossValidatedTransferError(
                TransferError(clf),
                NFoldSplitter(cvtype=1),
                harvest_attribs=harvest_attribs,
                enable_states=['confusion', 'training_confusion'])

        # run cross-validation
        error = cv(dataset)

        # output list of transfer errors of all splits
        clf_terr.append([c.error for c in
                         cv.harvested['transerror.confusion']])
        clf_trerr.append([c.error for c in
                          cv.harvested['transerror.clf.training_confusion']])

        assert(N.abs(N.mean(clf_terr[-1]) - error) <= 0.0001)

        # total confusion matrix of predictions
        verbose(2, "Training confusion matrix:\n%s" % cv.training_confusion)
        verbose(2, "Confusion matrix:\n%s" % cv.confusion)
        verbose(2, "Classifier transfer errors %s" % `clf_terr[-1]`)

        labels.append(clf_descr)
        clf_trtime.append(
            N.mean(cv.harvested['transerror.clf.training_time']))
        clf_ttime.append(
            N.mean(cv.harvested['transerror.clf.predicting_time']))

        if isinstance(clf, BoostedClassifier):
            # for split classifier we need to access internal split nfeatures
            feature_ids = []
            for innerclf_harvested in \
                    cv.harvested['transerror.clf.harvested']:
                feature_ids += innerclf_harvested['clf.feature_ids']
        else:
            # for simple classifiers
            feature_ids = cv.harvested['transerror.clf.feature_ids']
        print "Number of feature_ids sets is ", len(feature_ids)
        clf_nfeatures.append(N.mean([len(x) for x in feature_ids]))

        votes = N.zeros(dataset.nfeatures)
        Nsplits = len(cv.harvested['transerror.confusion'])
        # next one is different from Nsplits for SplitClassifier
        Nfeatsplits = len(feature_ids)
        assert(len(cv.harvested['transerror.clf.feature_ids']) == Nsplits)

        # Store votes scaled by number of splits
        for ids in feature_ids:
            votes[ids] += 1

        votes *= 1.0/Nfeatsplits

        # ie not full brain all the time
        dataset.map2Nifti(votes).save(filename)


        # just store on each step so we could check even before all of
        # them are done
        if False:                       # barplots are kinda useless here
            verbose(2, 'Did: %s' % `labels`)
            makeBarPlot(clf_terr, labels=labels, ylim=(0,1),
                        ylabel='Prediction error')
            N.array(clf_terr).tofile(outprefix + '-dumb_clf_comp.svg')
            P.savefig(outprefix + '-dumb_clf_comp.svg')


        if True:                        # table is more interesting
            from table import RowTable
            tab = RowTable()
            tab.addColSeparator(0)
            tab.addRow(['kind', 'Classifier', 'Features utilized', 'Training Error',
                        'Training Time(sec)', 'Transfer Error', 'Testing Time(sec)' ])
            tab.defaultColFormat = 'c'
            tab.colFormats = {0:'r'}
            tab.addRowSeparator()

            def mean_pm_stderr(values):
                return '%.2f\small{$\\pm$%.2f}' % \
                       (N.mean(values), N.std(values)/N.sqrt(len(values)))

            for i,label in enumerate(labels):
                tab.addRow(['', label,
                            '%d' % clf_nfeatures[i],
                            '%.2f' % N.mean(clf_trerr[i]),
                            '%.1f' % clf_trtime[i],
                            mean_pm_stderr(clf_terr[i]),
                            '%.1f' % clf_ttime[i],
                            ]
                           )
            fout = open(outprefix + '-dumb_clf_comp.tex', 'w')
            fout.write(tab.getAsLatex())
            fout.close()
        # Lets don't store everything!
        clf.untrain()


    #plot_clfs=[('SMLR(lm=0.1)', 'analysis/master/subj1_simple-dumb_clf_comp-SMLR(lm=0.1).nii.gz'),
    #           ('SMLR(lm=1.0)', 'analysis/master/subj1_simple-dumb_clf_comp-SMLR(lm=1.0).nii.gz'),
    #           ('SMLR(lm=10.0)', 'analysis/master/subj1_simple-dumb_clf_comp-SMLR(lm=10.0).nii.gz'),
    #           ('LinSVM on 5%(SVM)', 'analysis/master/subj1_simple-dumb_clf_comp-LinSVM_on_5%(SVM).nii.gz'),
    #           ('LinSVM on 5%(ANOVA)', 'analysis/master/subj1_simple-dumb_clf_comp-LinSVM_on_5%(ANOVA).nii.gz'),
    #           ('LinSVM on SMLR(lm=10) non-0', 'analysis/master/subj1_simple-dumb_clf_comp-LinSVM_on_SMLR(lm=10)_non-0.nii.gz'),
    #           ('LinSVM+RFE(N-Fold)', 'analysis/master/subj1_simple-dumb_clf_comp-LinSVM+RFE(N-Fold).nii.gz')]

    actuallyplot_clfs=[ 'SMLR(lm=0.1)', 'SMLR(lm=1.0)', 'SMLR(lm=10.0)', 'LinSVM on 5%(SVM)',
                        'LinSVM on 5%(ANOVA)', 'LinSVM+RFE(N-Fold)' ]
    # Nsplits = 12; id_ = "subj1"
    # plot_clfs = [ (x, 'analysis/master/subj1-dumb_clf_comp-%s.nii.gz' % x.replace(' ','_'))
    #               for x in actuallyplot_clfs ]
    plot_clfs = filter(lambda x:x[0] in actuallyplot_clfs, plot_clfs)

    titles = { 'LinSVM on 5%(ANOVA)': '5% of ANOVA',
               'LinSVM on 5%(SVM)': '5% of SVM' }
    # just now create nice figure
    nfigs = len(plot_clfs)
    # open background image
    nimg_bg = NiftiImage(os.path.join(id_, 'anat_brain_mni.nii.gz'))
    nimg_bg_mask = NiftiImage(os.path.join(id_, 'anat_brain_mask_mni.nii.gz'))

    slice_selection = 58
    # and select slice of interest
    slice_bg = nimg_bg.asarray()[:,slice_selection]
    slice_bg_mask = nimg_bg_mask.asarray()[:,slice_selection]

    for fig, (clfname, filename) in enumerate(plot_clfs):
        # make a new subplot for each relevant feature selection
        # TODO -- refactor/unify with searchlight
        r,c = fig/3, fig%3
        P.subplot(2, nfigs/2, r*nfigs/2+c+1)

        P.title(titles.get(clfname, clfname), fontsize='small')

        if not os.path.exists(filename):
            raise RuntimeError, "We must have %s file generated by now" % filename

        filename_mni = filename.replace('.nii.gz','_mni.nii.gz')
        if not os.path.exists(filename_mni):
            mnifile = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
            matfile = os.path.join(id_, 'bold2mni.mat')
            cmd = 'flirt -ref %s -applyxfm -init "%s" -out "%s"' % \
                  (mnifile, matfile, filename_mni) + \
                  ' -paddingsize 0.0 -interp nearestneighbour -in "%s"' % filename
            os.system(cmd)

        if not os.path.exists(filename_mni):
            raise RuntimeError, "We must have %s file generated by now" % filename_mni

        # open result image
        nimg_sl = NiftiImage(filename_mni)
        # and select slice of interest
        slice_sl = nimg_sl.asarray()[:, slice_selection]


        # make background transparent
        slice_bg = N.ma.masked_array(slice_bg,
                                     mask=slice_bg_mask< 1)

        # treshold slice for meaningful value
        slice_sl_ = N.ma.masked_array(slice_sl,
                                     mask=N.logical_or(slice_sl < 0.50,
                                                       slice_bg_mask < 1))

        #P.xlim(-100, slice_sl_.shape[1] + 100)
        P.imshow(slice_bg,
                 interpolation='bilinear',
                 #aspect=1.25,
                 cmap=P.cm.gray,
                 origin='lower')

        P.imshow(slice_sl_,
                 interpolation='nearest',
                 #aspect=1.25,
                 cmap=P.cm.autumn,
                 origin='lower',
                 alpha=0.9)
        P.clim(0.5, 1.0)
        P.colorbar(shrink=0.6)
        # no tick labels
        P.axis('off')
    P.savefig(outprefix + '-dumb_clf_comp_slice.svg', dpi=300)
    #P.show()
    P.close()
    # historgrams
    for fig, (clfname, filename) in enumerate(plot_clfs):
        r,c = fig/3, fig%3
        P.subplot(2, nfigs/2, r*nfigs/2+c+1)
        P.title(titles.get(clfname, clfname), fontsize='small')

        # open result image
        nimg_sl = NiftiImage(filename)
        # and selxect slice of interest
        slice_sl = nimg_sl.asarray()[:, slice_selection]

        #P.subplot(2, nfigs, nfigs+fig+1)a
        P.hist(slice_sl[slice_sl>0.001],
               bins=Nsplits)
        ticks = N.arange(0, 1.1, 0.2)
        P.xticks(ticks, [str(x) for x in ticks])
    P.savefig(outprefix + '-dumb_clf_comp_hist.svg')




def runSearchlight(dataset):
    """
    """
    # enable debug output for searchlight call
    debug.active += ["SLC"]

    # choose classifier
    clf = LinearCSVMC()

    # setup measure to be computed by Searchlight
    # cross-validated mean transfer using an N-1 dataset splitter
    cv = CrossValidatedTransferError(TransferError(clf),
                                     OddEvenSplitter())

    fig=0

    # z-slices
    #slice_selection = {1:66, 5:68, 10:61, 20:64}
    # y-slices
    slice_selection = {1:65, 5:57, 10:67, 20:72}
    # do three radii resulting in spheres consisting of approx. 1, 15, 80 and
    # 675 features
    for radius in [1,5,10,20]:
        # tell which one we are doing
        verbose(2, "Running searchlight with radius: %i ..." % (radius))

        filename = '%s-searchlight_r%s.nii.gz' % (outprefix, `radius`)

        # only compute if there is no ready result
        if not os.path.exists(filename):
            # setup Searchlight with a custom radius
            # radius has to be in the same unit as the nifti file's pixdim
            # property: here we have mm
            sl = Searchlight(cv, radius=radius)

            # run searchlight on example dataset and retrieve error map
            sl_map = N.array(sl(dataset))

            # map sensitivity map into original dataspace and store as NIfTI
            # file
            orig_sl_map = dataset.map2Nifti(sl_map).save(filename)


        filename_mni = filename.replace('.nii.gz','_mni.nii.gz')
        if not os.path.exists(filename_mni):
            mnifile = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
            matfile = os.path.join(id_, 'bold2mni.mat')
            cmd = 'flirt -ref %s -applyxfm -init "%s" -out "%s"' % \
                  (mnifile, matfile, filename_mni) + \
                  ' -paddingsize 0.0 -interp nearestneighbour -in "%s"' % filename
            os.system(cmd)

        if not os.path.exists(filename_mni):
            raise RuntimeError, "We must have %s file generated by now" % filename_mni

        # open background image
        slice_bg = NiftiImage(os.path.join(id_, 'anat_brain_mni.nii.gz'))
        slice_bg_mask = NiftiImage(os.path.join(id_, 'anat_brain_mask_mni.nii.gz'))

        # and select slice of interest
        slice_bg = slice_bg.asarray()[:,slice_selection[radius]]
        slice_bg_mask = slice_bg_mask.asarray()[:,slice_selection[radius]]

        # open result image
        slice_sl = NiftiImage(filename_mni)
        slice_sl_mask = NiftiImage(os.path.join(id_, 'bold_example_brain_mask_mni.nii.gz'))
        # and select slice of interest
        slice_sl = slice_sl.asarray()[:,slice_selection[radius]]
        slice_sl_mask = slice_sl_mask.asarray()[:,slice_selection[radius]]


        # make background transparent
#        slice_bg = N.ma.masked_array(slice_bg,
#                                     mask=slice_bg_mask< 1)

        # treshold slice for meaningful value
        slice_sl = N.ma.masked_array(slice_sl,
                                     mask=N.logical_or(slice_sl > 0.35,
                                                       slice_sl_mask < 1))


        # make a new subplot for each searchlight
        fig += 1
        P.subplot(4,2,fig)

        P.title('Radius %i mm' % radius)

        P.imshow(slice_bg,
                 interpolation='sinc',
                 cmap=P.cm.gray,
                 origin='lower')

        P.imshow(slice_sl,
                 interpolation='nearest',
                 cmap=P.cm.autumn,
                 origin='lower',
                 alpha=0.9)
        P.clim(0, 0.35)
        P.colorbar()
        # no tick labels
        P.axis('off')

        # run searchlight again for selected ROI
        cv_roi = CrossValidatedTransferError(TransferError(clf),
                                             OddEvenSplitter(),
                                             combiner=N.array)
        sl_roi = \
            Searchlight(cv_roi, radius=radius,
                center_ids=[ dataset.mapper.getOutId(coord) for coord in
                ((25, 26, 29),
                 (27, 20,  8),
                 (27, 18,  8),
                 (26, 21, 13),
                 (27, 22, 10))])
        errors_roi = sl_roi(dataset)

        fig += 1
        P.subplot(4,2,fig)
        makeBarPlot(errors_roi, labels=('L1', 'R1', 'R5', 'R10', 'R20'), ylim=(0,0.55),
                    ylabel='Generalisation error')

#    P.savefig(outprefix + '-searchlight_slice.svg', dpi=300)
    P.show()


def runOverlapThresholder(dataset):
    """
    """
    # enable debug output for searchlight call
    debug.active += ["OTHRC"]

    terr_proc = TransferError(LinearCSVMC(C=1.0,
                              disable_states=['training_confusion']))

    # Use a simple and fast ANOVA to compute feature sensitivities
    sensa = OneWayAnova()

    # now prefer number of features
    elements = range(1, 10, 2) + range(10, 200, 10) + range(200, 1000, 100) \
               + range(1000, 10000, 2000) + range(10000, 35000, 5000) \
               + [dataset.nfeatures - 10, dataset.nfeatures]

    # select only meaningful values
    elements = N.array(filter(lambda x:x<=dataset.nfeatures, elements))

    # all thresholding types to be evaluated at each split
    thresholders = [ FixedNElementTailSelector(i, mode='select', tail='upper')
                     for i in elements ]

    inner_stats = ['terr_spread', 'fselected', 'fov']
    map_stats = ['ovstatmaps']
    outer_stats = ['outer_ovterr', 'outer_sprterr', 'outer_fullterr',
                   'ovstatmaps', 'full_statmaps']
    interesting = inner_stats + map_stats

    results = {}
    for k in interesting + outer_stats:
        results[k] = []

    chance_performance = 1.0/len(dataset.uniquelabels)
    chance_error = 1.0 - chance_performance

    # perform feature selection in out cross-validation
    for nfold, (training_ds, validation_ds) in \
            enumerate(NFoldSplitter()(dataset)):

        for k in outer_stats:
            results[k].append([])

        verbose(2, 'Doing validation-fold: %d' % nfold +
              'Working:' + str(training_ds) + 'Validation:' + str(validation_ds))

        othr = OptimalOverlapThresholder(
                    sensa,
                    thresholders,
                    NFoldSplitter(),
                    terr_proc,
                    enable_states=interesting)

        othr(training_ds)

        for i, ovmap in enumerate(othr.ovstatmaps):
            verbose(2, 'Compute outer transfer error for thresholder: %d' % i)
            # get feature ids of overlapping voxels
            overlap_ids = dataset.convertFeatureMask2FeatureIds(ovmap == 1.0)
            spread_ids = dataset.convertFeatureMask2FeatureIds(
                           N.logical_and(ovmap > 0.0,
                                         ovmap < 1.0))

            # transfer error on outer validation set for internally overlapping
            # features
            for errname, ids in ( ('outer_ovterr', overlap_ids),
                                  ('outer_sprterr', spread_ids) ):
                if len(ids)>0:
                    terr = terr_proc(training_ds.selectFeatures(ids),
                                     validation_ds.selectFeatures(ids))
                else:
                    terr = chance_error
                results[errname][nfold].append(terr)
            results['ovstatmaps'][nfold].append(othr.ovstatmaps)

        for k in inner_stats:
            if isinstance(othr.states.get(k), N.ma.MaskedArray):
                results[k].append(othr.states.get(k).filled())
            else:
                results[k].append(othr.states.get(k))

        # finally compare to simply thresholding on the full working dataset
        sensitivities = sensa(training_ds)

        for thr in thresholders:
            full_ids = thr(sensitivities)
            results['outer_fullterr'][nfold].append(
                terr_proc(training_ds.selectFeatures(full_ids),
                          validation_ds.selectFeatures(full_ids)))
            results['full_statmaps'][nfold].append(
                dataset.convertFeatureIds2FeatureMask(full_ids))


    for k in interesting:
        results[k] = FirstAxisMean(results[k])
        results[k] = N.ma.masked_array(
                                results[k],
                                mask=results[k] < 0)

    # select those which are overlapping in any of the outer splits
    # and compute fraction of selections across splits
    # splits x tresholders x features
    ovstats = FirstAxisMean(N.array(results['ovstatmaps']) == 1)
    ovstats = N.mean(ovstats == 1, axis=1)

    fullstats = FirstAxisMean(N.array(results['full_statmaps']) == 1)
    fullstats = N.mean(fullstats == 1, axis=1)

    P.plot(elements, results['terr_spread'])
    P.plot(elements, FirstAxisMean(results['outer_ovterr']))
    P.plot(elements, FirstAxisMean(results['outer_sprterr']))
    P.plot(elements, FirstAxisMean(results['outer_fullterr']))
    P.plot(elements, results['fov'] / results['fselected'])
    P.plot(elements, ovstats / results['fov'])
    P.plot(elements, fullstats / results['fselected'])

    P.ylim(0,1)

    P.savefig(outprefix + '-overlap_thr_nodescr.svg')

    P.legend(['Inner Spread-TError',
              'Outer Overlap-TError',
              'Outer Spread-TError',
              'Outer Full-TError',
              'Inner Overlap Fraction',
              'Outer Overlap Fraction',
              'Outer Full-Overlap Fraction'],
              loc=0)
#    P.show()
    P.savefig(outprefix + '-overlap_thr.svg')


def history2maps(history):
    """Convert history generated by RFE into the array of binary maps

    Example:
      history2maps(N.array( [ 3,2,1,0 ] ))
    results in
      array([[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  0.],
             [ 1.,  1.,  0.,  0.],
             [ 1.,  0.,  0.,  0.]])
    """

    # assure that it is an array
    history = N.array(history)
    nfeatures, steps = len(history), max(history) - min(history) + 1
    history_maps = N.zeros((steps, nfeatures))

    for step in xrange(steps):
        history_maps[step, history>=step] = 1

    return history_maps


def runRFEs(dataset):
    """
    """
    # enable debug output for searchlight call
    debug.active += ["RFE.*"]
    fig = P.figure(figsize=(12, 14), dpi=100)

    for clfname, clf, sens_ana, plotarg in \
            [#('Linear C-SVM', LinearCSVMC(C=1.0), None, 'r'),
             #('Split Linear C-SVM', SplitClassifier(clf=LinearCSVMC(C=1.0)), None, 'g'),
             #('Linear C-SVM/Anova', LinearCSVMC(C=1.0), OneWayAnova(), 'b'),
             ('SMLR(default:.1,1e-3)', SMLR(lm=.1, convergence_tol=1e-3), None, 'k'),
             ('SMLR(.1,1e-4)', SMLR(lm=.1, convergence_tol=1e-4), None, ''),
             ('SMLR(0.5,1e-3)', SMLR(lm=0.5, convergence_tol=1e-3), None, ''),
             ('SMLR(0.5,1e-4)', SMLR(lm=0.5, convergence_tol=1e-4), None, ''),
             ('SMLR(1.0,1e-3)', SMLR(lm=1.0, convergence_tol=1e-3), None, ''),
             ('SMLR(1.0,1e-4)', SMLR(lm=1.0, convergence_tol=1e-4), None, ''),
             #('SMLR', SMLR(), None, 'k'),
             ]:

        if sens_ana is None:
            sens_ana = selectAnalyzer(clf)

        if isinstance(sens_ana, ClassifierBasedSensitivityAnalyzer):
            update_senss = [ True, False ] # try both
        else:
            update_senss = [ False ]    # no need

        for update_sens in update_senss:
            maps, errors, nfeatures = [], [], []
            for split_no, (tr, te) in enumerate(params['outer_splitter'](dataset)):
                verbose(3, "Split #%d Classifier %s" %(split_no, clfname))
                rfe = RFE( sens_ana,
                           TransferError(clf),
                           feature_selector=FractionTailSelector(0.1),
                           stopping_criterion=lambda x: False, # do not stop - go through all
                           update_sensitivity=update_sens, # perform real RFE updating sensitivities at each step
                           enable_states=['history', 'errors', 'nfeatures']
                           )
                str, ste = rfe(tr, te)
                maps.append(history2maps(rfe.history))
                errors.append(rfe.errors)
                nfeatures.append(rfe.nfeatures)

            if not (N.std(N.array(nfeatures), axis=0)==0).all():
                raise ValueError, "RFE should have selected the same number of features in each run!"

            # convert to arrays for easy handling
            nfeatures = N.array(nfeatures[0])            # just first instance is enough
            maps = N.array(maps)
            errors = N.array(errors)

            # compute overlapping number of features across 2 splits
            fov = N.sum(N.mean(maps, axis=0) == 1.0, axis=1)

            clftitle = '%s (%s)' % (clfname, ['static sens', 'updated sens'][int(update_sens)])
            plotarg_ =  plotarg + ['-', '--'][int(update_sens)]
            # plot everything
            P.subplot(211)
            P.semilogx(nfeatures, FirstAxisMean(errors), '%s' % plotarg_, linewidth=1.2,
                   label='%s Error/min=%3.1f%%)' % (clftitle, 100.0*N.min(FirstAxisMean(errors))))
            P.subplot(212)
            P.semilogx(nfeatures, fov * 1.0 / nfeatures, '%s' % plotarg_, linewidth=1.0)
            #                   label='%s Overlap ratio' % (clftitle))

    P.subplot(211)
    P.legend(loc=0)
    P.axis('tight')
    P.ylim(0, 1.0)
    # P.grid(True)
    P.ylabel("Error");

    P.subplot(212)
    P.axis('tight')
    P.ylim(0, 1.0)
    P.ylabel("Overlap");
    #    P.grid(True)
    P.xlabel("Number of features");

    #P.gca().xaxis.grid(True, which='minor')  # minor grid on too
    #P.title("Recursive Feature Elimination (RFE)")
    fig.savefig(outprefix + '-rfe-smlrs.svg')
    #P.show()
    #P.savefig(outprefix + '-rfe-svm.svg')


def removeInvariantFeatures(dataset):
    """Helps if a mask does not really fit."""
    f = OneWayAnova()(dataset)
    no_problem = N.logical_not(N.logical_or(N.isinf(f), N.isnan(f)))
    return dataset.selectFeatures(no_problem.nonzero()[0])


def mkdir(filename):
    if not os.path.isdir(filename):
        try:
            verbose(1, "Creating directory " + filename)
            os.mkdir(filename)
        except:
            verbose(2, "Failed to create directory %s. Trying to proceed" % filename)

def doPreproc(dataset, baselinelabels=[0]):
    verbose(1, 'Original dataset: %s' % `dataset`)
    dataset = removeInvariantFeatures(dataset)
    verbose(1, 'Removed features with no within group variance: %s' % `dataset`)

    verbose(1, "zscore with baseline from label(s) '%s'" % `baselinelabels`)
    zscore(dataset, perchunk=True, baselinelabels=baselinelabels,
           targetdtype='float32')

    verbose(1, "Remove baseline samples")
    dataset = dataset.selectSamples( \
        N.array([ l not in baselinelabels for l in dataset.labels],
                dtype='bool'))
    verbose(1, 'Preprocessed dataset: %s' % `dataset`)

    return dataset


if len(sys.argv) < 2:
    print "Need dataset id."
    sys.exit(1)


id_ = sys.argv[1].rstrip('/')
dirname = os.path.join('analysis', os.path.basename(sys.argv[0])[:-3])
mkdir(dirname)
outprefix = os.path.join(dirname, id_)
#outprefix = os.path.join('analysis', os.path.basename(sys.argv[0])[:-3]+".20080221",id)

# specifics per dataset

if id_.startswith('fiac'):
    params = { 'attr_file' : 'numlabels_chunked.txt',
               'data_file' : 'blockbold_detrend.nii.gz',
               'mask_file' : 'example_func_mask.nii.gz',
               'labels_filter' : lambda l: l not in [99],
               'outer_splitter' : HalfSplitter()
               }
else:
    params = { 'attr_file' : 'numlabels.txt',
               'data_file' : 'bold_detrend.nii.gz',
               'mask_file' : 'bold_example_brain_mask.nii.gz',
               'labels_filter' : lambda l: l in [0,4,5],
               #               'outer_splitter' : NFoldSplitter() #OddEvenSplitter()
               'outer_splitter' : OddEvenSplitter()
               }


if id_.startswith('subjnoise'):
    # for those we still have 'face'/'house';-)
    params['labels_filter'] = lambda l: l in [0,1,2]

# load dataset (full-brain)
verbose(1, 'Load dataset')
tmp = os.path.join(id_, params['attr_file'])
print tmp
attr = SampleAttributes(os.path.join(id_, params['attr_file']))
dataset = \
    NiftiDataset(samples=os.path.join(id_, params['data_file']),
                 labels=attr.labels,
                 chunks=attr.chunks,
                 mask=os.path.join(id_, params['mask_file']))

# remove all samples except interesting ones
dataset = dataset.selectSamples(
            N.array([ params['labels_filter'](l) for l in dataset.labels],
            dtype='bool'))

# run common preprocessing
dataset = doPreproc(dataset)

verbose(1, 'Dataset after preprocessing: %s' % dataset)

# let labels be 0,1 to make it easy to use logistic regression
dataset.labels[dataset.labels == 2] = 0

# collection of analyses
#runDumbClfCrossvalidation(dataset)
runSearchlight(dataset)
#runOverlapThresholder(dataset)
#runRFEs(dataset)
