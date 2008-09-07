#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
TODO: need to be stripped down only for Frontier's relevant results
"""

from mvpa.suite import *
from warehouse import doSensitivityAnalysis
# do manually until Yarik commits his storage class
import cPickle
import hcluster as clust

# report everything
verbose.level = 100

# some entertainment during the long waiting times ;-)
debug.active += ["CROSSC"]


datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'fmri.haxby')
verbose(1, 'Datapath is %s' % datapath)

subj = 'subj1'

# if we are to change backend to something which doesn't need DISPLAY
#import matplotlib
#matplotlib.use('pdf')
#matplotlib.use('agg')
#matplotlib.use('svg')

#P.rc("axes", linewidth=1.1)
#P.rc("axes", labelsize=14)
#P.rc("xtick", labelsize=14)
#P.rc("ytick", labelsize=14)
#P.rc("lines", markeredgewidth=0.5)
#P.rc("lines", markersize=4)
#P.rc("lines", linewidth=1)  # this also grows the points/lines in the plot..


#
# This is still a dinosaur! (big one, though)
#
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



def mkdir(filename):
    if not os.path.isdir(filename):
        try:
            verbose(1, "Creating directory " + filename)
            os.mkdir(filename)
        except:
            verbose(2, "Failed to create directory %s. Trying to proceed" % filename)


def plotROISensitivity(senses, roi_mask, roilab):
    ids = []
    colors = ['0.2', '0.6']
    for i, (sid, sens) in enumerate(senses):
        sens = L2Normed(N.array(sens))
        ids.append(sid)
        plotBars((sens[:,roi_mask == roilab['LOC']],
                  sens[:, roi_mask == roilab['TOFC']]),
                  labels=['LOC', 'TOFC'],
                  width=0.2, offset=0.2 * (i+1), color=colors[i])
    P.legend(ids)



def plotSampleDistanceDendrogram(ds):
    # generate map from num labels to literal labels
    lmap = dict([(v, k) for k,v in ds.labels_map.iteritems()])

    # compute distance matrix
    dist = clust.pdist(ds.samples)

    # determine clusters
    link = clust.linkage(dist, 'complete')

    # plot dendrogram with literal labels on leaves
    # this does not work with etch's version of matplotlib
    clust.dendrogram(link, colorthreshold=0,
                     labels=[lmap[l] for l in ds.labels])



def doClusterAnalysis(ds, senses, roi_mask, roilab):
    ids = []
    plots = 1
    nrows = len(senses)
    ncols = len(roilab) + 1

    # selector to get the lowest 50%
    fts = FractionTailSelector(0.5, mode='select', tail='lower')

    # for all ROIs
    for roi, roi_id in roilab.iteritems():
        # first plot for full ROI
        full_roi_ds = ds.selectFeatures((roi_mask == roi_id).nonzero()[0])
        # do block-averaging to bring the number of samples down
        m = SampleGroupMapper(fx=FirstAxisMean)
        full_roi_ds = full_roi_ds.applyMapper(samplesmapper=m)

        P.subplot(nrows, ncols, plots)
        plotSampleDistanceDendrogram(full_roi_ds)
        P.title('full ' + roi)
        plots += 1


        for i, (sid, sens) in enumerate(senses):
            # first average over splits
            sens = N.mean(sens, axis=0)

            # mask sensitivities with ROI
            roi_sens = sens * (roi_mask == roi_id)
            roi_sens[fts(Absolute(sens))] = 0

            # get new dataset only with features in ROI and non-zero
            # sensitivity
            roi_ds = ds.selectFeatures(roi_sens.nonzero()[0])

            # do block-averaging to bring the number of samples down
            m = SampleGroupMapper(fx=FirstAxisMean)
            roi_ds = roi_ds.applyMapper(samplesmapper=m)

            P.subplot(nrows, ncols, plots)
            plots += 1

            plotSampleDistanceDendrogram(roi_ds)

            P.title(sid + ' -- ' + roi)


def plotNonZeroSensitivityDistribution(sens, bins=10):
    ns = []
    for s in sens.T:
        n, bin = P.mlab.hist(s[s.nonzero()], bins)
        ns.append(n)
        if not isSequenceType(bins):
            bins = bin

    colors = ['blue', 'red', 'green', 'cyan']
    for i, n in enumerate(ns):
        plotBars(n, yerr=None, offset=0.2*(i+1), width=0.2, color=colors[i])
    loc, labels = P.xticks()
    P.xticks(loc, ['' for b in bins])


def plotBestPerClassSensitivities(sens, n):
    # determine the n 'best' features for each class
    el = FixedNElementTailSelector(n, mode='select', tail='upper')
    best = []
    for s in sens.T:
        best += el(s).tolist()
    best = N.unique(best)

    colors = ['blue', 'red', 'green', 'cyan']
    for i in range(sens.shape[1]):
        plotBars(sens[best, i], yerr=None, offset=0.2*(i+1), width=0.2,
                 color=colors[i])


def loadData(subj):
    verbose(1, "Loading fMRI data from basepath %s" % datapath)

    attr = SampleAttributes(os.path.join(datapath, subj, 'labels.txt'),
                            literallabels=True)
    dataset = \
      NiftiDataset(samples=os.path.join(datapath, subj, 'bold_detrend.nii.gz'),
                   labels=attr.labels,
                   labels_map=True,
                   chunks=attr.chunks,
                   mask=os.path.join(datapath, subj,
                                     'bold_example_brain_mask.nii.gz'))

    # go with just four classes to speed up things -- still multiclass enough
    # FOR YOUR EYES ONLY
    # classes illegally chosen from this confusion matrix
    #
    # Computed using SMLR(lm=0.1) on full dataset
    #
    # predictions\targets  1.0    2.0    3.0    4.0    5.0    6.0    7.0    8.0
    #             `------ -----  -----  -----  -----  -----  -----  -----  -----
    #         1.0          103     0      3     12      0      0      4      1
    #         2.0           0     106     0      2      0      6      0      5
    #         3.0           0      1     60      3     18      2     25     12
    #         4.0           2      0      5     73      7      0      4      3
    #         5.0           0      0     19      8     49      6     17     24
    #         6.0           0      0      1      1      5     89      3      0
    #         7.0           3      0      8      5     17      5     42     22
    #         8.0           0      1     12      4     12      0     13     41
    #     Per target:     -----  -----  -----  -----  -----  -----  -----  -----
    #          P           108    108    108    108    108    108    108    108
    #          N           756    756    756    756    756    756    756    756
    #          TP          103    106    60     73     49     89     42     41
    #          TN          460    457    503    490    514    474    521    522
    #       SUMMARY:      -----  -----  -----  -----  -----  -----  -----  -----
    #         ACC         0.65
    #         ACC%        65.16
    #      # of sets       12
    #
    # DESTROY AFTER READING ;-)

    # only faces, houses, shoes, cats
    dataset = dataset['labels', [1,2,3,4]]

    # speed up even more by just using 6 instead of 12 chunks
    coarsenChunks(dataset, 6)

    return dataset


if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData(subj)

    # run common preprocessing
    zscore(ds, perchunk=True, targetdtype='float32')

    verbose(1, 'Dataset after preprocessing:\n%s' % ds.summary())

    do_analyses = False
    if do_analyses == True:
        # some classifiers to test
        clfs = {
            'SMLR': SMLR(lm=0.1)
#            'lCSVM': LinearCSVMC(),
            # 'sglCSVM': sg.SVM(), # lets see if we observe the same flip effect
#            'lGPR': GPR(kernel=KernelLinear()),
            }

        # define some pure sensitivities (or related measures)
        sensanas={
                  'ANOVA': OneWayAnova(),
                  # no I-RELIEF for now -- takes too long
#                  'I-RELIEF': IterativeReliefOnline(),
                  # gimme more !!
                 }
        # perform the analysis and get all sensitivities
        senses = doSensitivityAnalysis(ds, clfs, sensanas, NFoldSplitter())

        # save countless hours of time ;-)
        picklefile = open(os.path.join(datapath, subj + '_4cat_pickled.dat'), 'w')
        cPickle.dump(senses, picklefile)
        picklefile.close()
    else: # if not doing analyses just load pickled results
        picklefile = open(os.path.join(datapath, subj + '_4cat_pickled.dat'))
        senses = cPickle.load(picklefile)
        picklefile.close()

    #doClusterAnalysis(ds, senses, roi_mask, roilab)

    # Now take sensitivities and split them into meaningful ROIs. For subj1 this
    # could be: Name (HarvardOxford-Code):
    #  1.   Lateral Occipital Cortex, inferior division (23)
    #  2.   Temporal Occipital Fusiform Cortex (39)
    # both ROIs determined by binarized probabilistic map. Overlapping voxels
    # (with non-zero probabilities for both structures) are exclusively
    # associated with one or the other ROI by choosing the one with the higher
    # probability.
    roilab = {'LOC': 1, 'TOFC': 2}
    roi_mask_nim = NiftiImage(os.path.join(datapath, subj,
                                           'HarvardOxford_loc1_tofc2.nii.gz'))

    roi_mask = ds.mapForward(roi_mask_nim.data)

    # SMLR sensitivity analyzer that fits a weight per feature for each class
    sa = SMLR(lm=0.01,
              fit_all_weights=True).getSensitivityAnalyzer(combiner=N.array)

    # run ROI analysis
    # when running full brain picked voxels are clustered but quite distributed
    # left TOFC seems to be good candidate with sensitivities for all 4 classes
    roi_ds = ds.selectFeatures((roi_mask == roilab['TOFC']).nonzero()[0])

    # get sensitivities for a fit using the whole dataset this time
    # (nfeatures x ncategories)
    sens = sa(roi_ds)

    # store full sensitivities
    #roi_ds.map2Nifti(sens.T).save('smlr_tofc_sens.nii.gz')
    roi_ds.map2Nifti(sens.T).save('smlr_loc_sens.nii.gz')

    # now limit to sensitivities in ROI
    #sens[roi_mask != roilab['TOFC']] = 0
    #sens[roi_mask != roilab['LOC']] = 0

    plotNonZeroSensitivityDistribution(sens, 20)
    plotBestPerClassSensitivities(sens, 10)

    senses = doSensitivityAnalysis(roi_ds, {'SMLR(0.01)': SMLR(lm=0.01)},
                                   {}, NFoldSplitter())

    # generate map from num labels to literal labels
    lmap = dict([(v, k) for k,v in ds.labels_map.iteritems()])

    # for all class combinations
    combinations = getUniqueLengthNCombinations(roi_ds.uniquelabels, 2)
    max_sens = Absolute(sens).max()
    for i, (k, l) in enumerate(combinations):
        P.subplot(2, 3, i+1)
        P.plot(sens[:,k-1], sens[:,l-1], '+')
        P.xlim((-max_sens, max_sens))
        P.ylim((-max_sens, max_sens))
        P.title(lmap[k] + ' vs. ' + lmap[l])

