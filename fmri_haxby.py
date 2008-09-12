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
This file contains the fMRI specific source code of an analysis done for
the paper

  "PyMVPA: A Unifying Approach to the Analysis of Neuroscientific Data"

in the special issue 'Python in Neuroscience' of the journal 'Frontiers
in Neuroinformatics'.
"""

__docformat__ = 'restructuredtext'

# import functionality, common to all analyses
from warehouse import *

# functionality for cluster analyis
import hcluster as clust


# configure the data source directory
datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'fmri.haxby')
verbose(1, 'Datapath is %s' % datapath)

# use a single subject for this analysis
subj = 'subj1'


# read HarvardOxford-Cortical atlas index map, which is provided by the FSL
# package
import xml.dom.minidom as md
atlas = md.parse('/usr/share/fsl/data/atlases/HarvardOxford-Cortical.xml')
# and convert into dict (index is shifted by one in volume, correcting here)
atlas = dict([(int(el.getAttribute('index')) + 1,
               el.firstChild.data)
                    for el in atlas.getElementsByTagName('label')])

# define some abbrevations for structure names to limit their size in the
# final figures
atlas_abbrev = {
    'Lateral Occipital Cortex, inferior division': 'LOC, inf.',
   # need to preserve typo in Atlas label ;-)
   'Lateral Occipital Cortex, superoir division': 'LOC, sup.',
   'Temporal Occipital Fusiform Cortex': 'TOFC',
   'Temporal Fusiform Cortex, posterior division': 'TFC, post.',
   "Heschl's Gyrus (includes H1 and H2)": "Heschl's G.",
   'Angular Gyrus': 'Angular G.',
   'Precuneous Cortex': 'Precuneous',
   'Precentral Gyrus': 'Precentral G.',
   'Postcentral Gyrus': 'Postcentral G.',
   'Superior Temporal Gyrus, anterior division': 'STG, ant.',
   'Lingual Gyrus': 'Lingual G.',
   'Superior Frontal Gyrus': 'SFG',
   'Middle Frontal Gyrus': 'MFG',
   'Occipital Fusiform Gyrus': 'OFG',
   'Superior Parietal Lobule': 'SupParL',
   'Inferior Temporal Gyrus, temporooccipital part': 'ITG, temp-occ.',
   'Cingulate Gyrus, posterior division': 'CG, post.',
   'Inferior Frontal Gyrus, pars triangularis': 'IFG, pt',
   'Paracingulate Gyrus': 'Paracingulate G.',
               }


def makeFinalFigure(ds, senses, atlas_ids, atlas_map):
    """Top-level function to plot cluster dendrograms for four ROIs.

    It takes care of aranging all subplots and labels and scales them
    appropriately.

    :Parameters:
      ds: Dataset
        The dataset providing the samples for this cluster analysis.
      senses: list of 2-tuples (sensitiv. ID, sensitvities (nfolds x nfeatures)
        The sensitvities used to select a subset of voxels in each ROI
      atlas_ids: list of 4 ints
        ROI codes of the selected ROIs
      atlas_map: 1D vector (nfeatures)
        Vector with atlas ROI indeces.
    """
    # store axes of all subplots for uniform scaling  later on
    axes = []

    # for all ROIs
    plts = 1
    for atlas_id in atlas_ids:
        # store axis handle
        axes.append(P.subplot(2, 4, plts))

        # first use SMLR to determine the number of voxels per ROI to consider
        # and plot dendrogram
        nvoxels = plotAtlasROISampleDistanceDendrogram(
                        ds,
                        ('SMLR',
                         [s[1] for s in senses if s[0].startswith('SMLR')][0]),
                        atlas_id, atlas_map)
        plts += 1

        # now same for ANOVA
        axes.append(P.subplot(2, 4, plts))
        plotAtlasROISampleDistanceDendrogram(
                ds,
                ('ANOVA',
                 [s[1] for s in senses if s[0].startswith('ANOVA')][0]),
                 atlas_id, atlas_map, limit=nvoxels)

        # put number of used voxels into axis label
        P.ylabel(atlas[atlas_id] + '(nvoxels: ' + str(nvoxels) + ')')
        plts += 1

    # scale all subplot to the maximum distance range
    ymax = max([ax.get_ylim()[1] for ax in axes])

    for ax in axes:
        ax.set_ylim((0, ymax))


def plotAtlasROISampleDistanceDendrogram(ds, sens, atlas_id, atlas_map,
                                         limit=None):
    """Generate cluster dendrogram plot for a single ROI and a single
    sensitivity vector.

    :Parameters:
      ds: Dataset
        The dataset providing the samples for this cluster analysis.
      senses: 2-tuple (sensitiv. ID, sensitvities (nfolds x nfeatures)
        The sensitvities used to select a subset of voxels in the ROI
      atlas_id: int
        ROI code of the selected ROI
      atlas_map: 1D vector (nfeatures)
        Vector with atlas ROI indeces.
      limit: None | int
        If None, only voxels with non-zero sensitivities are considered. If an
        integer, the voxels are ranked by their sensitivity and the
        corresponding number of voxels with the highest rank is considered.

    :Returns:
      Number of voxels used to compute the distances.
    """
    # determine ROI mask
    mask = atlas_map == atlas_id

    # perform block/chunk averaging
    m = SampleGroupMapper(fx=FirstAxisMean)
    avg_ds = ds.applyMapper(samplesmapper=m)

    # whether to use non-zeros or highest ranked features
    if not limit:
        # only use voxels with nonzero sensitivities
        mask = N.logical_and(mask, Absolute(sens[1].mean(axis=0)) != 0)
    else:
        # limit to the highest scoring voxels
        s = Absolute(N.mean(sens[1], axis=0))
        # kill voxels outside ROI
        s[mask == False] = 0
        s[s.argsort()[:-limit]] = 0

        mask = s > 0

    # plot
    plotSampleDistanceDendrogram(
            avg_ds.selectFeatures(mask.nonzero()[0]))

    # note sensitivity id in title
    P.title(sens[0])

    # return the number of used voxels
    return mask.sum()


def plotSampleDistanceDendrogram(ds):
    """Plot a sample distance cluster dendrogram using all samples and features
    of a dataset.

    :Parameter:
      ds: Dataset
        The source dataset.
    """
    # generate map from num labels to literal labels
    # to put them on the dendrogram leaves
    lmap = dict([(v, k) for k,v in ds.labels_map.iteritems()])

    # compute distance matrix, default is squared euclidean distance
    dist = clust.pdist(ds.samples)

    # determine clusters
    link = clust.linkage(dist, 'complete')

    # plot dendrogram with literal labels on leaves
    # this does not work with etch's version of matplotlib (verified for
    # matplotlib 0.98)
    clust.dendrogram(link, colorthreshold=0,
                     labels=[lmap[l] for l in ds.labels],
                     # all black
                     link_color_func=lambda x: 'black',
                     distance_sort=False)
    labels = P.gca().get_xticklabels()
    # rotate labels
    P.setp(labels, rotation=90, fontsize=9)


def plotROISensitivityScores(sens_scores, nmin_rois, nmax_rois, ranks):
    """Generate a barplot with sensitivity scores plotted for ranked ROIs

    :Parameters:
      sens_scores: list of 2-tuples (str, dict)
        This list contains the scores of all sensitivities.
      nmin_rois: int
        Number of lowest ranked ROIs to plot (on the right)
      nmax_rois: int
        Number of highest ranked ROIs to plot (on the left)
      ranks: dict
        Highest sensitivity score of *any* sensitivity per ROI.
    """
    # determine highest and lowest scoring ROIs
    max_rank = sorted([(k, max(v)) for k, v in ranks.iteritems()],
                        cmp=lambda x,y: -1 * cmp(x[1], y[1]))
    min_rank = sorted([(k, min(v)) for k, v in ranks.iteritems()],
                      cmp=lambda x,y: -1 * cmp(x[1], y[1]))

    # take 'nrois' from highest and lowest ROIs for the plot
    rois = [i[0] for i in max_rank[:nmax_rois]] \
           + [i[0] for i in min_rank[-nmin_rois:]]

    # plot properties
    bar_width = 0.3333
    bar_offset = bar_width
    colors = ['0.3', '0.7']

    # plot for all sensitivities
    max_val = 0
    for i, (sid, scores) in enumerate(sens_scores.iteritems()):
        # extract scores
        values = [scores[roi] for roi in rois]
        # plot bars
        bars = plotBars(values, width=bar_width, offset=bar_offset * (i + 1),
                        color=colors[i], label=sid)
        # determine absolute max for y-axis scaling
        if N.max(values) > max_val:
            max_val = N.max(values)

    # add a legend
    P.legend()
    # scale with some margin to absolute maximum
    P.ylim((0, max_val * 1.02))

    # use abbrevations for ROIs labels if available
    for i, r in enumerate(rois):
        if atlas_abbrev.has_key(r):
            rois[i] = atlas_abbrev[r]

    # compute x labels positions
    P.xticks(N.arange(len(rois)) \
             + bar_offset + bar_width * len(sens_scores) / 2.0, rois)
    labels = P.gca().get_xticklabels()
    # rotate labels
    P.setp(labels, rotation=90)


def loadData(subj):
    """Load data for one subject and return dataset.

    :Parameter:
      subj: str
        ID of the subject who's data should be loaded.

    :Returns:
      NiftiDataset instance.
    """
    verbose(1, "Loading fMRI data from basepath %s" % datapath)

    # load labels and chunk information from file
    # layout: one line per voxel, two rows (label(str), chunk(int)
    # chunk corresponds to the experimental run where the volume was recorded
    attr = SampleAttributes(os.path.join(datapath, subj, 'labels.txt'),
                            literallabels=True)
    # load fMRI data from a NIfTI file, the data was previously detrended
    # using PyMVPA's detrend() function.
    dataset = \
      NiftiDataset(samples=os.path.join(datapath, subj, 'bold_detrend.nii.gz'),
                   labels=attr.labels,
                   # define fixed mapping of literal to numerical labels
                   labels_map={'rest': 0, 'face': 1, 'house': 2, 'shoe': 3,
                               'cat': 4, 'scissors': 5, 'scrambledpix': 6,
                               'bottle': 7, 'chair': 8},
                   chunks=attr.chunks,
                   # load brain mask image to automatically remove non-brain
                   # voxels
                   mask=os.path.join(datapath, subj,
                                     'bold_example_brain_mask.nii.gz'))

    # go with just four classes to speed up things -- still multiclass enough
    # only faces, houses, shoes, cats
    dataset = dataset['labels', [1,2,3,4]]

    # speed up even more by just using 6 instead of 12 chunks, this will put two
    # successive chunks into one.
    coarsenChunks(dataset, 6)

    # done
    return dataset



if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData(subj)

    # run common preprocessing
    zscore(ds, perchunk=True, targetdtype='float32')

    # give status report
    verbose(1, 'Dataset after preprocessing:\n%s' % ds.summary())

    # can be disable to save time, when intermediate results are already stored
    # (see below)
    do_analyses = True
    if do_analyses == True:
        # define classifiers to be used
        clfs = {'SMLR': SMLR(lm=0.1)}
        # define some pure sensitivities (or related measures)
        sensanas={'ANOVA': OneWayAnova()}

        # perform the analysis and get all sensitivities
        # using a generic function common to all analyses, but with custom
        # list of classifiers and measures and custom dataset resampling
        # for the cross-validation procedure
        senses = doSensitivityAnalysis(ds, clfs, sensanas, NFoldSplitter())

        # save countless hours of time ;-)
        picklefile = open(os.path.join(datapath, subj + '_4cat_pickled.dat'), 'w')
        cPickle.dump(senses, picklefile)
        picklefile.close()
    else: # if not doing analyses just load pickled results
        picklefile = open(os.path.join(datapath, subj + '_4cat_pickled.dat'))
        senses = cPickle.load(picklefile)
        picklefile.close()

    # load atlas volume in subject's functional space. The volume is stored in
    # NIfTI format and was derived from the Harvard-Oxford cortical atlas, as
    # shipped with the FSL package
    atlas_nim = \
        NiftiImage(os.path.join(datapath, subj,
                                'HarvardOxford-cort-maxprob-thr25_bold.nii.gz'))

    # transform from volume into features representation (1D vector)
    atlas_mask = ds.mapForward(atlas_nim.data)

    #
    # Post-processing: Compute ROI-wise scores
    #

    # used later on to rank all ROIs by their scores in any of the computed
    # measures
    rank = {}

    # will contain the final scores per ROI
    sens_scores = {}

    # for all available sensitivities
    for sid, sens in senses:
        # convert into NumPy array for easier handling
        sens = N.array(sens)

        # normalize sensitivities per split/fold
        for i, s in enumerate(sens):
            sens[i] = L1Normed(s)

        # generate score dict with atlas ROI names as keys
        # score is the simply sum of absolutes across all voxels in an ROI
        scores = [(name, N.sum(Absolute(sens[:, atlas_mask == index]),
                                    axis=1))
                      for index, name in atlas.iteritems()]

        # also store mean sensitivity for ranking ROIs later on
        for id, s in scores:
            if not rank.has_key(id):
                rank[id] = []
            rank[id].append(s.mean())

        # finally store the scores as dict for easy access by ROI
        sens_scores[sid] = dict(scores)

    # generate figure with ROI-wise sensitivity scores for the 20 highest and 3
    # lowest scoring ROIs
    P.figure()
    plotROISensitivityScores(sens_scores, 3, 20, rank)
    P.ylabel('L1-normed sensitivities')

    # generate figure with cluster dendrograms for four exemplary ROIs
    P.figure(figsize=(4,7))
    makeFinalFigure(ds, senses, [39, 1, 21, 9], atlas_mask)

