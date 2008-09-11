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

datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'fmri.haxby')
verbose(1, 'Datapath is %s' % datapath)

subj = 'subj1'


# read HarvardOxford-Cortical atlas index map
import xml.dom.minidom as md
atlas = md.parse('/usr/share/fsl/data/atlases/HarvardOxford-Cortical.xml')
# and convert into dict (index is shifted by one in volume, correcting here)
atlas = dict([(int(el.getAttribute('index')) + 1,
               el.firstChild.data)
                    for el in atlas.getElementsByTagName('label')])

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
    axes = []

    plts = 1
    for atlas_id in atlas_ids:
        axes.append(P.subplot(2, 4, plts))
        # first use SMLR to determine the number of voxels per ROI to consider
        nvoxels = plotAtlasROISampleDistanceDendrogram(

                        ds,
                        ('SMLR',
                         [s[1] for s in senses if s[0].startswith('SMLR')][0]),
                        atlas_id, atlas_map)
        plts += 1
        axes.append(P.subplot(2, 4, plts))
        plotAtlasROISampleDistanceDendrogram(
                ds,
                ('ANOVA',
                 [s[1] for s in senses if s[0].startswith('ANOVA')][0]),
                 atlas_id, atlas_map, limit=nvoxels)

        P.ylabel(atlas[atlas_id] + '(nvoxels: ' + str(nvoxels) + ')')
        plts += 1
    # maximum distance range 
    ymax = max([ax.get_ylim()[1] for ax in axes])

    for ax in axes:
        ax.set_ylim((0, ymax))


def plotAtlasROISampleDistanceDendrogram(ds, sens, atlas_id, atlas_map,
                                         limit=None):
    # determine ROI mask
    mask = atlas_map == atlas_id

    # perform block/chunk averaging
    m = SampleGroupMapper(fx=FirstAxisMean)
    avg_ds = ds.applyMapper(samplesmapper=m)

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

    plotSampleDistanceDendrogram(
            avg_ds.selectFeatures(mask.nonzero()[0]))

    P.title(sens[0])

    # return the number of used voxels
    return mask.sum()


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
                     labels=[lmap[l] for l in ds.labels],
                     link_color_func=lambda x: 'black',
                     distance_sort=False)
    labels = P.gca().get_xticklabels()
    # rotate labels
    P.setp(labels, rotation=90, fontsize=9)




def loadData(subj):
    verbose(1, "Loading fMRI data from basepath %s" % datapath)

    attr = SampleAttributes(os.path.join(datapath, subj, 'labels.txt'),
                            literallabels=True)
    dataset = \
      NiftiDataset(samples=os.path.join(datapath, subj, 'bold_detrend.nii.gz'),
                   labels=attr.labels,
                   labels_map={'rest': 0, 'face': 1, 'house': 2, 'shoe': 3,
                               'cat': 4, 'scissors': 5, 'scrambledpix': 6,
                               'bottle': 7, 'chair': 8},
                   chunks=attr.chunks,
                   mask=os.path.join(datapath, subj,
                                     'bold_example_brain_mask.nii.gz'))

    # go with just four classes to speed up things -- still multiclass enough
    # only faces, houses, shoes, cats
    dataset = dataset['labels', [1,2,3,4]]

    # speed up even more by just using 6 instead of 12 chunks
    coarsenChunks(dataset, 6)

    return dataset


def plotROISensitivityScores(sens_scores, nmin_rois, nmax_rois, ranks):
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
    legend = []
    max_val = 0
    for i, (sid, scores) in enumerate(sens_scores.iteritems()):
        print sid
        values = [scores[roi] for roi in rois]
        bars = plotBars(values, width=bar_width, offset=bar_offset * (i + 1),
                        color=colors[i], label=sid)
        # determine absolute max for y-axis scaling
        if N.max(values) > max_val:
            max_val = N.max(values)
    P.legend()
    P.ylim((0, max_val * 1.02))

    # use abbrevations for ROIs if available
    for i, r in enumerate(rois):
        if atlas_abbrev.has_key(r):
            rois[i] = atlas_abbrev[r]

    # compute x labels positions
    P.xticks(N.arange(len(rois)) \
             + bar_offset + bar_width * len(sens_scores) / 2.0, rois)
    labels = P.gca().get_xticklabels()
    # rotate labels
    P.setp(labels, rotation=90)



if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData(subj)

    # run common preprocessing
    zscore(ds, perchunk=True, targetdtype='float32')

    verbose(1, 'Dataset after preprocessing:\n%s' % ds.summary())

    do_analyses = False
    if do_analyses == True:
        # some classifiers to test
        clfs = {'SMLR': SMLR(lm=0.1)}
        # define some pure sensitivities (or related measures)
        sensanas={'ANOVA': OneWayAnova()}

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

    atlas_nim = \
        NiftiImage(os.path.join(datapath, subj,
                                'HarvardOxford-cort-maxprob-thr25_bold.nii.gz'))

    atlas_mask = ds.mapForward(atlas_nim.data)

    del (senses[1])
    rank = {}
    sens_scores = {}
    # for all available sensitivities
    for sid, sens in senses:
        sens = N.array(sens)
        # normalize sensitivities per split/fold
        for i, s in enumerate(sens):
            sens[i] = L1Normed(s)
        # generate score dict with atlas ROI names as keys
        scores = [(name, N.sum(Absolute(sens[:, atlas_mask == index]),
                                    axis=1))
                      for index, name in atlas.iteritems()]
        # also store mean sensitivity for ranking ROIs later on
        for id, s in scores:
            if not rank.has_key(id):
                rank[id] = []
            rank[id].append(s.mean())
        # finally store as dict for easy access by ROI
        sens_scores[sid] = dict(scores)

    P.figure()
    plotROISensitivityScores(sens_scores, 3, 20, rank)
    P.ylabel('L1-normed sensitivities')
    P.figure(figsize=(4,7))
    makeFinalFigure(ds, senses, [39, 1, 21, 9], atlas_mask)
