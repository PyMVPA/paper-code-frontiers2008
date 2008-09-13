#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

__docformat__ = 'restructuredtext'

# import functionality, common to all analyses
from warehouse import *

# configure the data source directory
datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'eeg.fruend')
verbose(1, 'Datapath is %s' % datapath)

# use a single subject for this analysis
subj = 'ga14'

# load names and location information of EEG electrodes
sensors = XAVRSensorLocations(os.path.join(datapath, 'xavr1010.dat'))
verbose(1, 'Loaded sensor information')


# Define mapping of literal to numerical labels
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])

# which binary problem to analyze
mode = 'color' # object, delayed

# set target sampling rate for preprocessing
target_samplingrate = 200.0


def labels2binlabels(ds, mode):
    """Recode labels of a dataset (inplace) to a certain binary problem.

    :Parameters:
      ds: Dataset
        Dataset to be recoded.
      mode: str ('delayed' | 'color' | 'object')
    """
    try:
        filt = {'delayed': [1, 2, 3, 4],
                'color':   [1, 2, 5, 6],
                'object':  [2, 4, 6, 8]}[mode]
    except KeyError:
        raise ValueError, 'Unknown label recoding mode %s' % mode

    ds.labels[:]=N.array([i in filt for i in ds.labels], dtype='int')


def loadData(subj):
    """Load data for one subject and return dataset.

    :Parameter:
      subj: str
        ID of the subject who's data should be loaded.

    :Returns:
      EEPDataset instance.
    """
    # list of datasets
    ds = []

    verbose(1, "Loading EEG data from basepath %s" % datapath)

    # load data from individual files
    for k, v in id2label.iteritems():
        filename = os.path.join(datapath, subj, v + '.bin')
        verbose(2, "Loading data '%s' with labels '%i'" % (v, k))

        ds += [EEPDataset(filename, labels=k)]

    # combine into a single dataset
    d = reduce(lambda x,y: x+y, ds)     # combine into a single dataset

    verbose(1, 'Limit to binary problem: ' + mode)
    labels2binlabels(d, mode)

    # inplace resampling
    d.resample(sr=target_samplingrate)
    verbose(2, 'Downsampled data to %.1f Hz' % d.samplingrate)

    return d


def finalFigure(ds_pristine, ds, senses, channel):
    """Generate final ERP, sensitivity and topography plots

    :Parameters:
      ds_pristine: Dataset
        Original (pristine dataset) used to generate ERP plots
      ds: Dataset
        Dataset as used for the sensitivity analyses to generate
        sensitivity and topography plots
      senses: list of 2-tuples (sensitiv. ID, sensitvities (nfolds x nfeatures)
        The sensitvities used to select a subset of voxels in each ROI
      channel: str
        Id of the channel to be used for ERP and sensitivity plots over time.
    """
    # sampling rate
    SR = ds_pristine.samplingrate
    # data is already trials, this would correspond sec before onset
    pre = -(int(ds_pristine.t0*100)/100.0)   # round to 2 digits
    # number of channels, samples per trial
    nchannels, spt = ds_pristine.mapper.mask.shape
    # compute seconds in trials after onset
    post = spt * 1.0/ SR - pre

    # index of the channel of interest
    ch_of_interest = ds_pristine.channelids.index(channel)

    # error type to use in all plots
    errtype=['std', 'ci95']

    fig = P.figure(facecolor='white', figsize=(12, 6))

    # plot ERPs
    ax = fig.add_subplot(2, 1, 1, frame_on=False)

    # map dataset samples back into original (electrode) space
    responses = [ ds_pristine['labels', i].O[:, ch_of_interest, :]
                  for i in [0, 1] ]
    # compute difference wave between the two conditions
    dwave = N.array(responses[0].mean(axis=0) - responses[1].mean(axis=0),
                    ndmin=2)
    # plot them all at once
    plotERPs( [{'label':'lineart', 'color':'r', 'data':responses[0]},
               {'label':'picture', 'color':'b', 'data':responses[1]},
               {'label':'dwave',   'color':'0', 'data':dwave, 'pre_mean':0}],
               pre=pre, pre_mean=pre, post=post, SR=SR, ax=ax, errtype=errtype,
              ylformat='%d', xlabel=None)

    # plot sensitivities over time
    ax = fig.add_subplot(2, 1, 2, frame_on=False)

    sens_labels = []
    erp_cfgs = []
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']

    # for all available sensitivities
    for i, sens_ in enumerate(senses[::-1]):
        (sens_id, sens) = sens_[:2]
        sens_labels.append(sens_id)
        # back-project into electrode space
        backproj = ds.mapReverse(sens)

        # and normalize so that all non-zero weights sum up to 1
        # ATTN: need to norm sensitivities for each fold on their own --
        # who knows what's happening otherwise
        for f in xrange(backproj.shape[0]):
            backproj[f] = L2Normed(backproj[f])

        # take one channel: yields (nfolds x ntimepoints)
        ch_sens = backproj[:, ch_of_interest, :]

        # sign of sensitivities is up to classifier relabling of the
        # input classes.
        if ch_sens.mean() < 0:
            ch_sens *= -1

        # charge ERP definition
        erp_cfgs.append({'label': sens_id, 'color': colors[i], 'data': ch_sens})

    # just ci95 error here, due to the low number of folds not much different
    # from std; also do _not_ demean based on initial baseline as we want the
    # untransformed sensitivities
    plotERPs(erp_cfgs, pre=pre, post=post, SR=SR, ax=ax, errtype='ci95',
             ylabel=None, ylformat='%.2f', pre_mean=0)

    # add a legend to the figure
    P.legend(sens_labels)

    return fig


def topoFigure(ds, senses):
    """Plot topographies of given sensitivities
    """

    # how many sensitivities do we have
    nsens = len(senses)

    # new figure for topographies
    fig = P.figure(facecolor='white', figsize=((nsens+1)*3, 4))

    # again for all available sensitvities
    for i, sens_ in enumerate(senses):
        (sens_id, sens) = sens_[:2]
        ax = fig.add_subplot(1, nsens+1, i+1, frame_on=False)
        # back-project: yields (nfolds x nchannels x ntimepoints)
        backproj = ds.mapReverse(sens)
        # go with abs(), as negative sensitivities are as important
        # as positive ones...
        # we can do that only after we avg across splits
        avgbackproj = backproj.mean(axis=0)
        # compute per channel scores and average across folds
        # (yields (nchannels, )
        scores = N.sum(Absolute(avgbackproj), axis=1)

        # strip EOG scores (which are zero anyway,
        # as they had been stripped of before cross-validation)
        scores = scores[:-3]

        # and normalize so that all scores squared sum up to 1
        scores = L2Normed(scores)

        # plot all EEG sensor scores
        plotHeadTopography(scores, sensors.locations(),
                           plotsensors=True, resolution=50,
                           interpolation='nearest')
        # ensure uniform scaling
        P.clim(vmin=0, vmax=0.4)
        # No need for full title
        P.title(re.sub(' .*', '', sens_id)) # just plot name

        axis = P.axis()                 # to preserve original size
        # Draw a color 'bar' for the given sensitivity
        ax.bar(-0.4, 0.1, 0.8, 1.4, color=colors[i], edgecolor=colors[i]);
        P.axis(axis)

    ax = fig.add_subplot(1, nsens+1, nsens+1, frame_on=False)
    cb = P.colorbar(shrink=0.95, fraction=0.05, drawedges=False,
                    ticks=[0, 0.1, 0.2, 0.3, 0.4])
    ax.axison = False
    # Expand things a bit
    fig.subplots_adjust(left=0.06, right=1.05, bottom=0.01, wspace=-0.2)
    P.show()

    return fig


if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData(subj)

    # artificially group into chunks
    nchunks = 6
    verbose(1, 'Group data into %i handy chunks' % nchunks)
    coarsenChunks(ds, nchunks)

    verbose(1, 'A-priori feature selection')
    # a-priori feature selection
    mask = ds.mapper.getMask()
    # throw away EOG channels
    mask[-3:] = False

    # apply selection
    ds = ds.selectFeatures(ds.mapForward(mask).nonzero()[0])
    # print short summary to give confidence ;-)
    print ds.summary()

    verbose(1, 'Z-scoring')
    zscore(ds, perchunk=True)
    print ds.summary()

    do_analyses = True
    if do_analyses == True:
        # some classifiers to test
        clfs = {
            # explicitly instruct SMLR just to fit a single set of weights for our
            # binary task
            'SMLR': SMLR(lm=0.1, fit_all_weights=False),
            'lCSVM': LinearCSVMC(),
            'lGPR': GPR(kernel=KernelLinear()),
            }

        # define some pure sensitivities (or related measures)
        sensanas={
                  'ANOVA': OneWayAnova(),
                  'I-RELIEF': IterativeReliefOnline(),
                 }
        # perform the analysis and get all sensitivities
        senses = doSensitivityAnalysis(ds, clfs, sensanas, NFoldSplitter())

        # save countless hours of time ;-)
        picklefile = open(os.path.join(datapath, subj + '_pickled.dat'), 'w')
        cPickle.dump(senses, picklefile)
        picklefile.close()
    else: # if not doing analyses just load pickled results
        picklefile = open(os.path.join(datapath, subj + '_pickled.dat'))
        senses = cPickle.load(picklefile)
        picklefile.close()

    # (re)get pristine dataset for plotting of ERPs
    ds_pristine=loadData(subj)

    # plot figure for channel of choice
    fig_sens = finalFigure(ds_pristine, ds, senses, 'Pz')

    # plot figure for topographies
    fig_topo = topoFigure(ds, senses)
