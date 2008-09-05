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

from mvpa.suite import *
import os.path
import cPickle

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='data'),
                        'eeg.fruend')
verbose(1, 'Datapath is %s' % datapath)

subj = 'ga14'

sensors = XAVRSensorLocations(os.path.join(datapath, 'xavr1010.dat'))
verbose(1, 'Loaded sensor information')

# Code our poor labels
# XXX: only need id2label
label2id = {'dfn': 1, 'dfo': 2, 'dmn': 3, 'dmo': 4,
             'sfn': 5, 'sfo': 6, 'smn': 7, 'smo': 8}
id2label = dict( [(x[1], x[0]) for x in label2id.iteritems()])

mode = 'color' # object, delayed
target_samplingrate = 200.0


# plotting helper function

# XXX not used -- should migrate into mainline PyMVPA?
# older version is also present in fmri_haxby.py
#
def makeBarPlot(data, labels=None, title=None, ylim=None, ylabel=None,
               width=0.2, offset=0.2, color='0.6', distance=1.0):

    # determine location of bars
    xlocations = (N.arange(len(data)) * distance) + offset

    # work with arrays
    data = N.array(data)

    # plot bars
    plot = P.bar(xlocations,
                 data.mean(axis=1),
                 yerr=data.std(axis=1) / N.sqrt(data.shape[1]),
                 width=width,
                 color=color,
                 ecolor='black')

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xlocations + width / 2, labels)

    if ylabel:
        P.ylabel(ylabel)

    # leave some space after last bar
    P.xlim(0, xlocations[-1] + width + offset)

    return plot


def labels2binlabels(ds, mode):
    try:
        filt = {'delayed': [1, 2, 3, 4],
                'color':   [1, 2, 5, 6],
                'object':  [2, 4, 6, 8]}[mode]
    except KeyError:
        raise ValueError, 'Unknown label recoding mode %s' % mode

    # XXX N.setmember1d should do smth like what we need but it does
    #     smth else ;-)
    # ACTUALLY: this should work:
    # N.logical_or.reduce(dataset.labels[:,None] == filt, axis=1).astype(int)
    # it seems not to be shorter though ;-) but more efficient! (may be ;-))

    ds.labels[:]=N.array([i in filt for i in ds.labels], dtype='int')

    # also we have now where, so smth like
    # l1 = ds.where(labels=[filt]); ds.labels[:] = 0; ds.labels[l1] = 1
    # should do


def loadData(subj):
    ds = []                             # list of datasets

    verbose(1, "Loading EEG data from basepath %s" % datapath)
    for k, v in id2label.iteritems():
        filename = os.path.join(datapath, subj, v + '.bin')
        verbose(2, "Loading data '%s' with labels '%i'" % (v, k))

        ds += [EEPDataset(filename, labels=k)]

    d = reduce(lambda x,y: x+y, ds)     # combine into a single dataset

    verbose(1, 'Limit to binary problem: ' + mode)
    labels2binlabels(d, mode)

    d = d.resample(sr=target_samplingrate)
    verbose(2, 'Downsampled data to %.1f Hz' % d.samplingrate)

    return d


def finalFigure(ds_pristine, ds, senses, channel):
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

    responses = [ ds_pristine['labels', i].O[:, ch_of_interest, :]
                  for i in [0, 1] ]
    dwave = N.array(responses[0].mean(axis=0) - responses[1].mean(axis=0),
                    ndmin=2)
    plotERPs( [{'label':'lineart', 'color':'r', 'data':responses[0]},
               {'label':'picture', 'color':'b', 'data':responses[1]},
               {'label':'dwave',   'color':'0', 'data':dwave, 'pre_mean':0}],
               pre=pre, pre_mean=pre, post=post, SR=SR, ax=ax, errtype=errtype,
              ylformat='%d', xlabel=None)

    # plot sensitivities
    ax = fig.add_subplot(2, 1, 2, frame_on=False)

    sens_labels = []
    erp_cfgs = []
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']

    for i, (sens_id, sens) in enumerate(senses[::-1]):
        sens_labels.append(sens_id)
        # back-project
        backproj = ds.mapReverse(sens)

        # and normalize so that all non-zero weights sum up to 1
        # ATTN: need to norm sensitivities for each fold on their own --
        # who knows what's happening otherwise
        for f in xrange(backproj.shape[0]):
            backproj[f] = L2Normed(backproj[f])

        # take one channel: yields (nfolds x ntimepoints)
        ch_sens = backproj[:, ch_of_interest, :]

        # go with abs(), as negative sensitivities are as important
        # as positive ones
        #ch_sens = Absolute(ch_sens)

        # charge ERP definition
        erp_cfgs.append(
            {'label': sens_id,
             'color': colors[i],
             'data' : ch_sens})

    # just ci95 error here, due to the low number of folds not much different
    # from std; also do _not_ demean based on initial baseline as we want the
    # untransformed sensitivities
    plotERPs(erp_cfgs, pre=pre, post=post, SR=SR, ax=ax, errtype='ci95',
             ylabel=None, ylformat='%.2f', pre_mean=0)

    P.legend(sens_labels)

    # how many sensitivities do we have
    nsens = len(senses)

    # new figure for topographies
    fig = P.figure(facecolor='white', figsize=((nsens+1)*3, 4))

    for i, (sens_id, sens) in enumerate(senses):
        ax = fig.add_subplot(1, nsens+1, i+1, frame_on=False)
        # back-project: yields (nfolds x nchannels x ntimepoints)
        backproj = ds.mapReverse(sens)
        # go with abs(), as negative sensitivities are as important
        # as positive ones...
        # XXX
        # YOH: we can do that only after we avg across splits
        # backproj = Absolute(backproj)

        avgbackproj = backproj.sum(axis=0)

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
        P.clim(vmin=0, vmax=0.4)
        # No need for full title
        # P.title(sens_id + '\n%s=%.3f' % ('Pz', scores[sensors.names.index('Pz')]))
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



if __name__ == '__main__':
    # load dataset for some subject
    ds=loadData(subj)

    # artificially group into chunks
    nchunks = 6
    verbose(1, 'Group data into %i handy chunks' % nchunks)
    coarsenChunks(ds, nchunks)

    # Re-reference the data relative to avg reference... not sure if
    # that would give any result
    do_avgref = False
    if do_avgref:
        verbose(1, 'Rereferencing data')
        ebdata = ds.mapper.reverse(ds.samples)
        ebdata_orig = ebdata
        avg = N.mean(ebdata[:,:-3,:], axis=1)
        ebdata_ = ebdata.swapaxes(1,2)
        ebdata_[:,:,:-3] -= avg[:,:,N.newaxis]
        ebdata = ebdata_.swapaxes(1,2)
        ds.samples = ds.mapper.forward(ebdata)

    verbose(1, 'A-priori feature selection')
    # a-priori feature selection
    mask = ds.mapper.getMask()
    # throw away EOG channels
    mask[-3:] = False
    # throw away timepoints prior onset
#    mask[:, :int(-ds.t0 * ds.samplingrate)] = False

    print ds.summary()
    # apply selection
    ds = ds.selectFeatures(ds.mapForward(mask).nonzero()[0])
    print ds.summary()

    do_zscore = True
    if do_zscore:
        verbose(1, 'Z-scoring')
        zscore(ds, perchunk=True)
    print ds.summary()

    do_analyses = True
    if do_analyses == True:
        # eats all sensitivities
        senses = []

        # splitter to use for all analyses
        splttr = NFoldSplitter()

        # some classifiers to test
        clfs = {
            # explicitly instruct SMLR just to fit a single set of weights for our binary task
            'SMLR': SMLR(lm=0.1, fit_all_weights=False),
            'lCSVM': LinearCSVMC(),
            # 'sglCSVM': sg.SVM(), # lets see if we observe the same flip effect
            'lGPR': GPR(kernel=KernelLinear()),
            }

        # run classifiers in cross-validation
        for label, clf in clfs.iteritems():
            cv = \
              CrossValidatedTransferError(
                TransferError(clf),
                splttr,
                harvest_attribs=\
                  ['transerror.clf.getSensitivityAnalyzer(force_training=False, transformer=None)()'],
                enable_states=['confusion', 'training_confusion'])
            # we might need to device some 'CommonSide' transformer so all
            # sensitivities point to the 'approx' the same side. Now we have them flipped -- SVM vs GPR/SMLR

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
        # define some pure sensitivities (or related measures)
        sensanas={
                  'ANOVA': OneWayAnova(),
                  # Crashes for Yarik -- I guess openopt issue
                  #'GPR_Model': GPRWeights(GPR(kernel=KernelLinear()), combiner=None),
                  #
                  # no I-RELIEF for now -- takes too long
                  #'I-RELIEF': IterativeReliefOnline(),
                  # gimme more !!
                 }

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

    # and finally plot figure for channel of choice
    finalFigure(ds_pristine, ds, senses, 'Pz')
