#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:

#
# Main idea for MEG analysis example might be -- finding best wavelet component(s) providing
# nice generalization
#

from mvpa.suite import *
from warehouse import doSensitivityAnalysis

if not locals().has_key('__IP'):
    opt.verbose.default = 3                    # for now
    parser.add_options([opt.zscore, opt.do_sweep])
    parser.option_groups = [opts.common, opts.wavelet]
    (options, files) = parser.parse_args()
else:
    class O(object): pass
    options = O()
    options.wavelet_family = None #'db1'
    options.wavelet_decomposition = 'dwp'
    options.zscore = True
    options.do_sweep = False

verbose.level = 4

# set MVPA_PATHS_DATA_ROOT accordingly!
datapath = cfg.get('paths', 'data root', default='data/meg.rieger')
verbose(1, 'Datapath is %s' % datapath)

conditions = {'false': 'cosu', 'unsure': 'fausu'}
subj = 'vp02'
# sampling rate after preprocessing in Hz
target_samplingrate = 80


def loadData(subj):
    datasets = []

    for cond_id, cond in conditions.iteritems():
        verbose(1, 'Loading data for condition %s:%s' % (cond, cond_id))
        # for now just with tiny dataset
        meg = TuebingenMEG(os.path.join(datapath,
                                        subj + 'cf-3f' + cond + '.dat.gz'))

        # just select MEG channels
        data = meg.data[:, [i for i, v in enumerate(meg.channelids)
                                            if v.startswith('M')]]
        # keep list of corresponding channel ids
        channelids = [i for i in meg.channelids if i.startswith('M')]

        verbose(2, 'Selected %i channels' % data.shape[1])

        datasets.append(ChannelDataset(samples=data,
                            labels=[cond_id] * len(data),
                            labels_map=True,
                            channelids=channelids, dt=1./meg.samplingrate,
                            t0=meg.timepoints[0]))

    # merge all datasets
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d
    # set uniq chunk id per each sample
    dataset.chunks = N.arange(dataset.nsamples)

    dataset = dataset.resample(sr=target_samplingrate)
    verbose(2, 'Downsampled data to %.1f Hz' % dataset.samplingrate)

    # substract the baseline from the data; uses t0 to determine the length
    # of the baseline window
    # XXX: shouldn't this be done per chunk?
    dataset.substractBaseline()
    verbose(2, 'Substracted %f sec baseline' % N.abs(dataset.t0))

    # XXX shorten a bit please
    #
    # select time window of interest: from onset to 600 ms after onset
    mask = dataset.mapper.getMask()
    # deselect timespoints prior to onset
    mask[:, :int(N.round(-dataset.t0 * dataset.samplingrate))] = False
    # deselect timepoints after 600 ms after onset
    mask[:, int(N.round((-dataset.t0 + 0.6) * dataset.samplingrate)):] = False
    # finally transform into feature selection list
    mask = dataset.mapForward(mask).nonzero()[0]
    # and apply selection
    dataset = dataset.selectFeatures(mask)
    verbose(2, 'Applied a-priori feature selection, ' \
               'leaving %i timepoints per channel' % dataset.mapper.dsshape[1])

    # We might want to rechunk differently and coarsen the chunks
    if True:
        # arbitrarily group the sample into chunks
        # done to make the cross-validation a bit quicker; using 10 chunks
        # yields about the same performance as a full 'leave-really-only-a-
        # single-sample-out cross-validation'
        # samples are distributed, so that each chunk contains at least one
        # of each condition
        #nchunks = min(dataset.samplesperlabel.values())

        # hmm, the code below does a better job than coarsenChunks, wrt to
        # equalized distribution of samples....
        nchunks = 10
        for l in dataset.uniquelabels:
            dataset.chunks[dataset.labels==l] = \
                N.arange(N.sum(dataset.labels == l)) % nchunks

    # If we decide to do searchlight, so we get 'generalization' per
    # each time point after onset, using all sensors
    #
    #  POSTPONED
    #
    # dataset.mapper.setMetric(
    #     DescreteMetric(elementsize=[1, 100],
    #                    # distance function should be some custom one
    #                    # to don't count different time points at all
    #                    distance_function=cartesianDistance))

    return dataset

def preprocess(ds):
    """Additional preprocessing
    """
    if options.wavelet_family is not None:
        verbose(2, "Converting into wavelets family %s."
                % options.wavelet_family)
        ebdata = ds.mapper.reverse(ds.samples)
        kwargs = {'dim': 1, 'wavelet': options.wavelet_family}
        if options.wavelet_decomposition == 'dwt':
            verbose(3, "Doing DWT")
            WT = WaveletTransformationMapper(**kwargs)
        else:
            verbose(3, "Doing DWP")
            WT = WaveletPacketMapper(**kwargs)
        ds_orig = ds
        ebdata_wt = WT(ebdata)
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels, chunks=ds_orig.chunks)
        ds.labels_map = ds_orig.labels_map

    verbose(1, 'Precondition data')
    doZScore = True
    if doZScore == True:
        zscore(ds, perchunk=False)
    else:
        # Just divide by max value among first 33 sensors (34 and 91 seems to
        # be bad, thus we need to exclude them)
        ds.samples *= 1.0/N.max(N.abs(ds.O[:,:33,:]))
    print ds.summary()

    doSelectNFeaturesAnova = True
    if doSelectNFeaturesAnova:
        # For now lets just cheat and do on the whole ds, although it
        # doesnt bias selection much (if at all) if later on we just do
        # LOO testing (instead of coarse chunks)
        verbose(1, 'Cruel feature selection')
        ss = SensitivityBasedFeatureSelection(
            OneWayAnova(),
            FractionTailSelector(0.10, mode='select', tail='upper')
            #FixedNElementTailSelector(2, mode='select', tail='upper')
            )
        ds = ss(ds)[0]
        print ds.summary()

    return ds


def clfSweep(ds):
    """Test various classifiers
    """
    # Test few classifiers
    best = {}
    # libsvr never converges for some reason
    # for clf in clfs['linear', '!lars', '!blr', '!libsvr', '!meta']:
    for clf in [sg.SVM(kernel_type='linear')]:
        # C=-2.0 gives 84% when properly scaled and 5% ANOVA voxels
        # C=-1.0 and RFE gives up to 85% correct
        # clf = sg.SVM(kernel_type='linear')
        C = -2.0

        # Scale C according  to the number of samples per class
        spl = ds.samplesperlabel
        ratio = N.sqrt(float(spl[0])/spl[1])
        clf.C = (C/ratio, C*ratio)

        #
        # Two flavors of RFE:
        #
        # This classifier will do RFE while taking transfer error to testing
        # set of that split. Resultant classifier is voted classifier on top
        # of all splits, let see what that would do ;-)
        #rfesvm = sg.SVM(kernel_type='linear')
        #rfesvm.C = clf.C
        #clf = \
        #  SplitClassifier(                      # which does splitting internally
        #   FeatureSelectionClassifier(
        #    clf = clf,
        #    feature_selection = RFE(             # on features selected via RFE
        #        sensitivity_analyzer=\
        #            rfesvm.getSensitivityAnalyzer(transformer=Absolute),
        #        transfer_error=TransferError(rfesvm),
        #        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
        #        feature_selector=FractionTailSelector(
        #                           0.2, mode='discard', tail='lower'),
        #                           # remove 20% of features at each step
        #        update_sensitivity=True)),
        #        # update sensitivity at each step
        #    descr='LinSVM+RFE(N-Fold)')

        #rfesvm_split = SplitClassifier(rfesvm)
        #clf = FeatureSelectionClassifier(
        # clf = clf,
        # feature_selection = RFE(             # on features selected via RFE
        #     # based on sensitivity of a clf which does splitting internally
        #     sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(
        #         transformer=Absolute),
        #     transfer_error=ConfusionBasedError(
        #        rfesvm_split,
        #        confusion_state="confusion"),
        #        # and whose internal error we use
        #     feature_selector=FractionTailSelector(
        #                        0.2, mode='discard', tail='lower'),
        #                        # remove 20% of features at each step
        #     update_sensitivity=True),
        #     # update sensitivity at each step
        # descr='LinSVM+RFE(splits_avg, static)' )

        cv2A = CrossValidatedTransferError(
                  TransferError(clf),
                  NFoldSplitter(),
                  enable_states=['confusion', 'training_confusion', 'splits'])

        verbose(1, "Running cross-validation on %s" % clf.descr)
        error2A = cv2A(ds)
        verbose(2, "Figure 2A LOO performance:\n%s" % cv2A.confusion)
        if best.get('2A', (100, None, None))[0] > error2A:
            best['2A'] = (error2A, cv2A.confusion, clf.descr)

        # assign original single C
        clf.C = C
        # to get results from Figure2B
        cv2B = CrossValidatedTransferError(
                  TransferError(clf),
                  NFoldSplitter(nperlabel='equal',
                                # increase to reasonable number
                                nrunspersplit=4),
                  enable_states=['confusion', 'training_confusion'])

        error2B = cv2B(ds)

        verbose(2, "Figure 2B LOO performance:\n%s" % cv2B.confusion)
        if best.get('2B', (100, None, None))[0] > error2B:
            best['2B'] = (error2B, cv2B.confusion, clf.descr)


    verbose(1, "Best result for 2A was %g achieved on %s, and for 2B " \
            "was %g achieved using %s" %
            (best['2A'][0], best['2A'][2],
             best['2B'][0], best['2B'][2]))


def analysis(ds):

    # Lets first replicate the obtained resuls.  We can do slightly
    # better using RFEs and initial feature selection, but lets just
    # replicate
    #
    clf = sg.SVM(kernel_type='linear')
    # C=-2.0 gives 84% when properly scaled and 5% ANOVA voxels
    # C=-1.0 and RFE gives up to 85% correct
    # clf = sg.SVM(kernel_type='linear')
    C = -2.0

    # Scale C according  to the number of samples per class
    spl = ds.samplesperlabel
    ratio = N.sqrt(float(spl[0])/spl[1])
    clf.C = (C/ratio, C*ratio)

    # If we were only to do classification, following snippet is sufficient.
    # But lets reuse doSensitivityAnalysis
    #
    # cv2A = CrossValidatedTransferError(
    #           TransferError(clf),
    #           NFoldSplitter(),
    #           enable_states=['confusion', 'training_confusion', 'splits'])
    #
    # verbose(1, "Running cross-validation on %s" % clf.descr)
    # error2A = cv2A(ds)
    # verbose(2, "Figure 2A LOO performance:\n%s" % cv2A.confusion)

    # Used in RFE implementations
    rfesvm = sg.SVM(kernel_type='linear')
    rfesvm.C = clf.C
    rfesvm_split = SplitClassifier(rfesvm)
    clfs = {
        # explicitly instruct SMLR just to fit a single set of weights for our
        # binary task
        'SMLR': SMLR(lm=0.1, fit_all_weights=False),
        'lCSVM': clf,
        'lGPR': GPR(kernel=KernelLinear()),
        'lCSVM+RFE(farm)': SplitClassifier( # which does splitting internally
           FeatureSelectionClassifier(
            clf = clf,
            feature_selection = RFE(             # on features selected via RFE
                sensitivity_analyzer=\
                    rfesvm.getSensitivityAnalyzer(transformer=Absolute),
                transfer_error=TransferError(rfesvm),
                stopping_criterion=FixedErrorThresholdStopCrit(0.05),
                feature_selector=FractionTailSelector(
                                   0.2, mode='discard', tail='lower'),
                                   # remove 20% of features at each step
                update_sensitivity=True)),
                # update sensitivity at each step
            descr='LinSVM+RFE(farm,N-Fold)'),
        'lCSVM+RFE(mean)': FeatureSelectionClassifier(
          clf = clf,
          feature_selection = RFE(             # on features selected via RFE
            # based on sensitivity of a clf which does splitting internally
            sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(
                transformer=Absolute),
            transfer_error=ConfusionBasedError(
               rfesvm_split, confusion_state="confusion"),
               # and whose internal error we use
            feature_selector=FractionTailSelector(
                               0.2, mode='discard', tail='lower'),
                               # remove 20% of features at each step
            update_sensitivity=True),
            # update sensitivity at each step
          descr='LinSVM+RFE(avg,N-Fold)' )
        }

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

    # perform the analysis and get all sensitivities
    senses = doSensitivityAnalysis(ds, clfs, sensanas, NFoldSplitter())

    # assign original single C
    clf.C = C
    # get results from Figure2B with resampling of the samples to
    # ballance number of samples per label
    cv2B = CrossValidatedTransferError(
              TransferError(clf),
              NFoldSplitter(nperlabel='equal',
                            # increase to reasonable number
                            nrunspersplit=4),
              enable_states=['confusion', 'training_confusion'])

    error2B = cv2B(ds)

    verbose(2, "Figure 2B LOO performance:\n%s" % cv2B.confusion)

    # Sure we repeat ourselves here but for the sake of clarity
    return senses


def finalFigure(ds_pristine, ds, senses, channel):
    return fig

if __name__ == '__main__':
    # load the only subject that we have
    verbose(1, 'Loading data for subject: ' + subj)
    ds = loadData(subj)
    ds_pristine = ds.copy()
    print ds.summary()
    ds = preprocess(ds)
    if options.do_sweep:
        clfSweep(ds)
    else:
        senses = analysis(ds)
        finalFigure(ds_pristine, ds, senses, 0)




