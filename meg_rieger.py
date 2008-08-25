#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:

from mvpa.suite import *
from scipy.signal import resample

# set MVPA_PATHS_DATA_ROOT accordingly!
dataroot = cfg.get('paths', 'data root', default='data/meg.rieger')

conditions = {1: 'cosu', 2: 'fausu'}

# time window of interest (in seconds; starting with epoch)
toi = 0.8
# baseline pre-event duration (in seconds; starting with epoch)
tob = 0.2

# sampling rate after preprocessing in Hz
target_samplingrate = 80


def loadData(subj):
    datasets = []

    for cond_id, cond in conditions.iteritems():
        verbose(1, 'Loading data for condition %s' % cond)
        # for now just with tiny dataset
        meg = TuebingenMEG(os.path.join(dataroot,
                                        subj + 'cf-3f' + cond + '.dat.gz'))

        # mimic data preprocessing from the paper

        # just select MEG channels
        data = meg.data[:, [i for i, v in enumerate(meg.channelids)
                                            if v.startswith('M')]]
        verbose(2, 'Selected %i channels' % data.shape[1])

        # Filter/downsample
        # low-pass filter / resample each channels timeseries
        # cutoff frequency = target_samplingrate/2
        data = resample(data,
                        meg.ntimepoints * target_samplingrate / meg.samplingrate,
                        window='ham', axis=2)
        verbose(2, 'Downsampled data to %d Hz' % target_samplingrate)

        # Consider first 200ms to be baseline, thus shift signal accordingly
        nsamples_base = int(target_samplingrate * tob)
        baseline = N.mean(data[:, :, :nsamples_base], axis=2)
        # remove baseline
        data = data - baseline[..., N.newaxis]
        verbose(2, 'Removed %f sec baseline' % tob)

        # extract time window of interest (starting at first sample)
        nsamples_toi = int(target_samplingrate * toi)
        verbose(1, 'Only using first %i timepoints, corresponding to %.2f s' \
                % (nsamples_toi, nsamples_toi / target_samplingrate))

        ### YOH: NOT SURE... and they did select only 600ms after
        ### onset for classification, first 200ms were used for
        ### removing baseline
        ##
        # keep sample, keep channels, select TOI
        data = data[:, :, :nsamples_toi]



        # no chunks specified, ie. each sample will be in its own chunks
        datasets.append(MaskedDataset(
                            samples=data, labels=cond_id,
                            chunks=0))

    # merge all datasets
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d
    # set uniq chunk id per each sample
    dataset.chunks = N.arange(dataset.nsamples)

    return dataset


# load the only subject that we have
data = loadData('vp02')

# Simple preconditioning of the data

# Just divide by max value among first 33 sensors (34 and 91 seems to
# be bad, thus we need to exclude them)
data.samples *= 1.0/N.max(N.abs(
    data.mapper.reverse(data.samples)[:,:33,:]))

# or zscore?
# zscore(data, perchunk=False)

# We might want to rechunk differently and coarsen the chunks
if False:
    data.chunks[data.labels==1] = range(len(data.idsbylabels(1)))
    data.chunks[data.labels==2] = range(len(data.idsbylabels(2)))
    coarsenChunks(data, 10)


# Some basic generalization testing
if False:
    te = TransferError(SMLR(lm=0.01), enable_states=['confusion', 'training_confusion', 'samples_error'])
    splt = NFoldSplitter(nperlabel='equal')
                         #nrunspersplit=2), # increase to reasonable number
    splits = [x for x in splt(data)]
    trd, ted = splits[0]
    error = te(ted, trd)

# to get results from Figure2A
cv2A = CrossValidatedTransferError(
          TransferError(SMLR(lm=0.01)),
          NFoldSplitter(),
          enable_states=['confusion', 'training_confusion'])

# to get results from Figure2B
cv2B = CrossValidatedTransferError(
          TransferError(SMLR(lm=0.01)),
          NFoldSplitter(nperlabel='equal',
                      nrunspersplit=2), # increase to reasonable number
                      enable_states=['confusion', 'training_confusion'])


# zscore -- when every sample is in its own chunk 'perchunk' is pointless
#zscore(data, perchunk=False)

verbose(1, "Running cross-validation on SMLR")
#cv2A(data)
cv2B(data)
