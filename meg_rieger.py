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
# sampling rate after preprocessing in Hz
target_samplingrate = 80


def loadData(subj):
    datasets = []

    # use to generate unique chunks for all samples
    chunk_counter = 0

    for cond_id, cond in conditions.iteritems():
        # for now just with tiny dataset
        meg = TuebingenMEG(os.path.join(dataroot,
                                        subj + 'cf-3f' + cond + '.dat.gz'))

        # mimic data preprocessing from paper

        # just select MEG channels
        data = meg.data[:, [i for i, v in enumerate(meg.channelids)
                                            if v.startswith('M')]]
        print 'Selected %i channels' % data.shape[1]

        # extract time window of interest (starting at first sample)
        nsamples_toi = int(meg.samplingrate * toi)
        print 'Only using first %i timepoints, corresponding to %.2f s' \
                % (nsamples_toi, nsamples_toi / meg.samplingrate)

        # keep sample, keep channels, select TOI
        data = data[:, :, :nsamples_toi]

        # low-pass filter / resample each channels timeseries
        # cutoff frequency = target_samplingrate/2
        data = resample(data,
                        nsamples_toi * target_samplingrate / meg.samplingrate,
                        window='ham', axis=2)

        # no chunks specified, ie. each sample will be in its own chunks
        datasets.append(MaskedDataset(
                            samples=data, labels=cond_id,
                            chunks=range(chunk_counter,
                                         chunk_counter + data.shape[0])))

        chunk_counter += data.shape[0]

    # merge all datasets
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset += d

    return dataset


# load the only subject that we have
data = loadData('vp02')

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

#cv2A(data)
#cv2B(data)
