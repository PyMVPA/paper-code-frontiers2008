#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simply functors that transform something."""

__docformat__ = 'restructuredtext'

from warehouse import *
# To read .mat files with data
from scipy.io import loadmat

import os.path

if not locals().has_key('__IP'):
    # If not within IPython
    opt.do_lfp = \
                 Option("--lfp",
                        action="store_true", dest="do_lfp",
                        default=False,
                        help="Either to process LFP instead of spike counts")

    parser.add_options([opt.zscore, opt.do_lfp])
    parser.option_groups = [opts.common, opts.wavelet]
    (options, files) = parser.parse_args()
else:
    class O(object): pass
    options = O()
    options.wavelet_family = None
    options.wavelet_decomposition = 'dwt'
    options.zscore = True # XXX ?
    options.do_lfp = False


datapath = os.path.join(cfg.get('paths', 'data root', default='../data'),
                        'cell.luczak/')
verbose(1, 'Datapath is %s' % datapath)

def loadData():
    # Both lfp and spike counts share the same labels which are
    # stored only in counts datafile. So we need to load both
    filepath = datapath + 'AL22_psth400.mat'
    verbose(1, "Loading Spike counts data from %s" % filepath)
    cell_mat = loadmat(filepath)
    samples =  cell_mat['tc_spk']
    labels = cell_mat['tc_stim']

    if options.do_lfp:
        filepath = datapath + 'tc_eeg_AL22.mat'
        verbose(1, "Loading LFP data from %s" % filepath)
        lfp_mat = loadmat(filepath)
        tc_eeg = lfp_mat['tc_eeg']
        samples = tc_eeg

    d = MaskedDataset(samples=samples, labels=labels)
    # assign descriptions (mapping) of the numerical labels
    tones = (3, 7, 12, 20, 30)
    d.labels_map = dict(
        [('%dkHz' % (tones[i]), i+38) for i in xrange(43-38)] +
        [('song%d' % (i+1), i+43) for i in xrange(48-43)])

    coarsenChunks(d, nchunks=8)         # lets split into 8 chunks

    return d


def preprocess(ds):
    # If we were provided wavelet family to use
    if options.wavelet_family not in ['-1', None]:
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
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels,
                           chunks=ds_orig.chunks)
        # copy labels mapping as well
        ds.labels_map = ds_orig.labels_map

    if options.zscore:
        verbose(2, "Z-scoring full dataset")
        zscore(ds, perchunk=False)

    nf_orig = ds.nfeatures
    ds = removeInvariantFeatures(ds)
    verbose(2, "Removed invariant features. Got %d out of %d features"
            % (ds.nfeatures, nf_orig))

    return ds


def analysis(ds):
    verbose(1, "Running generic pipeline")
    senses = doSensitivityAnalysis(
        ds, {'SMLR': SMLR(descr='SMLR(defaults)')}, {}, NFoldSplitter(),
        sa_args=', combiner=None')
    return senses[0][2], N.array(senses[0][1])


def limshow(data, ax=None, cmap=P.cm.jet, *args, **kwargs):
    """Helper: labeled imshow (to add literal labels as  given in ds)
    """
    ret = ax.imshow(data, cmap=cmap, *args, **kwargs)

    P.yticks(())
    dx = ax.axis()[1]/80
    # plot literal labels
    labels_map_rev = dict([reversed(x) for x in ds.labels_map.iteritems()])
    for i,l in enumerate(ds.UL):
        ax.text(-dx, (len(ds.UL)-i)-0.5, labels_map_rev[l],
               horizontalalignment='right',
               verticalalignment='center')
    cb = P.colorbar(ret, shrink=0.9)
    return ret, cb


def finalFigure(senses):
    # Create  a custom colormap
    RdBu_rev = inverseCmap('RdBu')

    # 1. norm each sensitivity per split/class
    snormed = senses / N.sqrt(N.sum(senses*senses, axis=1))[:, N.newaxis, :]

    # 2. take mean across splits
    smeaned = N.mean(snormed, axis=0)

    sensO = ds.mapReverse(smeaned.T)
    sensOn = L2Normed(sensO)

    # Sum of sensitivities across time bins -- so per each unit/class
    sensOn_perunit1 = N.sum(N.abs(sensOn), axis=1)

    nsx,nsy = 2,2
    fig = P.figure(figsize=(8*nsx, 4*nsy))
    c_n_aspect = 6.0                           # aspect ratio for class x units
    c_tb_aspect = 401/105.0*c_n_aspect         # aspect ratio for class x time

    ckwargs = {'interpolation': 'nearest', 'origin': 'upper'}
    # Lets plot mean counts per each class
    ax = fig.add_subplot(nsy, nsx, 1);

    mcounts, mvar = [], []
    # map data into original space
    dsO = ds.O
    for l in ds.UL:
        dsl = dsO[ds.labels == l, :, :]
        mcounts += [P.mean(P.sum(dsl, axis=2), axis=0)]
        mvar += [N.mean(N.var(dsl, axis=1), axis=0)]
    mcounts = N.array(mcounts)
    mvar = N.array(mvar)

    im,cb = limshow(mcounts, ax=ax, cmap = P.cm.YlOrRd,
                          aspect=c_tb_aspect, vmin=0, **ckwargs)
    ax.set_yticklabels( ( ) )
    P.xlabel('time (ms)')
    P.title('Mean spike counts')

    ax = fig.add_subplot(nsy, nsx, 4);

    vmax = N.max(N.abs(sensOn_perunit1))
    limshow(sensOn_perunit1, ax=ax, cmap=RdBu_rev,
                  aspect=c_n_aspect, vmin=-vmax, vmax=vmax, **ckwargs);
    P.xlabel('Unit')
    P.title('Aggregate units sensitivities')

    ax = fig.add_subplot(nsy, nsx, 2)
    # Var per class/unit
    im,cb = limshow(mvar, ax=ax, cmap = P.cm.YlOrRd,
                    aspect=c_n_aspect, vmin=0, **ckwargs)
    P.xlabel('Unit')
    P.title('Mean variance')

    ax = fig.add_subplot(nsy, nsx, 3);
    sensOn_perunit = N.sum(sensOn_perunit1, axis=0)
    strongest_unit = N.argsort(sensOn_perunit)[-1]

    # Lets plot sensitivities in time bins per each class for the 'strongest'
    sens_unit = sensOn[:, :, strongest_unit]
    mmax = N.max(N.abs(sens_unit))
    im, cb = limshow(sens_unit, ax=ax, cmap=RdBu_rev,
                     aspect=c_tb_aspect, vmin=-mmax, vmax=mmax, **ckwargs)
    P.xlabel('time (ms)')
    P.title('Unit #%d sensitivities' % strongest_unit)

    return fig


if __name__ == '__main__':
    ds = loadData()
    verbose(1, "Dataset for processing summary:\n%s" % ds.summary())
    ds = preprocess(ds)
    confusion, senses = analysis(ds)
    P.figure()
    fig, im, cb = confusion.plot(
        labels=("3kHz","7kHz","12kHz","20kHz","30kHz", None,
                "song1","song2","song3","song4","song5"))
    fig.savefig('figs/cell_luczak-confusion.svg')
    fig = finalFigure(senses)
    fig.savefig('figs/cell_luczak-sens.svg')


