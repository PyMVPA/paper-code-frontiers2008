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

import matplotlib as mpl

from mvpa.suite import *
from scipy.io import loadmat

import os.path

if not locals().has_key('__IP'):
    opt.do_lfp = \
                 Option("--lfp",
                        action="store_true", dest="do_lfp",
                        default=False,
                        help="Either to process LFP instead of spike counts")

    opt.do_sweep = \
                 Option("--sweep",
                        action="store_true", dest="do_sweep",
                        default=False,
                        help="Either to only sweep through various classifiers")

    opt.verbose.default=2                    # for now
    parser.add_options([opt.zscore, opt.do_lfp, opt.do_sweep])
    parser.option_groups = [opts.common, opts.wavelet]
    (options, files) = parser.parse_args()
else:
    class O(object): pass
    options = O()
    options.wavelet_family = None
    options.wavelet_decomposition = 'dwt'
    options.zscore = False
    options.do_lfp = False
    options.do_sweep = False

verbose.level = 4

datapath = os.path.join(cfg.get('paths', 'data root', default='../data'),
                        'cell.luczak/')
verbose(1, 'Datapath is %s' % datapath)

# Code our poor labels

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


    coarsenChunks(d, nchunks=8)         # lets split into 4 chunks

    return d


def preprocess(ds):
    # TODO we need to make EEPBin available from the EEPDataset
    # DONE some basic assignment of attributes to dsattr

    # XXX: many things look ugly... we need cleaner interface at few
    # places I guess

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
        ds = MaskedDataset(samples=ebdata_wt, labels=ds_orig.labels, chunks=ds_orig.chunks)
        # copy labels mapping as well
        ds.labels_map = ds_orig.labels_map

    if options.zscore:
        verbose(2, "Z-scoring full dataset")
        # every sample here is independent and we chunked only to
        # reduce the computation time.  there is close to none effect
        # for ANOVA from removing just 1 sample out if we did LOO, so,
        # lets z-score full dataset entirely
        zscore(ds, perchunk=False)

    nf_orig = ds.nfeatures
    ds = removeInvariantFeatures(ds)
    verbose(2, "Removed invariant features. Got %d out of %d features"
            % (ds.nfeatures, nf_orig))

    return ds


def clfSweep(ds):
    """Simple sweep over various classifiers with basic feature
    selections to assess performance
    """

    verbose(1, "Sweeping through classifiers with NFold splitter for generalization")

    dsc = ds
    best_ACC = 0
    best_MCC = -1
    for clf_ in clfs['multiclass']:
      for clf in [ FeatureSelectionClassifier(
                    clf_,
                    SensitivityBasedFeatureSelection(
                      OneWayAnova(),
                      FractionTailSelector(0.010, mode='select', tail='upper')),
                    descr="%s on 1%%(ANOVA)" % clf_.descr),
                   FeatureSelectionClassifier(
                    clf_,
                    SensitivityBasedFeatureSelection(
                      OneWayAnova(),
                      FractionTailSelector(0.05, mode='select', tail='upper')),
                    descr="%s on 5%%(ANOVA)" % clf_.descr),
                   clf_
                 ]:
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        verbose(2, "Classifier " + clf.descr, lf=False)
        error = cv(dsc)
        tstats = cv.training_confusion.stats
        stats = cv.confusion.stats

        mMCC = N.mean(stats['MCC'])
        if stats['ACC'] > best_ACC:
            best_ACC = stats['ACC']
            best_ACC_cm = cv.confusion
        if mMCC > best_MCC:
            best_MCC = mMCC
            best_MCC_cm = cv.confusion

        verbose(2, " Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                (tstats['ACC'], N.mean(tstats['MCC']), stats['ACC'], mMCC))

        if verbose.level > 3:
            print str(cv.confusion)

    verbose(1, "Best results were ACC=%.2g MCC=%.2g" % (best_ACC, best_MCC))
    verbose(2, "Confusion matrix for best result according to ACC:\n%s" % best_ACC_cm)
    verbose(2, "Confusion matrix for best result according to MCC:\n%s" % best_MCC_cm)


def analysis(ds):
    """Simple sweep over various classifiers with basic feature
    selections to assess performance
    """

    verbose(1, "Running favorite classifier with NFold splitter for generalization testing")

    dsc = ds
    best_ACC = 0
    best_MCC = -1

    clf = SMLR(descr='SMLR(defaults)')

    cv = CrossValidatedTransferError(
        TransferError(clf),
        NFoldSplitter(),
        harvest_attribs=\
          ['transerror.clf.getSensitivityAnalyzer(force_training=False, transformer=None, combiner=None)()'],
        enable_states=['confusion', 'training_confusion'])
    verbose(2, "Classifier " + clf.descr, lf=False)
    error = cv(dsc)
    tstats = cv.training_confusion.stats
    stats = cv.confusion.stats
    sensitivities = N.array(cv.harvested.values()[0])

    verbose(2, " Training finished. Training: ACC=%.2g MCC=%.2g, Testing: ACC=%.2g MCC=%.2g" %
                (tstats['ACC'], tstats['mean(MCC)'], stats['ACC'], stats['mean(MCC)']))

    return cv, sensitivities


def imshow_alphad(data, ax=None, cmap=P.cm.jet, alpha_power=4, *args, **kwargs):
    """
    Tried to make use of alpha to make low values invisible. Can plot,
    but couldn't find a way for colorbar to also change it accordingly
    (not alpha for the whole colorrange). Proper way seems to be to
    define custom child of LinearSegmentedColormap which on __call__
    would tune up returned alpha accordingly.

    may be relevant example is on
    http://www.nabble.com/Re:-Trying-p8831162.html

    So for now just plot as is
    """
    method = 'reg'

    # we don't need class ticks here
    #ax.yaxis.set_major_formatter(P.NullFormatter())

    #ax.yaxis.set_major_locator(P.NullLocator())
    if method == 'cmap':
        dumb = cmap(0)                  # force computation of _lut
        # adjust lut to incorporate alpha
        n = cmap._lut.shape[0]-3
        cmap._lut[:n, 3] = 1-(1-N.linspace(0, 1, n, True))**alpha_power
        ret = ax.imshow(data, cmap=cmap, *args, **kwargs)
        # doesn't work since it resets cmap._lut
    elif method == 'alpha_data':
        #data_rgba = mpl.image.AxesImage(ax).to_rgba(data)
        data_rgba = cmap(data)
        # scale alpha by value
        data_rgba[:, :, 3] = 1-(1-data/N.max(data))**alpha_power
        ret = ax.imshow(data_rgba, cmap=cmap, *args, **kwargs)
    else:
        ret = ax.imshow(data, cmap=cmap, *args, **kwargs)

    P.yticks(())
    dx = ax.axis()[1]/80

    labels_map_rev = dict([reversed(x) for x in ds.labels_map.iteritems()])
    for i,l in enumerate(ds.UL):
        ax.text(-dx, (len(ds.UL)-i)-0.5, labels_map_rev[l],
               horizontalalignment='right',
               verticalalignment='center')
    cb = P.colorbar(ret, shrink=0.9)
    return ret, cb


def inverse_cmap(cmap_name):
    """Create a new colormap from the named colormap, where it got reversed"""
    import matplotlib._cm as _cm
    import matplotlib as mpl
    try:
        cmap_data = eval('_cm._%s_data' % cmap_name)
    except:
        raise ValueError, "Cannot obtain data for the colormap %s" % cmap_name
    new_data = dict( [(k, [(v[i][0], v[-(i+1)][1], v[-(i+1)][2])
                           for i in xrange(len(v))])
                      for k,v in cmap_data.iteritems()] )
    return mpl.colors.LinearSegmentedColormap('%s_rev' % cmap_name, new_data, _cm.LUTSIZE)


RdBu_rev = inverse_cmap('RdBu')


def finalFigure(sensitivities):
    # pre-process sensitivities slightly
    # which we should have actually done in transformers

    # 1. norm each sensitivity per split/class
    s = sensitivities
    snormed = s / N.sqrt(N.sum(s*s, axis=1))[:, N.newaxis, :]

    # 2. take mean across splits
    smeaned = N.mean(snormed, axis=0)

    sensO = ds.mapReverse(smeaned.T)
    sensOn = L2Normed(sensO)

    # Sum of sensitivities across time bins -- so per each neuron/class
    sensOn_perneuron1 = N.sum(N.abs(sensOn), axis=1)
    #sensOn_perneuron1 = N.sum(sensOn, axis=1)

    nsx = 2
    nsy = 2
    fig = P.figure(figsize=(8*nsx, 4*nsy))
    c_n_aspect = 6.0                           # aspect ratio for class x neurons
    c_tb_aspect = 401/105.0*c_n_aspect         # aspect ratio for class x time

    dsO = ds.O

    ckwargs = {'interpolation': 'nearest',
               'origin': 'upper'}
    # Lets plot mean counts per each class
    ax = fig.add_subplot(nsy, nsx, 1);
    if True:
        # if plotting with imshow
        mcounts = []
        mvar = []
        for l in ds.UL:
            dsl = dsO[ds.labels == l, :, :]
            mcounts += [P.mean(P.sum(dsl, axis=2), axis=0)]
            mvar += [N.mean(N.var(dsl, axis=1), axis=0)]
        mcounts = N.array(mcounts)
        mvar = N.array(mvar)

        im,cb = imshow_alphad(mcounts, ax=ax, cmap = P.cm.YlOrRd, #P.cm.OrRd,
                              aspect=c_tb_aspect, vmin=0, **ckwargs)
        ax.set_yticklabels( ( ) )
        P.xlabel('Time(ms)')
        #P.ylabel('Class')
        P.title('Mean spike counts')


    if False:
        ax = fig.add_subplot(nsy, nsx, fi); fi += 1

        # plot using plotERPs
        plots = []
        # tones will go in the redish, songs in greenish
        colors = {38: '#550000', 39: '#770000', 40: '#AA0000',
                  41: '#CC0000', 42: '#FF0000',
                  43: '#005500', 44: '#339900', 45: '#00BB33',
                  46: '#33dd00', 47: '#007755'}
        for l in [38, 47]:#ds.UL:
            plots += [{'data':P.sum(dsO[ds.labels == l, :, :], axis=2),
                       'color':colors[l],
                       'label':ds.labels_map[l]}]

        #colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
        # error type to use in all plots
        errtype=['std', 'ci95']
        # TODO: if no post is provided... plotERP fails
        plotERPs( plots, pre=0, pre_mean=0, SR=1, ax=ax, errtype=errtype,
                  ylformat='%.2f', ylabel='Spike count', xlabel='Time', post=dsO.shape[1],
                  legend=True)
        P.axis('off')
        #cb = P.colorbar(drawedges=False, alpha=0.0, ticks=False)
        #P.axis((0.0, 400.0, -1.0, 5.0)) # to size it to match previos one
        #ax.set_aspect(c_tb_aspect*10.0/6)
        #P.draw()

    ax = fig.add_subplot(nsy, nsx, 4);
    # TODO: proper labels on y-axis
    vmax = N.max(N.abs(sensOn_perneuron1))
    imshow_alphad(sensOn_perneuron1, ax=ax, cmap=RdBu_rev, #cmap = P.cm.YlOrRd,
                  aspect=c_n_aspect, vmin=-vmax, vmax=vmax, **ckwargs);
    P.xlabel('Neuron')
    #P.ylabel('Class')
    P.title('Aggregate neurons sensitivities')

    ax = fig.add_subplot(nsy, nsx, 2);
    # Var per class/neuron
    im,cb = imshow_alphad(mvar, ax=ax, cmap = P.cm.YlOrRd, #P.cm.OrRd,
                          aspect=c_n_aspect, vmin=0, **ckwargs)
    P.xlabel('Neuron')
    #P.ylabel('Class')
    P.title('Mean variance')

    ax = fig.add_subplot(nsy, nsx, 3);
    sensOn_perneuron = N.sum(sensOn_perneuron1, axis=0)
    strongest_neuron = N.argsort(sensOn_perneuron)[-1]

    # Lets plot sensitivities in time bins per each class for the 'strongest'
    sens_neuron = sensOn[:, :, strongest_neuron]
    mmax = N.max(N.abs(sens_neuron))
    im, cb = imshow_alphad(sens_neuron, ax=ax, cmap=RdBu_rev,
                  aspect=c_tb_aspect, vmin=-mmax, vmax=mmax, **ckwargs)
    P.xlabel('Time(ms)')
    #P.ylabel('Classes')
    P.title('Neuron #%d sensitivities' % strongest_neuron)

    # widen things up a bit
    #fig.subplots_adjust(hspace=0.2, wspace=0.2,
    #                    left=0.05, right=0.95, top=0.95, bottom=0.5)

if __name__ == '__main__':
    ds = loadData()
    verbose(1, "Dataset for processing summary:\n%s" % ds.summary())
    ds = preprocess(ds)
    if options.do_sweep:
        # To check what we can possibly get with different classifiers
        clfSweep(ds)
    else:
        cv, senses = analysis(ds)
        finalFigure(senses)
        pass


