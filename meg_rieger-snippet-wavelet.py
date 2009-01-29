 verbose(2,
         "Converting into wavelets family "
         "%s." % options.wavelet_family)
 ebdata = ds.mapper.reverse(ds.samples)
 kwargs = {'dim': 1,
           'wavelet': options.wavelet_family}
 if options.wavelet_decomposition == 'dwt':
   verbose(3, "Doing DWT")
   WT = WaveletTransformationMapper(**kwargs)
 else:
   verbose(3, "Doing DWP")
   WT = WaveletPacketMapper(**kwargs)
 ds_orig = ds
 # Perform choosen wavelet decomposition
 ebdata_wt = WT(ebdata)
 ds = MaskedDataset(samples=ebdata_wt,
                    labels=ds_orig.labels,
                    chunks=ds_orig.chunks)
 # copy labels_map
 ds.labels_map = ds_orig.labels_map
