 if options.zscore:
     verbose(2, "Z-scoring full dataset")
     zscore(ds, perchunk=False)

 # constant feature are not informative
 nf_orig = ds.nfeatures
 ds = removeInvariantFeatures(ds)
 verbose(2, "Removed invariant features. "
            "Got %d out of %d features"
            % (ds.nfeatures, nf_orig))
