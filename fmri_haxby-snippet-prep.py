 # run common preprocessing
 zscore(ds, perchunk=True,
        targetdtype='float32')

 # give status report
 verbose(1, 'Dataset after preprocessing:\n'
            '%s' % ds.summary())
