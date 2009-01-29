def doSensitivityAnalysis(ds, clfs,
                          sensanas,
                          splitter,
                          sa_args=""):
 """Generic function to perform sensitivity
 analysis (along classification)

 :Parameters:
   ds : Dataset
     Dataset to perform analysis on
   clfs : list of Classfier
     Classifiers to take sensitivities
     (default parameters) of
   sensanas : list of DatasetMeasure
     Additional measures to be computed
   splitter : Splitter
     Splitter to be used for cross-validation
   sa_args : basestring
     Additional optional arguments to provide
     to getSensitivityAnalyzer
 """
 # to absorb all sensitivities
 senses = []

 # run classifiers in cross-validation
 for label, clf in clfs.iteritems():
   sclf = SplitClassifier(clf, splitter,
       enable_states=['confusion',
                      'training_confusion'])

   verbose(1, 'Doing cross-validation with '
              + label)
   # Compute sensitivity, which in turn
   # trains the sclf
   sensitivities = \
     sclf.getSensitivityAnalyzer(
       # do not combine sensitivities across
       # splits, nor across classes
       combiner=None,
       slave_combiner=None)(ds)

   verbose(1, 'Accumulated confusion matrix '
              'for out-of-sample tests:\n' +
              str(sclf.confusion))

   # and store
   senses.append(
    (label + ' (%.1f%% corr.) weights' \
            % sclf.confusion.stats['ACC%'],
     sensitivities, sclf.confusion,
     sclf.training_confusion))

 verbose(1, 'Computing additional '
            'sensitivities')

 # wrap everything into
 # SplitFeaturewiseMeasure to get sense of
 # variance across our artificial splits
 # compute additional sensitivities
 for k, v in sensanas.iteritems():
   verbose(2, 'Computing: ' + k)
   sa = SplitFeaturewiseMeasure(
           v, splitter,
           enable_states=['maps'])
   # compute sensitivities
   sa(ds)
   # and grab them for all splits
   senses.append((k, sa.maps, None, None))

 return senses
