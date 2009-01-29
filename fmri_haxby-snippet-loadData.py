def loadData(subj):
 """Load data for one subject and return
 dataset.

 :Parameter:
   subj: str
     ID of the subject who's data should be
     loaded.

 :Returns:
   NiftiDataset instance.
 """
 verbose(1, "Loading fMRI data from basepath"
            " %s" % datapath)

 # load labels and chunk information from
 # file layout: one line per voxel, two rows
 # (label(str), chunk(int) chunk corresponds
 # to the experimental run where the volume
 # was recorded
 attr = SampleAttributes(
         os.path.join(datapath, subj,
                      'labels.txt'),
                      literallabels=True)
 # load fMRI data from a NIfTI file, the data
 # was previously detrended using PyMVPA's
 # detrend() function.
 dataset = \
   NiftiDataset(
     samples=os.path.join(
                datapath, subj,
                'bold_detrend.nii.gz'),
     labels=attr.labels,
     # define fixed mapping of literal to
     # numerical labels
     labels_map={
       'rest': 0, 'face': 1, 'house': 2,
       'shoe': 3, 'cat': 4, 'scissors': 5,
       'scrambledpix': 6, 'bottle': 7,
       'chair': 8},
     chunks=attr.chunks,
     # load brain mask image to automatically
     # remove non-brain voxels
     mask=os.path.join(
        datapath, subj,
        'bold_example_brain_mask.nii.gz'))

 # go with just four classes to speed up
 # things -- still multiclass enough only
 # faces, houses, shoes, cats
 dataset = dataset['labels', [1,2,3,4]]

 # speed up even more by just using 6 instead
 # of 12 chunks, this will put two
 # successive chunks into one.
 coarsenChunks(dataset, 6)

 # done
 return dataset
