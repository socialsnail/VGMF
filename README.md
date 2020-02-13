# VGMF
Our dataset and codes are released from the paper.
# Codes
Geographcial part of our model VGMF is located at /experiment/model/geo, where KMeansGMF.py is the variant GMF.
Visual part is located at /experiment/model/images, where /train/MF_visual.py is the variant VMF and /train/KMeansVGMF.py is the final model VGMF.
#Dataset
It includes two parts,
.csv files are check-in records of users, which are divided into train, validate and test dataset.
.tfrecords files are visual features of images posted by users and locations, which are generated after \experiment\model\images\train\extract_feartures.py running.
#Additional information
Please give citation when using our dataset.
