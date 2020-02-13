# VGMF
Our dataset and codes are released from the paper.
# Codes
Geographcial part of our model VGMF is located at /experiment/model/geo, where KMeansGMF.py is the variant GMF.  

Visual part is located at /experiment/model/images, where /train/MF_visual.py is the variant VMF and /train/KMeansVGMF.py is the final model VGMF.
# Dataset
It includes two parts,
.csv files are check-in records of users, which are divided into train, validate and test dataset, and .tfrecords files are visual features of images posted by users and locations, which are generated after \experiment\model\images\train\extract_feartures.py running.
# Additional information
Please give citation when using our dataset or code in your own work:  

@article{liuvgmf,
  title={VGMF: Visual contents and geographical influence enhanced point-of-interest recommendation in location-based social network},
  author={Liu, Bo and Meng, Qing and Zhang, Hengyuan and Xu, Kun and Cao, Jiuxin},
  journal={Transactions on Emerging Telecommunications Technologies},
  pages={e3889},
  publisher={Wiley Online Library}
}
