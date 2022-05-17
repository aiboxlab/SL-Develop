
FacialActionLibras/models/openface$ sudo chmod +x install.sh

FacialActionLibras/models/openface$ ./install.sh

FacialActionLibras/models/openface$ build/bin/FaceLandmarkImg -f ../../data/examples/me.jpg



./FacialActionLibras/models/openface/build/bin/FeatureExtraction2Way -f FacialActionLibras/data/raw/UCF-101-Analysis-Selected/v_ApplyEyeMakeup_g01_c02.avi  -out_dir FacialActionLibras/data/processed/examples/ -2Dfp