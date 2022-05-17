# import os
# os.system("./bin/FeatureExtraction2Way -f ../../../data/examples/tamires/RostoIntensidade-01Primeira-Acalmar.avi -2Dfp")
# os.system('ls')

import subprocess
return_code = subprocess.run(["./bin/FeatureExtraction2Way","-f RostoIntensidade-01Primeira-Acalmar.avi"])
