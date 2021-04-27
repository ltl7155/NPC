import numpy as np 

import sys 
p=sys.argv[1]


data=np.load(p,allow_pickle=True)

print (data)
