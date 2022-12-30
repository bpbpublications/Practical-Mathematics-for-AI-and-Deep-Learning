import numpy as np

arr1 = np.array([[7,8,9],[1,2,3]])
print("len", len(arr1))
print("size", np.size(arr1, axis=0))

import pandas as pd
import numpy as np

seq = np.random.random(10)
print("ewma \n", pd.DataFrame(seq).ewm(alpha=0.1,adjust=False))
print("ewma mean\n", pd.DataFrame(seq).ewm(alpha=0.1,adjust=False).mean()/2)

#Also, we can implement it as
S = seq[0]
alpha = 0.1
for i in range(len(seq)):
    S = alpha*seq[i]+(1-alpha)*S
    output = S/2
    print(output, S)
