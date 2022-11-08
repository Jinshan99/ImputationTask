import sys
import numpy as np
import json 
from sklearn.impute import KNNImputer

mask_m=np.load(sys.argv[2],allow_pickle=True)#"400x24-mask10-1.npy"
data_m = np.load(sys.argv[1],allow_pickle=True)#"w1d0m.npy"
# mask_m=np.load("400x24-mask10-1.npy")#sys.argv[1]
# data_m = np.load("w1d0m.npy")#sys.argv[0]

masked_data_m = data_m.copy()
masked_data_m[ np.where(mask_m == 0) ] = np.nan

imputer=KNNImputer(n_neighbors=5)
imputer.fit_transform(masked_data_m)
imputer_m=imputer.fit_transform(masked_data_m)



pos_m=np.argwhere(mask_m==0)
true_value=[]
pred_value=[]
for i in range(len(pos_m)):
    true_value.append(data_m[pos_m[i][0]][pos_m[i][1]])
    pred_value.append(imputer_m[pos_m[i][0]][pos_m[i][1]])


create_dict={}
create_dict['position']=np.argwhere(mask_m==0).tolist()
create_dict['true_value']=true_value
create_dict['pred_value']=pred_value

with open("nnimpute.json", "w") as outfile:
    json.dump(create_dict, outfile)




