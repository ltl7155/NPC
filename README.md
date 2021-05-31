### Overview of NPC ( Under construction, please wait util May30 2021 ):
![image](https://user-images.githubusercontent.com/26337247/118444619-f491d800-b71f-11eb-8947-a7deb62db2da.png)


### Derive LRP path.
We use the LRP method provided by https://github.com/moboehle/Pytorch-LRP.
In LRP_path, we run path_LRP.py to get a path according to the LRP method.
We set the parameters as follows to get a path of mnist dataset.
`python trained_models/download.py --file_id 1ED01iWDC13vWdr2_217HiNyuhWiBpMaf --save_name LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl`


or to reproduce instead of download ours  

` python path_LRP.py --gpu 0 --arc convmnist --threshold 0.8 --dataset mnist --suffix mnist --data_train`


### RQ1 Mask Experiment.
We can run mask_critical_units.py to get inconsistency rate after masking neurons in CDP and NCDP.

`python mask_critical_units.py --paths_path ./data/LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --data_train --dataset mnist --arc convmnist`

### RQ2 Cluster.
We can run cluster_three_level_mask.py to cluster theses paths and derive the abstract path.

`python cluster_three_level_mask.py --paths_path ./data/LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --arc convmnist --b_cluster --dataset mnist --gpu 1 --n_clusters 4 --threshold 0.8 --grids 5 --data_train`

### RQ3 SNPC,LSA,DSA,ANPC
In calc_NPC.py, we inplemented the SNPC. And in calc_sadl, we can run new_run_torch.py to get the LSA,DSA,ANPC metric.
</br>
* ANPC</br>
`batch_size=64  python new_run_torch.py -nma ` </br>
* LSA(LSC) DSA(DSC) </br>
`batch_size=64  python new_run_torch.py -lsa -dsa  --last_layer`</br>
* SNPC </br>
`batch_size=64  python calc_NPC.py `</br>
