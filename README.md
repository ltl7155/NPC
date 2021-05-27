### Overview of NPC  (Under construction, please wait until May 30 2021):
![image](https://user-images.githubusercontent.com/26337247/118444619-f491d800-b71f-11eb-8947-a7deb62db2da.png)


### Derive LRP path.
We use the LRP method provided by https://github.com/moboehle/Pytorch-LRP.
In LRP_path, we run path_LRP.py to get a path according to the LRP method.
We set the parameters as follows to get a path of mnist dataset.

`cd LRP_path && python path_LRP.py --gpu 0 --arc convmnist --threshold 0.8 --dataset mnist --suffix mnist --data_train`

### RQ1 Mask Experiment.
We can run mask_critical_units.py to get inconsistency rate after masking neurons in CDP and NCDP.

`python mask_critical_units.py --paths_path LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --data_train --dataset mnist --arch convmnist`

### RQ2 Cluster.
We can run cluster_three_level_mask.py to cluster theses paths and derive the abstract path.

`python cluster_three_level_mask.py --paths_path LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --arc convmnist --b_cluster --dataset imagenet --gpu 1 --n_clusters 4 --threshold 0.8 --grids 5 --data_train`

### RQ3 SNPC,LSA,DSA,ANPC
In test_neuron_coverage.py, we inplemented the SNPC. And in calc_sadl, we can run new_run_torch.py to get the LSA,DSA,ANPC metric.

`python new_run_torch.py -nma -d mnist -attack manu --arch convmnist --dataset mnist --gpu 0`
Or
`python new_run_torch.py -lsa -dsa -d mnist -attack manu --arch convmnist --dataset mnist --gpu 0 --last_layer`
