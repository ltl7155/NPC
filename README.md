### Overview of NPC :
![image](https://user-images.githubusercontent.com/26337247/118444619-f491d800-b71f-11eb-8947-a7deb62db2da.png)


### Derive LRP path.
first, please download some pretrained weight, data and seed case from drive.google.com by our scripts. </br>
or if your do want to reproduce instead of downloading, please refer to the [full_version.md](./fullversion.md)

* download MNIST's model's weight (4.7 Mb), MNIST dataset (53 Mb), one seed-case (36Mb)</br>

`cd  data &&  python download.py --download_mnist_example  && cd .. `


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
