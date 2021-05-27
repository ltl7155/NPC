import numpy as np
import time
import os
import sys
import torch
#sys.path.append("../LRP_path")

from multiprocessing import Pool
from tqdm import tqdm
# from keras.models import load_model, Model
from scipy.stats import gaussian_kde

from .utils_calc import *
from .torch_modelas_keras import TorchModel as Model 
from .get_a_single_path import getPath
from .get_all_similarities import sim_paths
import pickle

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def _get_saved_path(base_path, dataset, arch, dtype, layer_names, fileid=None):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """
    if fileid is not None :
        dtype= dtype+"_"+fileid

    joined_layer_names = "_".join(layer_names)
    joined_layer_names = joined_layer_names.replace("/", "_")
    joined_layer_names = "(" + joined_layer_names + ")"
    return (
        os.path.join(
            base_path,
            dataset + "_" + arch + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + arch + "_" + dtype + "_pred" + ".npy"),
#         os.path.join(base_path, dataset + "_" + arch + "_" + dtype + "_kdes" + ".npy"),
        os.path.join(base_path, dataset + "_" + arch + "_" + dtype + "_paths" + ".pkl"),
    )


def get_ats(
    model,
    dataset,
    name,
    layer_names,
    save_path=None,
    batch_size=128,
    is_classification=True,
    num_classes=10,
    num_proc=1,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        #inputs=model.input,
        #outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
        net= model,
        layer_names=layer_names
    )

    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
#         print(prefix + "Model serving", type(dataset), dataset.shape)
        print(prefix + "Model serving", type(dataset))
        pred = model.predict_classes(dataset, batch_size=batch_size, verbose=1)
        
        if len(layer_names) == 1:
            featuremap = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

            layer_outputs = [
                featuremap
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )
       
        ats = None

        for layer_name, layer_output in zip(layer_names, layer_outputs):
#             print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                print("calculating mean of feature maps.......")
                ret_list = [] 
                for i in  range(len(dataset)):
                    one_out = _aggr_output(layer_output[i]) 
                    ret_list.append(one_out)
                    
                layer_matrix = np.array(ret_list)
                
                #layer_matrix = np.array(
                #    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                #)
                print("\tfinished!")
            else:
                layer_matrix = np.array(layer_output)
            
#             print ("layer_matrix--->", layer_matrix.shape, layer_matrix.dtype)
            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None
# 
    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)
        p.close()
        
    return ats, pred


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.        
        train_ats (list): List of activation traces in training set (filtered)
        
    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), np.argmin(dist), train_ats[np.argmin(dist)])


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names, args):
    """Extract ats of train and target inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """
    train_name = "train"
    saved_train_path = _get_saved_path(args.save_path, args.dataset, args.arch, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_target_path = _get_saved_path(
        args.save_path, args.dataset, args.arch, target_name, layer_names, fileid=None
    )
    if os.path.exists(saved_target_path[0]):
        print(infog("warning:::::::::::::::::::::::Found saved test {} ATs, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
    else:
        target_ats, target_pred = get_ats(
            model,
            x_target,
            target_name,
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_target_path,
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))

    return train_ats, train_pred, target_ats, target_pred


def find_closest_at_paths(at, train_ats, paths, cla, clu, path_layer, samples_clusters, fakePath=False, rest=False):
    neurons = paths[cla][clu][path_layer]
    col_vectors = np.transpose(train_ats[samples_clusters[cla][clu]])
    removed_cols = [i for i in range(col_vectors.shape[0]) if i not in neurons]
    if fakePath:
        removed_cols = []
    if rest:
        removed_cols = neurons
        print("removing critical neurons.......................")
    refined_ats = np.transpose(train_ats[samples_clusters[cla][clu]])
    refined_ats = np.delete(refined_ats, removed_cols, axis=0)
    refined_ats = np.transpose(refined_ats)
    refined_at = np.delete(at, removed_cols, axis=0)
    dist, index, _ = find_closest_at(refined_at, refined_ats)
    dot = train_ats[samples_clusters[cla][clu]][index]
    return dist, dot


def fetch_dsa(model, x_train, x_target, target_name, layer_names, args, paths, path_layer, path=False):
    """Distance-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        dsa (list): List of dsa for each target input.
    """

    assert args.is_classification == True

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa = []

    print(prefix + "Fetching DSA" + "__layer" + str(path_layer))
    oods = []
    a_dists = []
    b_dists = []
    for i, at in enumerate(tqdm(target_ats)):
        label = target_pred[i]
        if path:
            dists = []
            a_dist, a_dot = find_closest_at_paths(at, train_ats, paths, label, path_layer, class_matrix)
            for l in range(args.num_classes):
                if l != label:
                    dist, _ = find_closest_at_paths(a_dot, train_ats, paths, l, path_layer, class_matrix)
                    dists.append(float(dist))
            b_dist = min(dists) 
        else:
            a_dist, _, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
            b_dist, _, _ = find_closest_at(
                a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
            )
        if a_dist / b_dist < 1:
#             print("OOD index:", i)
            oods.append(i)
    
        a_dists.append(a_dist)
        b_dists.append(b_dist)
        dsa.append(a_dist / b_dist)
#     print("OOD index:", oods)
#         print("dsa:::::::::", dsa)
    return dsa, oods, a_dists, b_dists 


def _get_kdes(train_ats, train_pred, class_matrix, args, paths, path_layer, path=False):
    """Kernel density estimation

    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """
    print("calculating kdes of training data........")
    removed_cols = [[] for _ in range(args.num_classes)]
    if args.is_classification:
        if not path:
            for label in range(args.num_classes):
                col_vectors = np.transpose(train_ats[class_matrix[label]])
                for i in range(col_vectors.shape[0]):
                    if (
                        np.var(col_vectors[i]) < args.var_threshold
                        and i not in removed_cols
                    ):
                        removed_cols[label].append(i)
        else:
            print("bug"*10)

        kdes = {}
        for label in tqdm(range(args.num_classes), desc="kde"):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols[label], axis=0)
#             print(refined_ats.shape)
#             print(removed_cols[label])
            if refined_ats.shape[0] == 0:
                print(
                    warn("ats were removed by threshold {}".format(args.var_threshold))
                )
                break
            kdes[label] = gaussian_kde(refined_ats)

    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < args.var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print(warn("ats were removed by threshold {}".format(args.var_threshold)))
        kdes = [gaussian_kde(refined_ats)]

#     print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols


def _get_lsa(kde, at, removed_c):
    refined_at = np.delete(at, removed_c, axis=0)
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))


def fetch_lsa(model, x_train, x_target, target_name, layer_names, args, paths, path_layer, path=False):
    """Likelihood-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or[] adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        lsa (list): List of lsa for each target input.
    """

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    if args.is_classification:
        for i, label in enumerate(train_pred):
            if label not in class_matrix:
                class_matrix[label] = []
            class_matrix[label].append(i)
#     print(class_matrix)
#     print (type(train_ats),train_ats.shape)
    
    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args, paths=paths, path_layer=path_layer, path=path)
#     print ("------"*8)
#     print (train_ats,kdes, removed_cols )
#     exit()
    lsa = []
    print(prefix + "Fetching LSA")
    if args.is_classification:
        for i, at in enumerate(tqdm(target_ats)):
            label = target_pred[i]
            kde = kdes[label]
            lsa.append(_get_lsa(kde, at, removed_cols[label]))
    else:
        kde = kdes[0]
        for at in tqdm(target_ats):
            lsa.append(_get_lsa(kde, at, removed_cols[label]))

    return lsa



def fetch_newMetric(model, ori_model, x_train, x_target, y_test, sample_threshold, target_name, layer_names, 
                    args, cluster_paths, path_layer, samples_clusters, fakePath=False, rest=False):   
    
    saved_target_path = _get_saved_path(
        args.save_path, args.dataset, args.arch, target_name, layer_names, fileid=None,
    )
    if os.path.exists(saved_target_path[2]):
        print(infog("warning:::::::::::::::::::::::Found saved test {} paths, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        with open(saved_target_path[2], 'rb') as handle:
            path_x = pickle.load(handle)
        t = 0

    else:
#         print("x_targetx_target", x_target.size())
#         print("x_targetx_target", y_test.size())
        if x_target.size(0) != y_test.size(0):
            y_test = torch.randint(0, 1, (x_target.size(0), ))
            print(
                    warn("size of y doesn't match size of x.")
                )
            
        test_loader1=torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x_target, y_test),
                    batch_size=25, shuffle=False)
        path_x = {}
        end = start = 0
        print("getting path......")
        s = time.time()
        for step, (x, y) in enumerate(test_loader1):           
            X = x.cuda()
            this_batch_size = X.shape[0]
    #         print("getting individual-paths......")
            the_path = getPath(X, ori_model, sample_threshold, arc=args.arch)
            for k in the_path.keys():
                path_x[start+k] = the_path[k]
            start += this_batch_size
        e = time.time()
        t = e-s
        print("\tDone!")
        with open(saved_target_path[2], 'wb') as handle:
            pickle.dump(path_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(infog(target_name + " Paths is saved at " + saved_target_path[2]))
    

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
        )

    def pick_train_ats(train_ats, samples):
        picked = []
        for i in samples:
            picked.append(train_ats[i])
        return picked
        
    dis_all = []
    clu_all = []
    cla_all = []
#     print("getting distance......")
    for i in tqdm(range(len(path_x))):
        p = path_x[i]
        cla = target_pred[i]
        clu, best_sim = find_closed_cluster(p, cluster_paths, cla)
        dis, _ = getDis(target_ats[i], train_ats, cluster_paths, cla, clu, path_layer, samples_clusters, fakePath=fakePath, rest=rest)
        dis_all.append(dis)   
        clu_all.append(clu)
        cla_all.append(cla)
    return dis_all, clu_all, cla_all, t


def getDis(picked_target_at, train_ats, cluster_paths, cla, clu, path_layer, samples_clusters, fakePath=False, rest=False):
    return find_closest_at_paths(picked_target_at, train_ats, cluster_paths, cla, clu, path_layer, 
                                 samples_clusters, fakePath=fakePath, rest=rest)


def find_closed_cluster(path, cluster_paths, cla):
    best_sim = 0
    position = 0
    for i, cp in enumerate(cluster_paths[cla]):
        sim, _ = sim_paths(path, cp)
        if sim > best_sim:
            position = i
            best_sim = sim
    return position, best_sim
            

def get_sc(lower, upper, k, sa):
    """Surprise Coverage
    mnist:
    lsa:2000   
    cifar
    lsa:100
    
    mnist
    dsa:2.0 
    cifar
    dsa:2.0 
    
    
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """
    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100, buckets
