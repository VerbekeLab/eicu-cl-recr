import numpy as np

# The binning is based on E. Rocheteau's code
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=True):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None

def cl_recruitment(args, dataset):
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = dataset

    recr_stats_local = {}
    
    for key, loader in train_data_local_dict.items():
        data_length = len(loader.dataset)
        output_dist = get_output_dist(loader)
        recr_stats_local[key] = (output_dist, data_length)

    global_output_dist, total_inst = get_global_dist(recr_stats_local)

    recr_stats_local = local_dist_divergence(global_output_dist, total_inst, recr_stats_local)
    
    # from this point on the recr_stats_local is of the form {key: (dist, inst, divergence)} 
    client_rp_order = representativeness(args, recr_stats_local)

    client_list = get_recr_clients(args, client_rp_order)

    print(f'The recruited clients ({len(client_list)}) for the federation are {client_list}')
    
    return client_list


def get_output_dist(loader):
    
    out = np.ndarray(0)

    for i in range(len(loader.dataset)):
        out = np.append(out, loader.dataset[i][1])

    out_bins = sum([get_bin_custom(x, CustomBins.nbins) for x in out])

    return out_bins

def update_dataset_args(args, client_list, dataset):
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = dataset

    args.client_num_in_total = len(client_list)
    args.client_num_per_round = round(len(client_list)*args.cl_num) 

    local_num_dict = {}
    train_dict = {}
    test_dict = {}

    for key, hospitalid in enumerate(client_list):
        loader = train_data_local_dict[hospitalid]

        local_num_dict[key] = len(loader.dataset)
        train_dict[key] = train_data_local_dict[hospitalid]
        test_dict[key] = test_data_local_dict[hospitalid]
        
    train_data_local_num_dict = local_num_dict
    train_data_local_dict = train_dict
    test_data_local_dict = test_dict

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return args, dataset

def get_global_dist(stats_dict):
    out = np.ndarray(0)
    total_instances = 0 

    for _ , (local_dist, local_size) in stats_dict.items():
        out = np.append(out, local_dist)
        total_instances += local_size
    
    out = np.reshape(out, (-1, CustomBins.nbins))
    global_dist = sum(out)

    return global_dist, total_instances

def local_dist_divergence(global_output_dist, total_inst, recr_stats_local):
    global_output_dist_norm = global_output_dist/total_inst

    for key, (local_dist, local_inst) in recr_stats_local.items():
        local_dist_norm = local_dist/local_inst
        local_divergence = abs(sum((global_output_dist_norm - local_dist_norm)))
        recr_stats_local[key] = (local_dist, local_inst, local_divergence)
    
    return recr_stats_local

def representativeness(args, recr_stats_local):
    cl_rep = {}

    for key, (_, inst, divergence) in recr_stats_local.items():
        div_w = args.div_weight
        inst_w = args.inst_weight
        c_rp = (div_w*divergence)+(inst_w*(inst**-0.5))
        cl_rep[key] = c_rp

    sorted_cl = dict(sorted(cl_rep.items(), key=lambda item: item[1])) 

    return sorted_cl

def get_recr_clients(args, sorted_cl):
    tot_rep = 0
    threshold_rep = 0
    clients = []
    
    for key, value in sorted_cl.items():
        tot_rep += value
    
    for key, value in sorted_cl.items():
        threshold_rep += value
        clients.append(key)
        if threshold_rep >= tot_rep*args.repr_perc:
            break

    return clients