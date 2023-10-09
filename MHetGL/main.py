import numpy as np
import torch
import pickle
from pathlib import Path
import time
import logging

from utils import get_args, setup_seed, get_logger, EarlyStop, evaluate, get_default_args
from load_data import init_data
from model import MultiSphere

def main(args):

    # Prepare work
    setup_seed(args.seed)
    logger, fh, ch = get_logger(args)
    logger.info(args)
    device = 'cuda:{}'.format(args.gpu)

    # Load data
    logger.info('Dataset: {}. Setting: {}.'.format(args.dataset, args.setting))
    data = init_data(path=args.data_path, data_name=args.dataset, setting=args.setting, split_ratio_list=[0.6, 0.1, 0.3], device=device, curv_mode=args.curv_mode)

    if not args.no_train:

        # Load model
        logger.info('Model: {}.'.format(args.gnn))
        input_dim = data.x.shape[-1] # node feature dim
        model = MultiSphere(args=args, input_dim=input_dim, data=data, device=device, logger=logger)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler()
        earlyStopper = EarlyStop(logger, args)

        # Init center
        model.init_center(data)
        
        logger.info('Training...')
        for epoch in range(args.n_epochs):

            model.train()
            loss, _, dis_loss, dir_loss, clu_loss, multi_dis_loss = model(data, data.train_mask)

            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()
            logger.info('{}'.format(loss))
            # for name, param in model.gnn.named_parameters():
            #     logger.info('############ ' + name + ' ###############')
            #     logger.info('{} {}'.format(param, param.grad))
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            if args.center_mode == 'update': model.update_center()
            model.update_radius()

            """
            Validation
            """
            model.eval()
            with torch.no_grad():
                val_loss, val_ano_scores, _, _, _, _ = model(data, data.val_mask)
                threshold = model.radius.item() ** 2
                metrics_dict = evaluate(val_ano_scores, data.y[data.val_mask], False)
            logger.info("Epoch {} | Train Loss {:.3E}(={:.3E},{:.3E},{:.3E},{:.3E}) | Val Loss {:.3f} | Val AUC {:.3f} | Val AUPR {:.3f} | Radius {:.3f} |" \
                .format(epoch, loss.item(), dis_loss.item(), dir_loss.item(), clu_loss.item(), multi_dis_loss.item(), val_loss.item(), metrics_dict['auc'], metrics_dict['aupr'], model.radius.item()))
            
            # EarlyStopping
            if earlyStopper.test_metric(metrics_dict['auc'], model, epoch): break

    if not args.no_test:
        logger.info('Testing...')
        saved_dict = torch.load(args.model_path + 'model_{}_{}.pth.tar'.format(args.dataset, args.setting))
        logger.info('Load best model trained before epoch {}.'.format(saved_dict['epoch']))
        model = saved_dict['model']
        epoch = saved_dict['epoch']

        model.eval()
        with torch.no_grad():
            test_loss, test_ano_scores, _, _, _, _ = model(data, data.test_mask)
            threshold = model.radius.item() ** 2
            metrics_dict = evaluate(test_ano_scores, data.y[data.test_mask], args.case_study, 
                                    nx_graph=data.nx_graph, mask=data.test_mask.cpu().numpy(), seed=args.seed)
        logger.info("Epoch {} | AUC {:.4f} | AUPR {:.4f} |".format(epoch, metrics_dict['auc'], metrics_dict['aupr']))

        logger.removeHandler(fh)
        logger.removeHandler(ch)
        return metrics_dict['auc'], metrics_dict['aupr']
    else: 
        logger.removeHandler(fh)
        logger.removeHandler(ch)
        return 0., 0.

args = get_args()

if args.tune:
    print('---- * ---- Tuning Mode ---- * ----')
    time_list = range(5)

    dataset_list = ['cora-struct', 'cora-syn', 'citeseer-struct', 'citeseer-syn', 
                    'ml-struct', 'ml-syn', 'pubmed-struct', 'pubmed-syn', 'reddit']
    # dataset_list = ['cora-struct', 'ml-struct', 'reddit']

    lr_list = [0.1, 0.01, 0.001, 0.0001]
    hidden_dim_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    # hidden_dim_list = [16, 32, 64, 128, 256]
    num_layers_list = [1, 2, 3, 4]
    
    center_mode_list = ['init', 'update', 'train']
    radius_mode_list = ['cut', 'none']
    con_center_mode_list = ['train', 'detach', 'none']
    mul_center_mode_list = ['detach', 'train']
    num_estimation_layers_list = [2, 1]

    lamda_reg_cluster_list = [0., 0.0001, 0.001, 0.01]
    num_clusters_list = [2, 4, 6, 8, 10]
    lamda_cluster_list = [10, 1, 0.1, 0.01, 0.001]
    lamda_local_list = [10, 1, 0.1, 0.01, 0.001]

    assert args.log_path is not None and args.model_path is not None
    args.log_path = './log_tune/'
    args.model_path = './best_model_tune/'

    total_dict = dict()
    for data in dataset_list:
        assert args.dataset is not None
        args.dataset = data
        args = get_default_args(args) # get default args according to dataset

        # fix dataset
        hyper_dict_list = []
        auc_list = []
        aupr_list = []
        for i in time_list:
            once_auc_list = []
            once_aupr_list = []
            
            # for num_clusters in num_clusters_list:
            #     assert args.num_clusters is not None
            #     args.num_clusters = num_clusters
            #     for lamda_reg_cluster in lamda_reg_cluster_list:
            #         assert args.lamda_reg_cluster is not None
            #         args.lamda_reg_cluster = lamda_reg_cluster
            # for lamda_cluster in lamda_cluster_list:
            #     assert args.lamda_cluster is not None
            #     args.lamda_cluster = lamda_cluster
            #     for lamda_local in lamda_local_list:
            #         assert args.lamda_local is not None
            #         args.lamda_local = lamda_local

            # for center_mode in center_mode_list:
            #     assert args.center_mode is not None
            #     args.center_mode = center_mode
            #     for radius_mode in radius_mode_list:
            #         assert args.radius_mode is not None
            #         args.radius_mode = radius_mode
            #         for mul_center_mode in mul_center_mode_list:
            #             assert args.mul_center_mode is not None
            #             args.mul_center_mode = mul_center_mode
            # for lr in lr_list:
            #     assert args.lr is not None
            #     args.lr = lr
            #     for hidden_dim in hidden_dim_list:
            #         assert args.hidden_dim is not None
            #         args.hidden_dim = hidden_dim

            # for num_clusters in num_clusters_list:
            #     assert args.num_clusters is not None
            #     args.num_clusters = num_clusters  
            for center_mode in center_mode_list:
                assert args.center_mode is not None
                args.center_mode = center_mode     
            # for hidden_dim in hidden_dim_list:
            #     assert args.hidden_dim is not None
            #     args.hidden_dim = hidden_dim     
                if i == 0: 
                    # hyper_dict = {'lamda_cluster': lamda_cluster, 'lamda_local': lamda_local}
                    # hyper_dict = {'lamda_reg_cluster': lamda_reg_cluster, 'num_clusters': num_clusters}
                    # hyper_dict = {'center_mode': center_mode, 'radius_mode': radius_mode, 'mul_center_mode': mul_center_mode}
                    # hyper_dict = {'lr': lr, 'hidden_dim': hidden_dim}                
                    # hyper_dict = {}
                    # hyper_dict = {'num_clusters': num_clusters}
                    # hyper_dict = {'hidden_dim': hidden_dim}
                    hyper_dict = {'center_mode': center_mode}
                    hyper_dict_list.append(hyper_dict)

                auc, aupr = main(args)
                # auc, aupr = 0., 0.

                once_auc_list.append(auc)
                once_aupr_list.append(aupr)
            
            auc_list.append(np.array(once_auc_list))
            aupr_list.append(np.array(once_aupr_list))
        
        avg_auc_array = np.stack(auc_list).mean(axis=0)
        std_auc_array = np.stack(auc_list).std(axis=0)
        max_auc_array = np.stack(auc_list).max(axis=0)
        min_auc_array = np.stack(auc_list).min(axis=0)
        avg_aupr_array = np.stack(aupr_list).mean(axis=0)
        std_aupr_array = np.stack(aupr_list).std(axis=0)
        max_aupr_array = np.stack(aupr_list).max(axis=0)
        min_aupr_array = np.stack(aupr_list).min(axis=0)

        # Sort by auc, descending order
        sorted_inds = np.argsort(-avg_auc_array)
        sorted_avg_auc_array = avg_auc_array[sorted_inds]
        sorted_std_auc_array = std_auc_array[sorted_inds]
        sorted_max_auc_array = max_auc_array[sorted_inds]
        sorted_min_auc_array = min_auc_array[sorted_inds]
        sorted_avg_aupr_array = avg_aupr_array[sorted_inds]
        sorted_std_aupr_array = std_aupr_array[sorted_inds]
        sorted_max_aupr_array = max_aupr_array[sorted_inds]
        sorted_min_aupr_array = min_aupr_array[sorted_inds]

        sorted_hyper_dict_list = []
        for ind in sorted_inds: sorted_hyper_dict_list.append(hyper_dict_list[ind])

        total_dict[data] = {'hyper_dict_list': sorted_hyper_dict_list,
                                'avg_auc_array': sorted_avg_auc_array,
                                'std_auc_array': sorted_std_auc_array,
                                'max_auc_array': sorted_max_auc_array,
                                'min_auc_array': sorted_min_auc_array,
                                'avg_aupr_array': sorted_avg_aupr_array,
                                'std_aupr_array': sorted_std_aupr_array,
                                'max_aupr_array': sorted_max_aupr_array,
                                'min_aupr_array': sorted_min_aupr_array}
    
    Path(args.tune_dict_path).mkdir(parents=True, exist_ok=True)
    f_save = open(args.tune_dict_path + 'dict_file_{}.pkl'.format(str(time.time())), 'wb')
    pickle.dump(total_dict, f_save)
    f_save.close()
    print('--- * --- End of tuning --- * ---')

    for k in total_dict:
        print('\nDataset {} top hyper-combinations.'.format(k))
        print(total_dict[k]['hyper_dict_list'][:5])
        print(total_dict[k]['avg_auc_array'][:5])
        print(total_dict[k]['avg_aupr_array'][:5])

else:
    args = get_default_args(args) # get default args according to dataset 
    main(args)