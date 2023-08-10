# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import copy
import json
import logging
from logging import getLogger

import torch
import pickle
import time
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

import numpy as np


def plot_svd_fig(model, dataset, config, svsname=None):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    embedding_matrix = model.item_embedding.weight[1:].cpu().detach().numpy()
    svd = TruncatedSVD(n_components=2)
    svd.fit(embedding_matrix)
    comp_tr = np.transpose(svd.components_)
    proj = np.dot(embedding_matrix, comp_tr)

    cnt = {}
    for i in dataset['item_id']:
        if i.item() in cnt:
            cnt[i.item()] += 1
        else:
            cnt[i.item()] = 1

    freq = np.zeros(embedding_matrix.shape[0])
    for i in cnt:
        freq[i - 1] = cnt[i]

    # freq /= freq.max()

    sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.scatter(proj[:, 0], proj[:, 1], s=1, c=freq, cmap='viridis_r')
    plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axis('square')
    plt.show()
    log_dir = 'saved_fig'
    plt.savefig(log_dir + '/' + config['model'] + '-' + config['dataset'] + '.pdf', format='pdf', transparent=False,
                bbox_inches='tight')

    from scipy.linalg import svdvals
    svs = svdvals(embedding_matrix)
    svs = svs.cumsum()
    svs /= svs.max()
    if svsname is not None:
        np.save('./saved_representation_np/' + svsname + '.npy', svs)

    sns.set(style='darkgrid')
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
    plt.figure(figsize=(6, 4.5))
    plt.plot(svs)
    plt.show()
    plt.savefig(log_dir + '/svs.pdf', format='pdf', transparent=False, bbox_inches='tight')


def plot_heatmap(data, heatmap=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.linalg import svdvals

    data = data.cpu().numpy()
    for i in range(data.shape[0]):
        d = data[i]
        if heatmap:
            plt.cla()
            plt.clf()
            hm = sns.heatmap(d)
            plt.savefig('./saved_heatmap/heatmap_{}.png'.format(i))

        svs = svdvals(d)
        svs /= svs.max()
        plt.cla()
        plt.clf()
        sns.set(style='darkgrid')
        sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
        plt.figure(figsize=(6, 4.5))
        plt.plot(svs)
        plt.savefig('./saved_heatmap/seq_svs_{}.png'.format(i))


def count_item_inter_num(inter_feat):
    item_inter_num = dict()
    user_id_list = inter_feat['user_id'].tolist()
    item_id_list = inter_feat['item_id'].tolist()
    for i in range(len(user_id_list) - 1):
        if user_id_list[i] == user_id_list[i + 1]:
            item_id1, item_id2 = item_id_list[i], item_id_list[i + 1]
            if item_id1 in item_inter_num:
                item_inter_num[item_id1] += 1
            else:
                item_inter_num[item_id1] = 1
            if item_id2 in item_inter_num:
                item_inter_num[item_id2] += 1
            else:
                item_inter_num[item_id2] = 1
    return item_inter_num


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    # item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    # test_item_inter_num = np.array(list(map(item_inter_num.get, test_data.dataset['item_id'].tolist())))
    # new_interaction = test_data.dataset[test_item_inter_num > test_item_inter_num.mean()]
    # test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])


    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    save_selected_results_to_txt(config,
                                 model_name+'_'+dataset_name,
                                 best_valid_result,
                                 test_result,
                                 dataset_info=dataset_info)

    # plot attention heatmap
    # heatmap_data = copy.deepcopy(test_data)
    # heatmap_interaction = train_data.dataset[train_data.dataset['item_length'] == train_data.dataset['item_length'].max()]
    # heatmap_data.dataset = train_data.dataset.copy(heatmap_interaction)
    # attention = trainer.get_attention(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(attention)
    # seq_representation = trainer.get_seq_representation(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(seq_representation, heatmap=False)

    # plot_svd_fig(model, dataset, config)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def watch_model(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, model_path=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    # item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    # test_item_inter_num = np.array(list(map(item_inter_num.get, test_data.dataset['item_id'].tolist())))
    # new_interaction = test_data.dataset[test_item_inter_num > test_item_inter_num.mean()]
    # test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    checkpoint_file = model_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    logger.info(message_output)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    save_selected_results_to_txt(config,
                                 model_name+'_'+dataset_name,
                                 None,
                                 test_result,
                                 dataset_info=dataset_info)

    # plot attention heatmap
    # heatmap_data = copy.deepcopy(test_data)
    # heatmap_interaction = train_data.dataset[train_data.dataset['item_length'] == train_data.dataset['item_length'].max()]
    # heatmap_data.dataset = train_data.dataset.copy(heatmap_interaction)
    # attention = trainer.get_attention(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(attention)
    # seq_representation = trainer.get_seq_representation(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(seq_representation, heatmap=False)

    plot_svd_fig(model, dataset, config, svsname=model_name+'_'+dataset_name)


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def save_to_txt(file_name, best_valid_result, test_result):
    t = time.strftime('_%Y-%m-%d_%H-%M-%S', time.localtime())
    f = open('./saved_result/'+file_name+t+'.txt', 'w')
    f.write(json.dumps(best_valid_result))
    f.write('\n')
    f.write(json.dumps(test_result))
    f.close()


def save_selected_results_to_txt(config, file_name, best_valid_result, test_result, dataset_info=None):
    t = time.strftime('_%Y-%m-%d_%H-%M-%S', time.localtime())
    f = open('./saved_result/'+file_name+t+'.txt', 'w')
    selected_results_name = ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
    if best_valid_result is not None:
        f.write('Best valid result')
        f.write('\n')
        for name in selected_results_name:
            f.write( str(best_valid_result[name]))
            f.write('\n')
        f.write('\n')
    f.write('Test result')
    f.write('\n')
    for name in selected_results_name:
        f.write(str(test_result[name]))
        f.write('\n')
    f.write('\n')
    if dataset_info is not None:
        f.write('Average action nums of users:{:.4f}\n'.format(dataset_info[0]))
        f.write('Average action nums of items:{:.4f}\n'.format(dataset_info[1]))
        f.write('Sparsity:{:.4f}\n'.format(dataset_info[2]))
    # f.write('loss type: {}'.format(config['loss_type']))
    # f.write('\n')
    # f.write('temperature: {:.4f}'.format(config['temperature']))
    f.close()


def watch_fmlp_filters(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, model_path=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    # item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    # test_item_inter_num = np.array(list(map(item_inter_num.get, test_data.dataset['item_id'].tolist())))
    # new_interaction = test_data.dataset[test_item_inter_num > test_item_inter_num.mean()]
    # test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    checkpoint_file = model_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    logger.info(message_output)

    filters = model.get_learnable_filters()
    np.save('./saved_fmlp_filters/' + model_name + '_' + dataset_name + '.npy', filters)


def watch_testmodel_filters(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, model_path=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    new_interaction = test_data.dataset[test_data.dataset['item_length'] == test_data.dataset['item_length'].max()]
    test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    checkpoint_file = model_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    logger.info(message_output)

    iter_data = (tqdm(test_data, total=len(test_data), ncols=100, desc=set_color(f"Evaluate   ", 'pink')))
    for batch_idx, batched_data in enumerate(iter_data):
        if batch_idx > 0:
            break
        filters = model.get_filters(batched_data[0].interaction)
    filters = np.concatenate([np.expand_dims(filters[0], 1), np.expand_dims(filters[1], 1)], axis=1)
    np.save('./saved_testmodel_filters/' + model_name + '_' + dataset_name + '.npy', filters)


def save_item_embedding_eigval_cumsum(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, model_path=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    new_interaction = test_data.dataset[test_data.dataset['item_length'] == test_data.dataset['item_length'].max()]
    test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    checkpoint_file = model_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    logger.info(message_output)

    embedding_matrix = model.item_embedding.weight[1:].cpu().detach().numpy()
    from scipy.linalg import svdvals
    val = svdvals(embedding_matrix)
    val = val.cumsum()
    val /= val.max()
    np.save('./saved_representation_np/' + model_name + '_' + dataset_name + '.npy', val)


def test_model(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, model_path=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    model_name = copy.deepcopy(model)
    dataset_name = copy.deepcopy(dataset)
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    # get item_inter_num
    item_inter_num = count_item_inter_num(dataset.inter_feat)
    logger.info(dataset)
    dataset_info = [dataset.inter_num / dataset.user_num,
                    dataset.inter_num / dataset.item_num,
                    1.0 - 1.0 * dataset.inter_num / (dataset.item_num * dataset.user_num)]

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # filter high frequency item in test_data
    new_interaction = test_data.dataset[test_data.dataset['item_length'] == test_data.dataset['item_length'].max()]
    test_data.dataset = test_data.dataset.copy(new_interaction)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    checkpoint_file = model_path
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    logger.info(message_output)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model evaluation
    # test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    output_mean = trainer.get_mean_output(test_data, load_best_model=False, show_progress=config['show_progress'])

    output_mean = output_mean.cpu().numpy()
    np.save('saved_mean_output/inverse_office_freqmixer_before.npy', output_mean)

    # save_selected_results_to_txt(config,
    #                              model_name+'_'+dataset_name,
    #                              None,
    #                              test_result,
    #                              dataset_info=dataset_info)

    # plot attention heatmap
    # heatmap_data = copy.deepcopy(test_data)
    # heatmap_interaction = train_data.dataset[train_data.dataset['item_length'] == train_data.dataset['item_length'].max()]
    # heatmap_data.dataset = train_data.dataset.copy(heatmap_interaction)
    # attention = trainer.get_attention(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(attention)
    # seq_representation = trainer.get_seq_representation(heatmap_data, load_best_model=saved, show_progress=config['show_progress'])
    # plot_heatmap(seq_representation, heatmap=False)

    # plot_svd_fig(model, dataset, config)




