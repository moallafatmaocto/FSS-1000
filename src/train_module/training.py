import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from src.data_preprocessing.batch_generator import get_training_batch
from src.display_data import decode_segmap
from src.network import RelationNetwork, CNNEncoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main(finetune: bool, feature_model: str, relation_model: str, learning_rate: int,
         start_episode: int, nbr_episode: int, class_num: int, sample_num_per_class: int,
         batch_num_per_class: int, train_result_path: str, model_save_path: str,
         result_save_freq: int, display_query: int, model_save_freq: int, gpu: int, load_imagenet: bool):
    # Step 1: init neural networks
    print("init neural networks")

    feature_encoder, feature_encoder_optim, feature_encoder_scheduler, relation_network, relation_network_optim, relation_network_scheduler = init_encoder_and_network_for_training(
        feature_model, finetune, gpu, learning_rate, load_imagenet, nbr_episode, relation_model)

    print("Training...")
    last_accuracy = 0.0
    for episode in range(start_episode, nbr_episode):
        init_saving_folders(model_save_path, train_result_path)
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        samples, sample_labels, batches, batch_labels, chosen_classes = get_training_batch(class_num,
                                                                                           sample_num_per_class,
                                                                                           batch_num_per_class)
        ft_list, relation_pairs = get_relation_pairs_and_encoded_features(batch_num_per_class, batches, class_num,
                                                                          feature_encoder, gpu, sample_num_per_class,
                                                                          samples)
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)
        loss = evaluate_loss(batch_labels, gpu, output)
        # training
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()
        # training result visualization
        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss.cpu().data.numpy())
        training_result_visualization(batch_labels, batch_num_per_class, batches, chosen_classes, class_num,
                                      display_query, episode, output, result_save_freq, sample_labels,
                                      sample_num_per_class, samples, train_result_path)
        # save models
        save_trained_network_model_and_feature_encoder(class_num, episode, feature_encoder, model_save_freq,
                                                       model_save_path, relation_network, sample_num_per_class)


def init_saving_folders(model_save_path, train_result_path):
    if not os.path.exists(train_result_path):
        os.makedirs(train_result_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


def evaluate_loss(batch_labels, gpu, output):
    mse = nn.MSELoss().cuda(gpu)
    loss = mse(output, Variable(batch_labels).cuda(gpu))
    return loss


def training_result_visualization(batch_labels, batch_num_per_class, batches, chosen_classes, class_num, display_query,
                                  episode, output, result_save_freq, sample_labels, sample_num_per_class, samples,
                                  train_result_path):
    if (episode + 1) % result_save_freq == 0:
        support_output = np.zeros((224 * 2, 224 * sample_num_per_class, 3), dtype=np.uint8)
        query_output = np.zeros((224 * 3, 224 * display_query, 3), dtype=np.uint8)
        chosen_query = random.sample(list(range(0, batch_num_per_class)), display_query)

        for i in range(class_num):
            for j in range(sample_num_per_class):
                supp_img = (np.transpose(samples.numpy()[j], (1, 2, 0)) * 255).astype(np.uint8)[:, :, :3][:, :,
                           ::-1]
                support_output[0:224, j * 224:(j + 1) * 224, :] = supp_img
                supp_label = sample_labels.numpy()[j][0]
                supp_label[supp_label != 0] = chosen_classes[i]
                supp_label = decode_segmap(supp_label)
                support_output[224:224 * 2, j * 224:(j + 1) * 224, :] = supp_label

            for cnt, x in enumerate(chosen_query):
                query_img = (np.transpose(batches.numpy()[x], (1, 2, 0)) * 255).astype(np.uint8)[:, :, :3][:, :,
                            ::-1]
                query_output[0:224, cnt * 224:(cnt + 1) * 224, :] = query_img
                query_label = batch_labels.numpy()[x][0]  # only apply to one-way setting
                query_label[query_label != 0] = chosen_classes[i]
                query_label = decode_segmap(query_label)
                query_output[224:224 * 2, cnt * 224:(cnt + 1) * 224, :] = query_label

                query_pred = output.detach().cpu().numpy()[x][0]
                query_pred = (query_pred * 255).astype(np.uint8)
                result = np.zeros((224, 224, 3), dtype=np.uint8)
                result[:, :, 0] = query_pred
                result[:, :, 1] = query_pred
                result[:, :, 2] = query_pred
                query_output[224 * 2:224 * 3, cnt * 224:(cnt + 1) * 224, :] = result
        extra = query_output.copy()
        for i in range(class_num):
            for cnt, x in enumerate(chosen_query):
                extra_label = batch_labels.numpy()[x][0]
                extra_label[extra_label != 0] = 255
                result1 = np.zeros((224, 224, 3), dtype=np.uint8)
                result1[:, :, 0] = extra_label
                result1[:, :, 1] = extra_label
                result1[:, :, 2] = extra_label
                extra[224 * 2:224 * 3, cnt * 224:(cnt + 1) * 224, :] = result1
        cv2.imwrite('%s/%s_query.png' % (train_result_path, episode), query_output)
        cv2.imwrite('%s/%s_show.png' % (train_result_path, episode), extra)
        cv2.imwrite('%s/%s_support.png' % (train_result_path, episode), support_output)


def save_trained_network_model_and_feature_encoder(class_num, episode, feature_encoder, model_save_freq,
                                                   model_save_path, relation_network, sample_num_per_class):
    if (episode + 1) % model_save_freq == 0:
        torch.save(feature_encoder.state_dict(), str(
            "./%s/feature_encoder_" % model_save_path + str(episode) + '_' + str(class_num) + "_way_" + str(
                sample_num_per_class) + "shot.pkl"))
        torch.save(relation_network.state_dict(), str(
            "./%s/relation_network_" % model_save_path + str(episode) + '_' + str(class_num) + "_way_" + str(
                sample_num_per_class) + "shot.pkl"))
        print("save networks for episode:", episode)


def get_relation_pairs_and_encoded_features(batch_num_per_class, batches, class_num, feature_encoder, gpu,
                                            sample_num_per_class, samples):
    # calculate features
    sample_features, _ = feature_encoder(Variable(samples).cuda(gpu))
    sample_features = sample_features.view(class_num, sample_num_per_class, 512, 7, 7)
    sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
    batch_features, ft_list = feature_encoder(Variable(batches).cuda(gpu))
    # calculate relations
    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_num_per_class * class_num, 1, 1, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
    return ft_list, relation_pairs


def init_encoder_and_network_for_training(feature_model, finetune, gpu, learning_rate, load_imagenet, nbr_episode,
                                          relation_model):
    feature_encoder = CNNEncoder(pretrained=load_imagenet)
    relation_network = RelationNetwork()
    relation_network.apply(weights_init)
    feature_encoder.cuda(gpu)
    relation_network.cuda(gpu)
    # fine-tuning
    if (finetune):
        if os.path.exists(feature_model):
            feature_encoder.load_state_dict(torch.load(feature_model))
            print("load feature encoder success")
        else:
            print('Can not load feature encoder: %s' % feature_model)
            print('starting from scratch')
        if os.path.exists(relation_model):
            relation_network.load_state_dict(torch.load(relation_model))
            print("load relation network success")
        else:
            print('Can not load relation network: %s' % relation_model)
            print('starting from scratch')
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=learning_rate)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=nbr_episode // 10, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=learning_rate)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=nbr_episode // 10, gamma=0.5)
    return feature_encoder, feature_encoder_optim, feature_encoder_scheduler, relation_network, relation_network_optim, relation_network_scheduler
