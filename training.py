import argparse
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from data_preprocessing.batch_generator import get_training_batch
from display_data import decode_segmap
from network import RelationNetwork, CNNEncoder

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=1)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=5)
parser.add_argument("-e", "--episode", type=int, default=800000)
parser.add_argument("-start", "--start_episode", type=int, default=0)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-d", "--display_query_num", type=int, default=5)
parser.add_argument("-ex", "--exclude_class", type=int, default=6)
parser.add_argument("-modelf", "--feature_encoder_model", type=str, default='')
parser.add_argument("-modelr", "--relation_network_model", type=str, default='')
parser.add_argument("-lo", "--loadImagenet", type=bool, default=False)
parser.add_argument("-fi", "--finetune", type=bool, default=True)
parser.add_argument("-rf", "--TrainResultPath", type=str, default='result_newvgg_1shot')
parser.add_argument("-rff", "--ResultSaveFreq", type=int, default=10000)
parser.add_argument("-msp", "--ModelSavePath", type=str, default='models_newvgg_1shot')
parser.add_argument("-msf", "--ModelSaveFreq", type=int, default=10000)

args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num
EXCLUDE_CLASS = args.exclude_class
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model


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


def main():
    # Step 1: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder(pretrained=args.loadImagenet)
    relation_network = RelationNetwork()

    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    # fine-tuning
    if (args.finetune):
        if os.path.exists(FEATURE_MODEL):
            feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
            print("load feature encoder success")
        else:
            print('Can not load feature encoder: %s' % FEATURE_MODEL)
            print('starting from scratch')
        if os.path.exists(RELATION_MODEL):
            relation_network.load_state_dict(torch.load(RELATION_MODEL))
            print("load relation network success")
        else:
            print('Can not load relation network: %s' % RELATION_MODEL)
            print('starting from scratch')

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=EPISODE // 10, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=EPISODE // 10, gamma=0.5)

    print("Training...")

    last_accuracy = 0.0

    for episode in range(args.start_episode, EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        samples, sample_labels, batches, batch_labels, chosen_classes = get_training_batch(CLASS_NUM,
                                                                                           SAMPLE_NUM_PER_CLASS,
                                                                                           BATCH_NUM_PER_CLASS)

        # calculate features
        sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 512, 7, 7)
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
        batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
        output = relation_network(relation_pairs, ft_list).view(-1, CLASS_NUM, 224, 224)

        mse = nn.MSELoss().cuda(GPU)
        loss = mse(output, Variable(batch_labels).cuda(GPU))

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss.cpu().data.numpy())

        if not os.path.exists(args.TrainResultPath):
            os.makedirs(args.TrainResultPath)
        if not os.path.exists(args.ModelSavePath):
            os.makedirs(args.ModelSavePath)

        # training result visualization
        if (episode + 1) % args.ResultSaveFreq == 0:
            support_output = np.zeros((224 * 2, 224 * SAMPLE_NUM_PER_CLASS, 3), dtype=np.uint8)
            query_output = np.zeros((224 * 3, 224 * DISPLAY_QUERY, 3), dtype=np.uint8)
            chosen_query = random.sample(list(range(0, BATCH_NUM_PER_CLASS)), DISPLAY_QUERY)

            for i in range(CLASS_NUM):
                for j in range(SAMPLE_NUM_PER_CLASS):
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
            for i in range(CLASS_NUM):
                for cnt, x in enumerate(chosen_query):
                    extra_label = batch_labels.numpy()[x][0]
                    extra_label[extra_label != 0] = 255
                    result1 = np.zeros((224, 224, 3), dtype=np.uint8)
                    result1[:, :, 0] = extra_label
                    result1[:, :, 1] = extra_label
                    result1[:, :, 2] = extra_label
                    extra[224 * 2:224 * 3, cnt * 224:(cnt + 1) * 224, :] = result1
            cv2.imwrite('%s/%s_query.png' % (args.TrainResultPath, episode), query_output)
            cv2.imwrite('%s/%s_show.png' % (args.TrainResultPath, episode), extra)
            cv2.imwrite('%s/%s_support.png' % (args.TrainResultPath, episode), support_output)

        # save models
        if (episode + 1) % args.ModelSaveFreq == 0:
            torch.save(feature_encoder.state_dict(), str(
                "./%s/feature_encoder_" % args.ModelSavePath + str(episode) + '_' + str(CLASS_NUM) + "_way_" + str(
                    SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
            torch.save(relation_network.state_dict(), str(
                "./%s/relation_network_" % args.ModelSavePath + str(episode) + '_' + str(CLASS_NUM) + "_way_" + str(
                    SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
            print("save networks for episode:", episode)


if __name__ == '__main__':
    main()
