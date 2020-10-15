import os
import subprocess

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from src.data_preprocessing.batch_generator import get_autolabel_batch
from src.display_data import maskimg
from src.evaluation.iou import iou
from src.network import CNNEncoder, RelationNetwork


def main(class_num, sample_num_per_class, batch_num_per_class, encoder_save_path, network_save_path,
         use_gpu, gpu, support_dir, test_dir, result_dir):
    # Step 1: init neural networks and feature encoder
    print("init neural networks")
    feature_encoder, relation_network = init_encoder_and_network(encoder_save_path, gpu, network_save_path, use_gpu)
    # Step 2: init result folder
    classname = support_dir
    init_result_path(classname, result_dir)
    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)
    # Step 3: Testing
    print("Testing...")
    testnames = os.listdir('%s' % test_dir)
    #print(testnames)
    print('%s testing images in class %s' % (len(testnames), classname))
    for cnt, testname in enumerate(testnames):
        if cv2.imread('%s/%s' % (test_dir, testname)) is None:
            continue
        # Step 3a: Get samples and batches (or support and query split)
        samples, sample_labels, batches, batch_labels = get_autolabel_batch(testname, class_num, sample_num_per_class,
                                                                            batch_num_per_class, support_dir, test_dir)
        # Step 3b: Predict the new label
        ft_list, relation_pairs = get_relation_pairs_and_encoded_features(batch_num_per_class, batches, class_num,
                                                                          feature_encoder, gpu, sample_num_per_class,
                                                                          samples, use_gpu)
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)
        # Step 3c: Evaluate the new label
        classiou, stick, batch_index = compute_iou_for_query(batch_labels, batches, output, stick, testname)
        # Step 3d: Visualize the autolabled mask and the image
        visualize_support_images_and_masks(classname, cnt, result_dir, sample_labels, samples)
        visualize_batch_images_and_masks(batch_index, batches, classname, cnt, result_dir, stick)


def visualize_support_images_and_masks(classname, cnt, result_dir, sample_labels, samples):
    if cnt == 0:
        for j in range(0, samples.size()[0]):
            suppimg = np.transpose(samples.numpy()[j][0:3], (1, 2, 0))[:, :, ::-1] * 255
            supplabel = np.transpose(sample_labels.numpy()[j], (1, 2, 0))
            supplabel = cv2.cvtColor(supplabel, cv2.COLOR_GRAY2RGB)
            supplabel = (supplabel * 255).astype(np.uint8)
            suppedge = cv2.Canny(supplabel, 1, 1)
            cv2.imwrite('./%s/%s/supp%s.png' % (result_dir, classname, j),
                        maskimg(suppimg, supplabel.copy()[:, :, 0], suppedge, color=[0, 255, 0]))


def visualize_batch_images_and_masks(batch_index, batches, classname, cnt, result_dir, stick):
    testimg = np.transpose(batches.numpy()[0][0:3], (1, 2, 0))[:, :, ::-1] * 255
    testlabel = stick[224 * 3:224 * 4, 224 * batch_index:224 * (batch_index + 1), :].astype(np.uint8)
    testedge = cv2.Canny(testlabel, 1, 1)
    cv2.imwrite('./%s/%s/test%s_raw.png' % (result_dir, classname, cnt), testimg)  # raw image
    cv2.imwrite('./%s/%s/test%s.png' % (result_dir, classname, cnt),
                maskimg(testimg, testlabel.copy()[:, :, 0], testedge))


def compute_iou_for_query(batch_labels, batches, output, stick, testname):
    classiou = 0
    for i in range(0, batches.size()[0]):
        # get prediction
        pred = output.data.cpu().numpy()[i][0]

        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        # vis
        demo = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) * 255
        stick[224 * 3:224 * 4, 224 * i:224 * (i + 1), :] = demo.copy()

        testlabel = batch_labels.numpy()[i][0].astype(bool)
        testlabel[testlabel <= 0.5] = 0
        testlabel[testlabel > 0.5] = 1
        # compute IOU
        iou_score = iou(pred, testlabel)
        classiou += iou_score
    classiou /= 1.0 * batches.size()[0]
    #print('iou=%0.4f for %s' % (classiou, testname))
    return classiou, stick, i


def get_relation_pairs_and_encoded_features(batch_num_per_class, batches, class_num, feature_encoder, gpu,
                                            sample_num_per_class, samples, use_gpu):
    if use_gpu:
        sample_features, _ = feature_encoder(Variable(samples).cuda(gpu))
    else:
        sample_features, _ = feature_encoder(Variable(samples))
    sample_features = sample_features.view(class_num, sample_num_per_class, 512, 7, 7)
    sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
    if use_gpu:
        batch_features, ft_list = feature_encoder(Variable(batches).cuda(gpu))
    else:
        batch_features, ft_list = feature_encoder(Variable(batches))
    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_num_per_class * class_num, 1, 1, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
    return ft_list, relation_pairs


def init_result_path(classname, result_dir):
    if os.path.exists(result_dir):
        os.system('rm -r %s' % result_dir)
    if os.path.exists('result.zip'):
        os.system('rm result.zip')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists('./%s/%s' % (result_dir, classname)):
        os.makedirs('./%s/%s' % (result_dir, classname))


def init_encoder_and_network(encoder_save_path, gpu, network_save_path, use_gpu):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) \
                                                            for x in subprocess.Popen("nvidia-smi -q -d Memory |\
                                        grep -A4 GPU | grep Free", shell=True,
                                                                                      stdout=subprocess.PIPE).stdout.readlines()]))
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()
    if use_gpu:
        feature_encoder.cuda(gpu)
        relation_network.cuda(gpu)
    if os.path.exists(encoder_save_path):
        feature_encoder.load_state_dict(torch.load(encoder_save_path, map_location=torch.device('cpu')))
        print("load feature encoder success")
    else:
        raise Exception('Can not load feature encoder: %s' % encoder_save_path)
    if os.path.exists(network_save_path):
        relation_network.load_state_dict(torch.load(network_save_path, map_location=torch.device('cpu')))
        print("load relation network success")
    else:
        raise Exception('Can not load relation network: %s' % network_save_path)
    return feature_encoder, relation_network
