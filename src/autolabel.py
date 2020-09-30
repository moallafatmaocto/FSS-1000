import argparse
import os
import subprocess

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from src.data_preprocessing.batch_generator import get_autolabel_batch
from src.network import CNNEncoder, RelationNetwork

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=1)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=5)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=1)
parser.add_argument("-e", "--episode", type=int, default=50000)
# parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-d", "--display_query_num", type=int, default=5)
parser.add_argument("-t", "--test_class", type=int, default=1)
parser.add_argument("-modelf", "--feature_encoder_model", type=str, default='models/feature_encoder.pkl')
parser.add_argument("-modelr", "--relation_network_model", type=str, default='models/relation_network.pkl')
parser.add_argument("-sd", "--support_dir", type=str, default='test_data/african_elephant/supp')
parser.add_argument("-td", "--test_dir", type=str, default='test_data/african_elephant/test')
parser.add_argument("--use_gpu", type=bool, default=False)

args = parser.parse_args()

USE_GPU = args.use_gpu

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) \
                                                        for x in subprocess.Popen("nvidia-smi -q -d Memory |\
                                    grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
# TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num
TEST_CLASS = args.test_class
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model


def maskimg(img, mask, edge, color=[0, 0, 255], alpha=0.5):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask == 255] = color
    edge_layer = img.copy()
    edge_layer[edge == 255] = color
    out = cv2.addWeighted(edge_layer, 1, out, 0, 0, out)
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return (out)


def main():
    # Step 1: init test_data folders
    print("init test_data folders")
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    if USE_GPU:
        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)

    if os.path.exists(FEATURE_MODEL):
        feature_encoder.load_state_dict(torch.load(FEATURE_MODEL, map_location=torch.device('cpu')))
        print("load feature encoder success")
    else:
        raise Exception('Can not load feature encoder: %s' % FEATURE_MODEL)
    if os.path.exists(RELATION_MODEL):
        relation_network.load_state_dict(torch.load(RELATION_MODEL, map_location=torch.device('cpu')))
        print("load relation network success")
    else:
        raise Exception('Can not load relation network: %s' % RELATION_MODEL)

    print("Testing...")
    meaniou = 0
    classname = args.support_dir
    if os.path.exists('result1'):
        os.system('rm -r result1')
    if os.path.exists('result.zip'):
        os.system('rm result.zip')
    if not os.path.exists('result1'):
        os.makedirs('result1')
    if not os.path.exists('./result1/%s' % classname):
        os.makedirs('./result1/%s' % classname)
    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)
    support_image = np.zeros((5, 3, 224, 224), dtype=np.float32)
    support_label = np.zeros((5, 1, 224, 224), dtype=np.float32)
    supp_demo = np.zeros((224, 224 * 5, 3), dtype=np.uint8)
    supplabel_demo = np.zeros((224, 224 * 5, 3), dtype=np.uint8)

    testnames = os.listdir('%s' % args.test_dir)
    print('%s testing images in class %s' % (len(testnames), classname))

    for cnt, testname in enumerate(testnames):

        print('%s / %s' % (cnt, len(testnames)))
        print(testname)
        if cv2.imread('%s/%s' % (args.test_dir, testname)) is None:
            continue

        samples, sample_labels, batches, batch_labels = get_autolabel_batch(testname, CLASS_NUM, SAMPLE_NUM_PER_CLASS,
                                                                            BATCH_NUM_PER_CLASS, args.support_dir,
                                                                            args.test_dir)

        # forward
        if USE_GPU:
            sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
        else:
            sample_features, _ = feature_encoder(Variable(samples))

        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 512, 7, 7)
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
        if USE_GPU:
            batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))
        else:
            batch_features, ft_list = feature_encoder(Variable(batches))

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
        output = relation_network(relation_pairs, ft_list).view(-1, CLASS_NUM, 224, 224)

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
            pred = pred.astype(bool)
            # compute IOU
            overlap = testlabel * pred
            union = testlabel + pred
            iou = overlap.sum() / float(union.sum())
            # print ('iou=%0.4f' % iou)
            classiou += iou
        classiou /= 5.0

        # visulization
        if (cnt == 0):
            for i in range(0, samples.size()[0]):
                suppimg = np.transpose(samples.numpy()[i][0:3], (1, 2, 0))[:, :, ::-1] * 255
                supplabel = np.transpose(sample_labels.numpy()[i], (1, 2, 0))
                supplabel = cv2.cvtColor(supplabel, cv2.COLOR_GRAY2RGB)
                supplabel = (supplabel * 255).astype(np.uint8)
                suppedge = cv2.Canny(supplabel, 1, 1)

                cv2.imwrite('./result1/%s/supp%s.png' % (classname, i),
                            maskimg(suppimg, supplabel.copy()[:, :, 0], suppedge, color=[0, 255, 0]))
        testimg = np.transpose(batches.numpy()[0][0:3], (1, 2, 0))[:, :, ::-1] * 255
        testlabel = stick[224 * 3:224 * 4, 224 * i:224 * (i + 1), :].astype(np.uint8)
        testedge = cv2.Canny(testlabel, 1, 1)
        cv2.imwrite('./result1/%s/test%s_raw.png' % (classname, cnt), testimg)  # raw image
        cv2.imwrite('./result1/%s/test%s.png' % (classname, cnt), maskimg(testimg, testlabel.copy()[:, :, 0], testedge))


if __name__ == '__main__':
    main()
