import argparse
import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from network import CNNEncoder, RelationNetwork

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
parser.add_argument("-sd", "--support_dir", type=str, default='data/african_elephant/supp')
parser.add_argument("-td", "--test_dir", type=str, default='data/african_elephant/test')
args = parser.parse_args()

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


def get_oneshot_batch(testname):  # shuffle in query_images not done
    # classes_name = os.listdir('./%s' % args.support_dir)
    # classes_name = ['android_robot', 'bucket_water' , 'nintendo_gameboy']

    support_images = np.zeros((CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((CLASS_NUM * SAMPLE_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32)
    query_images = np.zeros((CLASS_NUM * BATCH_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32)
    query_labels = np.zeros((CLASS_NUM * BATCH_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32)
    zeros = np.zeros((CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 224, 224), dtype=np.float32)
    class_cnt = 0

    # print ('class %s is chosen' % i)
    # classnames = ['english_foxhound', 'guitar']
    imgnames = os.listdir('./%s/label' % args.support_dir)
    # print (args.support_dir, imgnames)
    testnames = os.listdir('%s' % args.test_dir)
    indexs = list(range(0, len(imgnames)))[0:5]
    chosen_index = indexs
    j = 0
    for k in chosen_index:
        # process image
        image = cv2.imread('%s/image/%s' % (args.support_dir, imgnames[k].replace('.png', '.jpg')))
        if image is None:
            raise Exception('cannot load image ')

        image = image[:, :, ::-1]  # bgr to rgb
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        # labels
        # print ('%s/label/%s' % (args.support_dir, imgnames[k]))
        label = cv2.imread('%s/label/%s' % (args.support_dir, imgnames[k]))[:, :, 0]

        support_images[k] = image
        support_labels[k][0] = label

    testimage = cv2.imread('%s/%s' % (args.test_dir, testname))
    testimage = cv2.resize(testimage, (224, 224))
    testimage = testimage[:, :, ::-1]  # bgr to rgb
    testimage = testimage / 255.0
    testimage = np.transpose(testimage, (2, 0, 1))

    query_images[0] = testimage

    class_cnt += 1
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor, support_labels_tensor), dim=1)

    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r  # / 255.0
    rgb[:, :, 1] = g  # / 255.0
    rgb[:, :, 2] = b  # / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


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
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    # metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    if os.path.exists(FEATURE_MODEL):
        feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
        print("load feature encoder success")
    else:
        raise Exception('Can not load feature encoder: %s' % FEATURE_MODEL)
    if os.path.exists(RELATION_MODEL):
        relation_network.load_state_dict(torch.load(RELATION_MODEL))
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

        samples, sample_labels, batches, batch_labels = get_oneshot_batch(testname)

        # forward
        sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 512, 7, 7)
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
        batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))
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
