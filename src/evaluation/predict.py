import os

import numpy as np
import torch

from src.data_preprocessing.batch_generator import get_predict_batch
from src.evaluation.autolabel import get_relation_pairs_and_encoded_features, compute_iou_for_query
from src.network import CNNEncoder, RelationNetwork


def main(class_num, sample_num_per_class, batch_num_per_class, model_save_path,
         use_gpu, gpu, test_dir, result_dir):
    # Step 1: init neural networks and feature encoder
    print("init neural networks")
    feature_encoder, relation_network = init_encoder_and_network_for_predict(model_save_path, gpu, use_gpu, class_num,
                                                                             sample_num_per_class)
    # Step 2: init result folder
    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)
    # Step 3: Testing
    print("Testing...")
    classnames = os.listdir(test_dir)
    classiou_dict = dict()
    for classname in classnames:
        print(f'Testing images in class {classname}')
        # Step 3a: Get samples and batches (or support and query split)
        samples, sample_labels, batches, batch_labels = get_predict_batch(classname, class_num, sample_num_per_class,
                                                                          batch_num_per_class, test_dir)

        print('batch_labels sum =', batch_labels.sum())
        # Step 3b: Predict the new label
        ft_list, relation_pairs = get_relation_pairs_and_encoded_features(batch_num_per_class, batches, class_num,
                                                                          feature_encoder, gpu, sample_num_per_class,
                                                                          samples, use_gpu)
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)
        # Step 3c: Evaluate the new label
        print('output sum =', output.sum())
        classiou, stick, batch_index = compute_iou_for_query(batch_labels, batches, output, stick, classname)
        classiou_dict[classname] = classiou

    torch.save(classiou_dict, f"{result_dir}/class_iou_{class_num}_way_{sample_num_per_class}_shot.txt")


def init_encoder_and_network_for_predict(model_save_path, gpu, use_gpu, class_num, sample_num_per_class):
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()
    if use_gpu:
        feature_encoder.cuda(gpu)
        relation_network.cuda(gpu)
    relevent_models = [file_name for file_name in os.listdir(model_save_path) if
                       str(class_num) in file_name and str(sample_num_per_class) in file_name]
    relation_networks_paths = [f"{model_save_path}/{file_name}" for file_name in relevent_models if
                               file_name.startswith('relation_network_')]
    feature_encoders_paths = [f"{model_save_path}/{file_name}" for file_name in relevent_models if
                              file_name.startswith('feature_encoder_')]

    if os.path.exists(model_save_path):
        if use_gpu:
            feature_encoder.load_state_dict(torch.load(feature_encoders_paths[-1]))
        else:
            feature_encoder.load_state_dict(torch.load(feature_encoders_paths[-1], map_location=torch.device('cpu')))
        print("load feature encoder success")
    else:
        raise Exception(f"Can not load feature encoder: {model_save_path}/{feature_encoders_paths[-1]}")

    if os.path.exists(model_save_path):
        if use_gpu:
            relation_network.load_state_dict(torch.load(relation_networks_paths[-1]))
        else:
            relation_network.load_state_dict(torch.load(relation_networks_paths[-1], map_location=torch.device('cpu')))
        print("load relation network success")
    else:
        raise Exception(f"Can not load relation network: {model_save_path}/{relation_networks_paths[-1]}")

    return feature_encoder, relation_network