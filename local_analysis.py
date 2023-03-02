import mindspore
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from mindspore import load_checkpoint, load_param_into_net
from src.config import config_cub as config
from src.dataset import create_dataset
from src.tesnet_model import construct_TesNet
import re

import os
import copy
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose

from model_utils.helpers import makedir, find_high_activation_crop
import train_and_test as tnt
from model_utils.log import create_logger
from model_utils.preprocess import mean, std, undo_preprocess_input_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='7')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-img', nargs=1, type=str)
parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
parser.add_argument('-test_model_path', nargs=1, type=str)
args = parser.parse_args()


# specify the test image to be analyzed
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init
# 静态图 context.set_context(mode=context.GRAPH_MODE)
# 动态图 context.set_context(mode=context.PYNATIVE_MODE)
context.set_context(mode=context.PYNATIVE_MODE)

device_id = int(os.getenv('DEVICE_ID', args.gpuid[0]))
rank_size = int(os.getenv('RANK_SIZE', '1'))
rank_id = int(os.getenv('RANK_ID', '0'))

# init context
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=device_id)
if rank_size > 1:
    context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)
    context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
    init()

image_dir = "./local_analysis/class29"
image_name = "American_Crow_0001_25053.jpg"
image_label =  28

test_image_dir = image_dir
test_image_name = image_name
test_image_label = image_label

test_image_path = os.path.join(test_image_dir, test_image_name)

load_model_dir = "./saved_model/resnet34/"
load_model_name = "resnet34_tesnet_weight_epoch10.ckpt"

model_base_architecture = "resnet34"
experiment_run = "test_123"

save_analysis_path = os.path.join(test_image_dir, model_base_architecture, experiment_run, load_model_name)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir,load_model_name)
epoch_number_str = "10"
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

# 载入模型
tesnet_model = construct_TesNet(base_architecture=config.base_architecture, img_size=config.img_size,
                                prototype_shape=(2000, 64, 1, 1), num_classes=config.class_num,
                                prototype_activation_function=config.prototype_activation_function,
                                add_on_layers_type=config.add_on_layers_type)
#
param_dict = load_checkpoint(args.test_model_path)
load_param_into_net(tesnet_model, param_dict)


img_size = tesnet_model.img_size
prototype_shape = tesnet_model.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-' + epoch_number_str, 'bb' + epoch_number_str + '.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = ops.Argmax(axis=0,output_type=mindspore.int32)(tesnet_model.last_layer.weight, )
prototype_max_connection = prototype_max_connection.asnumpy()
if np.sum(prototype_max_connection == prototype_img_identity) == tesnet_model.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')


##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index + 1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.asnumpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])

    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch), 'prototype-img' + str(index) + '.png'))
    # plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-' + str(epoch),
                                    'prototype-img-original_with_self_act' + str(index) + '.png'))
    # plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(
        os.path.join(load_img_dir, 'epoch-' + str(epoch), 'prototype-img-original' + str(index) + '.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)


# load the test image and forward it through the network
preprocess = Compose([
    #py_vision.Decode(),
    py_vision.Resize([224,224]),
    py_vision.ToTensor(),
    py_vision.Normalize(mean,std)
])

img_pil = Image.open(test_image_path)
img_tensor = preprocess(img_pil)[0] #读出来是个tuple

ops_expand_dims = ops.ExpandDims()
img_variable = mindspore.Tensor(img_tensor)
img_variable = ops_expand_dims(img_variable,0)

images_test = img_variable
labels_test = mindspore.Tensor([test_image_label],dtype=mindspore.int32)

project_distances, cosine_distances = tesnet_model.prototype_distances(images_test)
prototype_activations = tesnet_model.global_max_pooling(project_distances)

logits, min_distances = tesnet_model(images_test)
conv_output, distances = tesnet_model.push_forward(images_test)
prototype_activation_patterns = -distances

tables = []
for i in range(logits.shape[0]):
    tables.append((ops.Argmax(axis=1,output_type=mindspore.int32)(logits)[i].item(0), labels_test[i].item(0)))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
op_sort = ops.Sort()
array_act, sorted_indices_act = op_sort(prototype_activations[idx])
for i in range(1, 11):
    log('top {0} activated prototype for this image:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act[-i].item(0))
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act[-i].item(0),
                                          bbox_height_start=prototype_info[sorted_indices_act[-i].item(0)][1],
                                          bbox_height_end=prototype_info[sorted_indices_act[-i].item(0)][2],
                                          bbox_width_start=prototype_info[sorted_indices_act[-i].item(0)][3],
                                          bbox_width_end=prototype_info[sorted_indices_act[-i].item(0)][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act[-i].item(0))
    log('prototype index: {0}'.format(sorted_indices_act[-i].item(0)))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item(0)]))
    if prototype_max_connection[sorted_indices_act[-i].item(0)] != prototype_img_identity[sorted_indices_act[-i].item(0)]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item(0)]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(
        tesnet_model.last_layer.weight[predicted_cls][sorted_indices_act[-i].item(0)]))

    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item(0)].asnumpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                              interpolation=cv2.INTER_CUBIC)

    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                     high_act_patch_indices[2]:high_act_patch_indices[3], :]
    log('most highly activated patch of the chosen image by this prototype:')
    # plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    log('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                        'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    log('prototype activation map of the chosen image:')
    # plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    log('--------------------------------------------------------------')

##### PROTOTYPES FROM TOP-k CLASSES
k = 10
log('Prototypes from top-%d classes:' % k)
op_topk  = ops.TopK()
topk_logits, topk_classes = op_topk(logits[idx], k)
for i, c in enumerate(topk_classes):
    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1)))

    log('top %d predicted class: %d' % (i + 1, c))
    log('logit of the class: {}'.format(topk_logits[i]))
    class_prototype_indices = np.nonzero(tesnet_model.prototype_class_identity.asnumpy()[:, c])[0]
    class_prototype_indices = mindspore.Tensor(class_prototype_indices,dtype=mindspore.int32)
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = op_sort(class_prototype_activations)

    #class_prototype_indices = class_prototype_indices.asnumpy()
    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act):
        prototype_index = class_prototype_indices[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                       start_epoch_number, prototype_index)
        save_prototype_original_img_with_bbox(
            fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                               'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
            epoch=start_epoch_number,
            index=prototype_index,
            bbox_height_start=prototype_info[prototype_index][1],
            bbox_height_end=prototype_info[prototype_index][2],
            bbox_width_start=prototype_info[prototype_index][3],
            bbox_width_end=prototype_info[prototype_index][4],
            color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                       start_epoch_number, prototype_index)
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(tesnet_model.last_layer.weight[c][prototype_index]))

        activation_pattern = prototype_activation_patterns[idx][prototype_index].asnumpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC)

        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        # plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                   high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        # plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                   overlayed_img)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')

if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')

logclose()
