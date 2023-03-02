"""test resnet."""
import os
import argparse
import ast

import mindspore
from mindspore import context, set_seed, Model
from mindspore.nn import Momentum
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication import init
from mindspore.common import initializer
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net
from src.config import config_cub as config
from src.dataset import create_dataset
from src.tesnet_model import construct_TesNet
from src.loss_function import TesNet_Loss


import sys
sys.path.append(os.path.abspath('model_utils'))
set_seed(1)

parser = argparse.ArgumentParser(description='TesNet training')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--test_model_path', type=str, default=None, help='test model path')
args_opt = parser.parse_args()

class TesNetWithEvalCell(nn.Cell):
    """自定义评估网络"""
    def __init__(self, network):
        super(TesNetWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        """输入数据为三个：一个数据及其对应的两个标签"""
        outputs,_ = self.network(data)
        return outputs, label


if __name__ == '__main__':
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE)


    device_id = int(os.getenv('DEVICE_ID', '5'))
    rank_size = int(os.getenv('RANK_SIZE', '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=device_id)
    if rank_size > 1:
       context.set_auto_parallel_context(device_num=rank_size, parallel_mode=ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True)
       context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
       init()

    # create dataset
    test_dataset = create_dataset(config.test_dir, config.test_batch_size, rank_size, rank_id)

    # define net
    tesnet_model = construct_TesNet(base_architecture=config.base_architecture, img_size=config.img_size,
                                    prototype_shape=(2000,64,1,1), num_classes=config.class_num,
                                    prototype_activation_function=config.prototype_activation_function, add_on_layers_type=config.add_on_layers_type)

    param_dict = load_checkpoint(args_opt.test_model_path)
    load_param_into_net(tesnet_model, param_dict)


    eval_net = TesNetWithEvalCell(tesnet_model)
    accu = nn.Accuracy('classification')
    accu.clear()
    eval_net.set_train(False)

    for data in test_dataset.create_dict_iterator():
        outputs = eval_net(data["image"], data["label"])
        accu.update(outputs[0], outputs[1])
    accu_result = accu.eval()
    print("accu: ", accu_result)

