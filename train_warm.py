"""train resnet."""
"""
用Models的方法来写
"""
import os
import argparse
import ast

import sys
sys.path.append(os.path.abspath('../src'))
import mindspore
from mindspore import context, set_seed, Model
from mindspore.nn import Momentum
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.communication import init
from mindspore.common import initializer
import mindspore.nn as nn
import mindspore.ops as ops
#from mindvision.engine.callback import LossMonitor #这个会控制lr
from mindspore.train.callback import ModelCheckpoint, LossMonitor
from mindvision.engine.callback import ValAccMonitor
from config import config_cub as config
from dataset import create_dataset
from tesnet_model import construct_TesNet
from loss_function import TesNet_Loss

import sys
sys.path.append(os.path.abspath('model_utils'))
set_seed(1)

parser = argparse.ArgumentParser(description='TesNet training')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--lr_decay_mode', type=str, default="cosine_decay_lr", help='Learning rate decay mode.')
parser.add_argument('--min_lr', type=float, default=0.0, help='The end learning rate.')
parser.add_argument('--max_lr', type=float, default=0.1, help='The max learning rate.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
args_opt = parser.parse_args()

class TesNetTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(TesNetTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, image,label):
        loss= self.network(image,label)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(image,label)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss

class TesNetWithEvalCell(nn.Cell):
    """自定义评估网络"""
    def __init__(self, network):
        super(TesNetWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        """输入数据为三个：一个数据及其对应的两个标签"""
        outputs,_ = self.network(data)
        return outputs, label


def warm_only(model):
    for p in model.features.trainable_params():
        p.requires_grad = False
    for p in model.add_on_layers.trainable_params():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.trainable_params():
        p.requires_grad = False

    print('\twarm ')

def joint(model):
    for p in model.features.trainable_params():
        p.requires_grad = True
    for p in model.add_on_layers.trainable_params():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.trainable_params():
        p.requires_grad = True

    print('\tjoint ')


def last_only(model, log=print):
    for p in model.features.trainable_params():
        p.requires_grad = False
    for p in model.add_on_layers.trainable_params():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.trainable_params():
        p.requires_grad = True

    log('\tlast layer')

if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE)

    device_id = int(os.getenv('DEVICE_ID', '2'))
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
    train_dataset = create_dataset(config.train_dir, config.batch_size, rank_size, rank_id,True)
    test_dataset = create_dataset(config.test_dir, config.test_batch_size, rank_size, rank_id,True)
    step_size = train_dataset.get_dataset_size()

    # define net
    tesnet_model = construct_TesNet(base_architecture=config.base_architecture, img_size=config.img_size,
                                    prototype_shape=(2000,64,1,1), num_classes=config.class_num,
                                    prototype_activation_function=config.prototype_activation_function, add_on_layers_type=config.add_on_layers_type)


    # define opt
    proto_params = list(filter(lambda x: 'prototype_vectors' in x.name, tesnet_model.trainable_params()))
    # warm opt
    warm_optimizer_specs = \
        [{'params': tesnet_model.add_on_layers.trainable_params(), 'lr': config.warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': proto_params, 'lr': config.warm_optimizer_lrs['prototype_vectors']},
         ]
    warm_optimizer =  nn.Adam(warm_optimizer_specs)
    #warm_optimizer = nn.Momentum(warm_optimizer_specs, momentum=0.9)
    if config.fixed_lr == "True":
        print("固定参数")
        joint_optimizer_specs = \
            [{'params': tesnet_model.features.trainable_params(), 'lr': config.joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
             # bias are now also being regularized
             {'params': tesnet_model.add_on_layers.trainable_params(), 'lr': config.joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
             {'params': proto_params, 'lr': config.joint_optimizer_lrs['prototype_vectors']},
             ]
    else:
        print("动态调整参数")
        joint_optimizer_specs = \
            [{'params': tesnet_model.features.trainable_params(), 'lr': nn.cosine_decay_lr(min_lr=0.0, max_lr=config.joint_optimizer_lrs['features'],
                                total_step=(config.num_train_epochs-config.num_warm_epochs) * step_size, step_per_epoch=step_size,
                                decay_epoch=(config.num_train_epochs-config.num_warm_epochs)), 'weight_decay': 1e-3},
             # bias are now also being regularized
             {'params': tesnet_model.add_on_layers.trainable_params(), 'lr': nn.cosine_decay_lr(min_lr=0.0, max_lr=config.joint_optimizer_lrs['add_on_layers'],
                                total_step=(config.num_train_epochs-config.num_warm_epochs) * step_size, step_per_epoch=step_size,
                                decay_epoch=(config.num_train_epochs-config.num_warm_epochs)), 'weight_decay': 1e-3},
             {'params': proto_params, 'lr': nn.cosine_decay_lr(min_lr=0.0, max_lr=config.joint_optimizer_lrs['prototype_vectors'],
                                total_step=(config.num_train_epochs-config.num_warm_epochs) * step_size, step_per_epoch=step_size,
                                decay_epoch=(config.num_train_epochs-config.num_warm_epochs))},
             ]
    lr_adaptive =  nn.cosine_decay_lr(min_lr=0.0, max_lr=config.joint_optimizer_lrs['add_on_layers'],
                                total_step=(config.num_train_epochs-config.num_warm_epochs) * step_size, step_per_epoch=step_size,
                                decay_epoch=(config.num_train_epochs-config.num_warm_epochs))
    joint_optimizer = nn.Momentum(joint_optimizer_specs,learning_rate=lr_adaptive,momentum=0.9)

    net_with_loss = TesNet_Loss(tesnet_model,config)


    eval_net = TesNetWithEvalCell(tesnet_model)
    eval_net.set_train(False)

    metrics = {"Accuracy": nn.Accuracy()}
    warm_model = Model(network=net_with_loss, eval_network=eval_net,loss_fn=None, optimizer=warm_optimizer,metrics={'acc'})
    joint_model = Model(network=net_with_loss, eval_network=eval_net,loss_fn=None, optimizer=joint_optimizer, metrics={'acc'})
    #回调 保存训练模型
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="{}_warm".format(config.base_architecture), directory="{}/{}_warm_ckpt".format(config.save_checkpoint_path,config.base_architecture), config=config_ck)
        cb += [ckpt_cb]

    warm_only(tesnet_model)
    warm_model.train(4, train_dataset, callbacks=cb, sink_size=step_size, dataset_sink_mode=False)

