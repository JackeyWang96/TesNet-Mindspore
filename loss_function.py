import mindspore

from mindspore import Model
from mindspore import dataset as ds
from mindspore import nn as nn
from mindspore import ops as ops
from tesnet_model import construct_TesNet

import mindspore as ms
from mindspore import Tensor
from mindspore.nn.loss import LossBase

class CrossEntropySmooth(LossBase):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss

class TesNet_Loss(nn.Cell):
    def __init__(self, backbone, config):
        super(TesNet_Loss, self).__init__()
        self._backbone = backbone
        self._config= config #不同权重的比例
        #cross entropy
        self.ce_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0,num_classes=200)

        #op
        self.transpose_op = ops.Transpose()
        self.argmax_op = ops.ArgMaxWithValue(axis=1)
        self.mean_op = ops.ReduceMean(keep_dims=False)
        self.reshape_op = ops.Reshape()
        self.eye_op = ops.Eye()
        self.abs_op = ops.Abs()
        self.sum_op = ops.ReduceSum(keep_dims=True)
        self.sum_op2 = ops.ReduceSum(keep_dims=False)
        self.relu_op = nn.ReLU()
        self.expand_dims_op = ops.ExpandDims()
        self.square_op = ops.Square()
        self.sqrt_op = ops.Sqrt()
        #value
        self.sqrt2 = self.sqrt_op(mindspore.Tensor(2, mindspore.float32))
    def construct(self, input, label):
        output, min_distances = self._backbone(input)
        #ce loss
        cross_entropy = self.ce_loss(output, label)

        max_dist = (self._backbone.prototype_shape[1]
                    * self._backbone.prototype_shape[2]
                    * self._backbone.prototype_shape[3])  #
        prototypes_of_correct_class = self.transpose_op(self._backbone.prototype_class_identity[:, label], (1,0))


        _, inverted_distances = self.argmax_op((max_dist - min_distances) * prototypes_of_correct_class)

        #cluster_loss
        cluster_cost = self.mean_op(max_dist - inverted_distances)
        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        _, inverted_distances_to_nontarget_prototypes = \
            self.argmax_op((max_dist - min_distances) * prototypes_of_wrong_class)
        separation_cost = self.mean_op(max_dist - inverted_distances_to_nontarget_prototypes)  # calculate avg cluster cost

        # optimize orthogonality of prototype_vector
        cur_basis_matrix = self._backbone.prototype_vectors.squeeze()  # [2000,128]
        subspace_basis_matrix = self.reshape_op(cur_basis_matrix, (self._backbone.num_classes,
                                                              self._backbone.num_prototypes_per_class,
                                                              self._backbone.prototype_shape[1]))  # [200,10,128]
        subspace_basis_matrix_T = self.transpose_op(subspace_basis_matrix, (0,2,1))  # [200,10,128]->[200,128,10]
        orth_operator = ops.matmul(subspace_basis_matrix,
                                   subspace_basis_matrix_T)  # [200,10,128] [200,128,10] -> [200,10,10]

        I_operator = self.eye_op(subspace_basis_matrix.shape[1], subspace_basis_matrix.shape[1],mindspore.int32)  # [10,10]
        difference_value = orth_operator - I_operator  # [200,10,10]-[10,10]->[200,10,10]
        orth_cost = self.sum_op(self.relu_op(self.sum_op2(self.abs_op(difference_value), (1, 2)) - 0))  # [200]->[1]


        # subspace sep
        projection_operator = ops.matmul(subspace_basis_matrix_T,
                                               subspace_basis_matrix)  # [200,128,10] [200,10,128] -> [200,128,128]

        projection_operator_1 = self.expand_dims_op(projection_operator, 1)  # [200,1,128,128]
        projection_operator_2 = self.expand_dims_op(projection_operator, 0)  # [1,200,128,128]
        pairwise_distance = self.sqrt_op(self.sum_op2(self.square_op(self.abs_op(projection_operator_1 - projection_operator_2 + 1e-10)),(2,3)))
        subspace_sep = 0.5 * self.sum_op2(self.abs_op(pairwise_distance),(0,1)) / self.sqrt2

        #l1 mask
        l1_mask = 1 - self.transpose_op(self._backbone.prototype_class_identity, (1,0))

        l1 = self.sum_op2(self.abs_op(self._backbone.last_layer.weight * l1_mask))
        # weight 200,2000   prototype_class_identity [2000,200]
        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1e-4 * orth_cost - 1e-7 * subspace_sep

        return loss
