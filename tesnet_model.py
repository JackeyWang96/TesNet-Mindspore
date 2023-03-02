import mindspore
from mindspore import ops
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal
#import models
from models.vgg.vgg16_features import vgg16_features
from models.vgg.vgg19_features import vgg19_features
from models.resnet.resnet34_features import resnet34_features
from models.resnet.resnet50_features import resnet50_features
from models.resnet.resnet152_features import resnet152_features
from models.densenet.densenet121_features import densnet121_features
from models.densenet.densenet161_features import densnet161_features
from model_utils.receptive_field import compute_proto_layer_rf_info_v2

base_architecture_to_features = {
                                'vgg16': vgg16_features,
                                'vgg19': vgg19_features,
                                'resnet34':resnet34_features,
                                'resnet50':resnet50_features,
                                'resnet152':resnet152_features,
                                'densnet121':densnet121_features,
                                'densnet161':densnet161_features
                                 }


class TESNet(nn.Cell):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(TESNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function  # log

        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        zeros = ops.Zeros()
        prototype_class_identity_shape = (self.num_prototypes,self.num_classes)
        self.prototype_class_identity = zeros(prototype_class_identity_shape, mstype.float32)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features  #

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = 512
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(
                    nn.Conv2d(in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1,
                              pad_mode='pad', has_bias=True))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(
                    nn.Conv2d(in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1,
                              pad_mode='pad', has_bias=True))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.SequentialCell([*add_on_layers])
        else:
            layers = []
            layers.extend([
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1, pad_mode='pad', has_bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1,
                          pad_mode='pad', has_bias=True),
                nn.Sigmoid()])
            self.add_on_layers = nn.SequentialCell(layers)
        uniformreal  = ops.UniformReal(seed=2)
        output = uniformreal(self.prototype_shape)
        self.prototype_vectors = mindspore.Parameter(output,requires_grad=True)
        #self.update = nn.ParameterUpdate(self.prototype_vectors)
        #self.update.phase = "update_param"

        ones = ops.Ones()
        self.ones = mindspore.Parameter(ones(self.prototype_shape, mstype.float32),
                                 requires_grad=False)

        self.last_layer = nn.Dense(in_channels=self.num_prototypes, out_channels=self.num_classes, has_bias=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):

        x = self.features(x) #x [1,512,7,7]
        x = self.add_on_layers(x)

        return x
    def normalize_prototype_vectors(self):
        l2_normalize = ops.L2Normalize(axis=1)
        now_prototype_vectors = l2_normalize(self.prototype_vectors)
        #self.prototype_vectors = self.update(now_prototype_vectors)
        #self.prototype_vectors.set_data(now_prototype_vectors)
        return now_prototype_vectors

    def _cosine_convolution(self, x):

        l2_normalize = ops.L2Normalize(axis=1)
        #x = F.normalize(x, p=2, dim=1)
        x = l2_normalize(x)
        now_prototype_vectors = l2_normalize(self.prototype_vectors)
        conv2d = ops.Conv2D(out_channel = self.prototype_vectors.shape[0], kernel_size=self.prototype_vectors.shape[-1])# x.shape (1,128,7,7) (2000,128,1,1)->(b,2000,7,7)
        distances = conv2d(x, now_prototype_vectors)
        distances = -distances

        return distances

    def _project2basis(self, x):
        l2_normalize = ops.L2Normalize(axis=1)
        now_prototype_vectors = l2_normalize(self.prototype_vectors)
        conv2d = ops.Conv2D(out_channel=self.prototype_vectors.shape[0], kernel_size=self.prototype_vectors.shape[-1])
        distances = conv2d(x, now_prototype_vectors)
        return distances

    def prototype_distances(self, x):

        conv_features = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)

        return project_distances, cosine_distances

    def distance_2_similarity(self, distances):

        if self.prototype_activation_function == 'log':
            log = ops.Log()
            return log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def global_min_pooling(self, distances):

        pool = nn.MaxPool2d(kernel_size=distances.shape[-1],stride=1,pad_mode="valid")
        mini_distances = -pool(-distances)
        mini_distances = mini_distances.view(-1,self.num_prototypes)
        return mini_distances

    def global_max_pooling(self, distances):
        pool = nn.MaxPool2d(kernel_size=distances.shape[-1], stride=1, pad_mode="valid")
        max_distances = pool(distances)
        max_distances = max_distances.view(-1, self.num_prototypes)
        return max_distances

    def construct(self, x):

        prototype_vectors = self.normalize_prototype_vectors()
        project_distances, cosine_distances = self.prototype_distances(x)
        cosine_min_distances = self.global_min_pooling(cosine_distances)

        project_max_distances = self.global_max_pooling(project_distances)
        prototype_activations = project_max_distances
        logits = self.last_layer(prototype_activations)
        return logits, cosine_min_distances

    def push_forward(self, x):

        conv_output = self.conv_features(x)  # [batchsize,128,14,14]

        distances = self._project2basis(conv_output)
        distances = - distances
        return conv_output, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):

        transpose = ops.Transpose()
        positive_one_weights_locations = transpose(self.prototype_class_identity,(1,0))
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        value = correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations
        self.last_layer.weight.set_data(value)#

    def _initialize_weights(self):
        for _, cell in self.add_on_layers.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'), cell.weight.shape, cell.weight.dtype))

                if cell.bias is not None:

                    cell.bias.set_data(initializer(0,cell.bias.shape,cell.bias.dtype))

            elif isinstance(cell, nn.BatchNorm2d):

                cell.weight.set_data(initializer(0,cell.weight.shape,cell.weight.dtype))
                cell.bias.set_data(initializer(0,cell.bias.shape,cell.bias.dtype))

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_TesNet(base_architecture,  img_size=224,
                     prototype_shape=(2000, 64, 1, 1), num_classes=200,
                     prototype_activation_function='log',
                     add_on_layers_type='regular'):
    features = base_architecture_to_features[base_architecture]()
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,  # 224
                                                         layer_filter_sizes=layer_filter_sizes,  #
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return TESNet(features=features,
                  img_size=img_size,
                  prototype_shape=prototype_shape,
                  proto_layer_rf_info=proto_layer_rf_info,
                  num_classes=num_classes,
                  init_weights=True,
                  prototype_activation_function=prototype_activation_function,
                  add_on_layers_type=add_on_layers_type)


if __name__ == "__main__":

    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE)
    tesnet_model = construct_TesNet(base_architecture="resnet34")
    input_shape = (1, 3, 224, 224)
    input_tensor = mindspore.Tensor(shape = input_shape, dtype = mindspore.dtype.float32, init= mindspore.common.initializer.Normal())
    print(tesnet_model.parameters_dict())
    logits, cosine_min_distances  = tesnet_model(input_tensor)
    print(logits.shape, cosine_min_distances.shape)
