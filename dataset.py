#数据预处理


"""
create train or eval dataset.
"""
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
#import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose

# 创建数据集
def create_dataset(dataset_path, batch_size=32, rank_size=1, rank_id=0, do_train=True):

    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=do_train,
                                     num_shards=rank_size, shard_id=rank_id)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    trans = [
        C.Decode(),
        C.Resize([224,224]),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)  # 精度转换

    # call data operations by map
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations batch_size
    data_set = data_set.batch(batch_size, drop_remainder=do_train)

    return data_set

def create_push_dataset(dataset_path, batch_size=32, rank_size=1, rank_id=0, do_train=False):

    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=do_train,
                                     num_shards=rank_size, shard_id=rank_id)

    transform_list = Compose([py_vision.Decode(),
                              py_vision.Resize([224,224]),
                              py_vision.ToTensor()])

    type_cast_op = C2.TypeCast(mstype.int32)  # 精度转换

    # call data operations by map
    data_set = data_set.map(operations=transform_list, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations batch_size
    data_set = data_set.batch(batch_size, drop_remainder=do_train)

    return data_set

