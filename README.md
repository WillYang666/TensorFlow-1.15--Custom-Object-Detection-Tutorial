# TensorFlow-TensorFlow-Custom-Object-Detection
本文内容来自于我的毕业设计，基于 TensorFlow 1.15.0，其他 TensorFlow 版本运行可能存在问题。
---
## 实现效果：
![factory_demo](/assets/factory_demo.gif)

## 环境配置
### TensorFlow 官方文件下载
* 首先在你想放置这个文件夹的路径下新建一个文件夹，这里我取名叫做 `TensorFlow`，路径为`C:\Users\Liam\TensorFlow`
* 下载 TensorFlow 的[官方 models](https://github.com/tensorflow/models)，注意需要使用 TensorFlow 1.x 版本，最新版本（2.x）不兼容1.x。这里我将我使用的 TensorFlow 1.15 版本给出，[百度网盘下载地址]()。
* 将下载的`models`文件夹放置到`TensorFlow`路径下，此时应该得到如下文件树结构：
```
TensorFlow/
└─ models/
   ├─ official/
   ├─ research/
   └── ...
```

* 在 `TensorFlow` 文件夹下新建 `workspace`文件夹，在 `workspace` 文件夹中再新建 `training_demo` 文件夹，在这里我们会放置训练自己的目标识别模型的一些文件。
```
TensorFlow/
├─ models/
│  ├─ official/
│  ├─ research/
│  └─ ...
└─ workspace/
   └─ training_demo/
```

*  `training_demo`文件夹中，结构如下

```
training_demo/
├─ annotations/
├─ exported-models/
├─ images/
│  ├─ test/
│  └─ train/
├─ models/
├─ pre-trained-models/
└─ README.md
```
这些文件夹的作用解释如下：
* `annotations`：存放之后由图片数据集标注生成的` *.csv` 文件和 `*.record` 文件。
* `exported-models`：放置最终导出的模型。
* `images`：用于保存制造训练集的图片（包括 JPG 图片和使用 labelImg 标注生成的`*.xml`文件。
    * `images/train`:保存训练用的图片文件及`*.xml`文件。
    * `images/test`：保存测试用的图片文件及`*.xml`文件。
* `models`：用于保存所使用的模型，配置文件`*.config`， 还有训练过程中生成的一些文件。
* `pre-trained-models`：保存下载的预训练模型，我们会利用其中的一些初始参数。
* `README.md`：解释性文档，有的话可以更清楚知道做了什么。


###  Anaconda 下载及使用
在[这个网站](https://www.anaconda.com/products/individual)上下载 Anaconda 64-Bit 版本，如需详细安装教程，可以查看 [Anaconda 官方文档](https://docs.anaconda.com/anaconda/install/windows/)

#### 建立一个新的 Anaconda 虚拟环境
Anaconda 的好处在于能够实现在一台机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间进行切换。这里我们创建一个专门用来实现本文效果的虚拟环境。
* 首先打开一个新的命令行窗口
* 键入如下命令：

```conda
conda create -n tensorflow pip python=3.6
```

* 上述命令可以创建一个新的虚拟环境，名字为`mytensorflow`

#### 启动创建好的 Anaconda 虚拟环境
* 如果需要启动这个名字为`mytensorflow`的虚拟环境，在命令行键入：

```conda
conda activate mytensorflow
```

* 一切正常的话，则可以在命令行中看见如下效果（关键在于前面的mytensorflow）：

```
(mytensorflow) C:\Users\Liam>
```

#### 利用 Anaconda 下载所需要的软件包
 打开 Anaconda 命令行，进行所需依赖包的下载。
 注意⚠️：由于国内网络的特性，下载的速度也许会很慢甚至卡死，因此我们可以更换成国内的源进行加速下载。
* 每次切换源之前都需要运行以下命令切换回默认源，再进行源的更改。

```conda
conda config --remove-key channels
```


**清华源**

```conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

**上交源**

```conda
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/mai
```

**中科大源**

```conda
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main
```

##### TensorFlow 1.15.0 的安装

* 在 Anaconda 命令行中，键入以下命令以安装 TensorFlow 1.15:

```conda
conda install tensorflow-gpu=1.15.0 #gpu版本安装
conda install tensorflow=1.15.0 #cpu版本安装
```

##### 其他依赖包安装

```conda
conda install -c anaconda protobuf
pip install -i https://pypi.doubanio.com/simple pillow
pip install -i https://pypi.doubanio.com/simple lxml
pip install -i https://pypi.doubanio.com/simple jupyter
pip install -i https://pypi.doubanio.com/simple matplotlib
pip install -i https://pypi.doubanio.com/simple pandas
pip install -i https://pypi.doubanio.com/simple opencv-python
pip install -i https://pypi.doubanio.com/simple cython
```

### 设置 Python 的运行目录
* 每次`activate`你的虚拟环境时，都需要设置 Python 的运行目录，将下面的`mytensorflow`路径修改成你实际放置文件的目录

```
set PYTHONPATH=C:\Liam\TensorFlow\models;C:\Liam\TensorFlow\models\research;C:\Liam\TensorFlow\models\research\slim
```

### 配置 protobuf
* 在`C:\Liam\TensorFlow\models\research`目录下执行以下命令：


```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto  .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```

### 执行 setup files
* 在`C:\Liam\TensorFlow\models\research`目录下执行以下命令：


```
python setup.py build
python setup.py install
```

## 目标识别模型搭建
### 数据集标注
* 通过搜寻图片数据集，利用 [labelImg](https://github.com/tzutalin/labelImg) 进行标注，生成 `*.xml` 文件。（你可以观看[这个视频](https://youtu.be/K_mFnvzyLvc?t=9m13s)来学习如何使用 labelImg。
* 把生成的`*.xml` 文件放入`training_demo/images`文件夹中。

#### 数据集分割（非必需）
将数据分割成训练和测试两部分的原因是可以在训练模型的过程中得出模型的准确率，进而判断模型的可用性。

* 在`TensorFlow`文件夹中，新建一个文件夹`TensorFlow/scripts`用于保存 python 脚本。
* 使用 Google Drive 中提供的`scripts`文件夹中的`partition_dataset.py`脚本将图片集进行分割，命令为：

```
python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1
```

* 完成之后，图片和对应的`*.xml` 文件就以 9:1 的比例分配到`training_demo/images/train`和`training_demo/images/test`,文件夹中了。

#### 利用 xml 文件生成 csv 文件
* 在`scripts`文件夹中找到`generate_csv.py`文件，将`*. xml`文件生成 `*.csv`文件，语句如下：


```
python generate_csv.py --input=[PATH_TO_IMAGES_FOLDER] --output=[PATH_TO_DESTINATION_FOLDER]/annotations.csv
```

#### 创建 Label Map
* 新建一个`txt`文档，将你要识别的几种类型写入，注意，你在 labelImg 中如何标注的，这里的名字要对应上，这里以 girl 和 boy 为例：


```
item {
    id: 1
    name: 'girl'
}

item {
    id: 2
    name: 'boy'
}
```

* 如果你有更多类型的话，按照此格式继续填写即可。
* 最后，将`txt`文件改名字为`label_map.pbtxt`

#### 利用 csv 文件生成 record 文件
* 打开`scripts`文件夹中的`generate_tfrecord.py`，修改其中的32行中的内容为你自己要识别的目标名称（这里以`girl`和`boy`为例，注意顺序需要和`label_map.pbtxt`中一致）


```  
if row_label == 'girl':
        return 1
    if row_label == 'boy':
        return 2
    else:
        return 0
```

### 下载及配置模型
* 在 [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) 中下载合适的模型。注意：TensorFlow 1.x 和 TensorFlow 2.x 的模型并不通用。这里以 `faster_rcnn_inception_v2_coco`模型为例。
* 打开下载好的模型中的`pipeline.config`进行修改
    * 需要修改的地方为：
        *  num_classes
        *  fine_tune_checkpoint
        *  label_map_path
        *   input_path (train）
        *   input_path (eval）


```
model {
  faster_rcnn {
    num_classes: 2 #修改成你的识别个数，比如这里我们就是 boy 和 girl
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }
    feature_extractor {
      type: "faster_rcnn_inception_v2"
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        height_stride: 16
        width_stride: 16
        scales: 0.25
        scales: 0.5
        scales: 1.0
        scales: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 1.0
        aspect_ratios: 2.0
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.00999999977648
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.699999988079
    first_stage_max_proposals: 100
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
        use_dropout: false
        dropout_keep_probability: 1.0
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.300000011921
        iou_threshold: 0.600000023842
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}
train_config {
  batch_size: 1
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  optimizer {
    momentum_optimizer {
      learning_rate {
        manual_step_learning_rate {
          initial_learning_rate: 0.000199999994948
          schedule {
            step: 0
            learning_rate: 0.000199999994948
          }
          schedule {
            step: 900000
            learning_rate: 1.99999994948e-05
          }
          schedule {
            step: 1200000
            learning_rate: 1.99999999495e-06
          }
        }
      }
      momentum_optimizer_value: 0.899999976158
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"  # 修改成你下载的模型文件的地址
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt" #修改成你放 labelmap文件的地址，注意文件名
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/file_name.record" #这里修改成 record 文件的地址及你的 record 文件名字
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt" #修改成你放 labelmap文件的地址，注意文件名
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record" #修改成 record 文件的地址及你的 record 文件名字
  }
}
```

## 开始训练

* 键入以下语句开始训练（使用了`scripts`文件夹中的 `train.py`文件，需要注意文件夹地址要根据你的情况进行修改）：

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

### 利用 TensorBoard 查看训练情况

* 若要查看训练情况，可以使用`TensorBoard` 查看，再开一个命令行输入以下语句（注意需要激活虚拟环境及设置 python 运行路径）

```
tensorboard --logdir=training
```

### 将训练结果导出
* 使用`scripts`文件夹中的 `export_inference_graph.py`文件，注意路径问题，同时`model.ckpt-XXXX`中的`XXXX`代表你最终生成文件的那个数字。
* 导出以后的文件为`frozen_inference_graph.pb`文件

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

### 查看模型结果
* 使用`scripts`文件夹中的 `Object_detection_image.py`、`Object_detection_video.py`、`Object_detection_webcam.py`文件，即可查看图片、视频、摄像头的识别效果！
* 注意以上这几个文件中的细节修改，`Object_detection_image.py`中有示例。
