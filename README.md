# Deep Learning 

## classification
分類問題を扱います.
| dataset |  resnet18 | resnext101 | efficientnetb0 |
| ----    | ----      | ----       | ----           |
| mnist   |    ○      |    ○       |         ○      |
| cifar10 |    ○      |    ○       |         ○      |

ResNet : https://arxiv.org/pdf/1512.03385.pdf  

ResNext: https://arxiv.org/pdf/1611.05431.pdf
 
EfficientNet: https://arxiv.org/pdf/1905.11946.pdf


### Example
mnistをefficientnetb0で学習する. 

`cd classification/src`

```sh
python train.py -o ../debug -i ./config/efficientnetb0.mnist.yaml -d ./config/mnist.yaml --log_level 10
```

### 実装解説
モデルは`torch.hub`を利用している. 
https://pytorch.org/hub/

これにより, 論文として公開されている学習済みモデルが入手でき, 転移学習やファインチューニングが可能となります.


## metric learning

mnist+resnet18でarcfaceを学習する. 
`cd metric/src`

```sh
python train.py -o ../debug -i ./config/arcface.yaml -d ./config/mnist.yaml --log_level 20
```

Arcface: https://arxiv.org/pdf/1801.07698.pdf

Triplet loss: https://arxiv.org/pdf/1412.6622.pdf

Beyond triplet loss:https://arxiv.org/pdf/1704.01719.pdf

## Notation
```sh
CUDA_VISIBLE_DEVICES=(Your GPU Device IDs) python ...
```

