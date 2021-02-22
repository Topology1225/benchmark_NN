# Deep Learning 

## classification
分類問題を扱います.
| dataset |  resnet18 | resnext101 | efficientnetb0 |
|----     | ----      | ----       | ----           |
| mnist   |    ○      |    ○       |         ○      |
| cifar10 |    ○      |    ○       |         ○      |


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

