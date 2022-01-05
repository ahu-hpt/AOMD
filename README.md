# AOMD: Architecture Oriented Mapping Distillation

Official implementation of Architecture Oriented Mapping Distillation has not yet been released, so stay tuned.
This repository contains source code of experiments for metric learning.


## Quick Start

```bash
python run.py --help    
python run_distill_aomd.py --help

# experiment on cars196
# base experiment
# Teacher Network
CUDA_VISIBLE_DEVICES=0 python run.py --dataset cars196 --epochs 40 --lr_decay_epochs 25 30 35 --lr_decay_gamma 0.5 --batch 128\
              --base resnet50 --sample distance --margin 0.2 --embedding_size 512 --save_dir cars196_resnet50_512

# Student Network(triplet)
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --triplet_ratio 1 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_triplet > cars196_triplet.log 2>&1 &

# AOMD(Changes to the mode are made in the /metric/loss.py)
# mode 1
# mode 2
CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode1_l2 > cars196_aomd_mode1_l2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python run_distill_aomd.py --dataset cars196 --epochs 80 --lr_decay_epochs 40 60 --lr_decay_gamma 0.1 --batch 128\
                      --base resnet18 --embedding_size 128 --l2normalize true --aomd_ratio 2e2 \
                      --teacher_base resnet50 --teacher_embedding_size 512 --teacher_load cars196_resnet50_512/best.pth \
                      --save_dir cars196_student_resnet18_128_aomd_mode2_l2 > cars196_aomd_mode2_l2.log 2>&1 &
```


##  Dependency

* Python 3.6
* Pytorch 1.0
* tqdm (pip install tqdm)
* h5py (pip install h5py)
* scipy (pip install scipy)

### Note
* Hyper-parameters that used for experiments in the paper are specified at scripts in ```exmples/```.
* Heavy teacher network (ResNet50 w/ 512 dimension) requires more than 12GB of GPU memory if batch size is 128.  
  Thus, you might have to reduce the batch size. (The experiments in the paper were conducted on GEFORCE RTX 3090 with 24GB of gpu memory. 
)

## Citation
In case of using this source code for your research, please cite Park's paper "Relational Knowledge Distillation" and our paper(Not released, stay tuned!).

```
@inproceedings{park2019relational,
  title={Relational Knowledge Distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
