# 解压缩文件
分part的压缩文件，在linux系统直接unzip可能会出错。
通过以下的方式即将Part.zip Part.z01等等打包为out.zip再解压缩
```bash
zip -FF Part.zip --out out.zip
unzip out.zip -d dst/
```

# 测试socks5和http

```bash
curl --socks5 ip:port http://www.baidu.com
curl --socks5 key:pwd@ip:port https://www.baidu.com

curl --connect-timeout 2 -x ip:port http://www.baidu.com
curl -x key:pwd@ip:port https:/www.baidu.com
```



# windows端口转发

> 增加、删除规则等需要以管理员权限执行！



```bash
# 增加转发规则，将0.0.0.0:80 -> 172.20.73.40:80
netsh interface portproxy add v4tov4 listenport=80 listenaddress=0.0.0.0 connectport=80 connectaddress=172.20.73.40
# example
netsh interface portproxy add v4tov4 listenport=88 listenaddress=0.0.0.0 connectport=80 connectaddress=172.20.73.40
# 查看已转发端口
netsh interface portproxy show all
# 删除单条规则
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=88
# 删除所有规则
netsh interface portproxy reset
```

# pip

## 	指定源

```bash
pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 代理

``` bash
pip install -r requirements.txt --proxy=10.106.14.29:20811
```



# npm

​	安装特定版本的node

```bash
npm install -g n
#n <version>
n latest
```





# wsl

## 	安装CUDA

 1. 只需要给windows装驱动，之后wsl2会自动链接驱动，也可以正常调用nvidia-smi

 2. 安装toolkits，按照nvidia官网

    ![image-20230417103208891](https://raw.githubusercontent.com/Annzstbl/image-host/main/img/image-20230417103208891.png)

​	3.报错 /sbin/ldconfig.real: /usr/lib/wsl/lib/libcuda.so.1 is not a symbolic link

​	原因：wsl2是复用了windows的驱动，实际地址为C:\Windows\System32\lxss\lib，在此地址下libcuda.so.1确实不是软连接，而是复制。

​	可能的修改方案：

1. disable automount in /etc/wsl.conf

> [automount]
> ldconfig = false

1. copy /usr/lib/wsl/lib to /usr/lib/wsl2/lib (in wsl, writable)
2. edit /etc/ld.so.conf.d/ld.wsl.conf

> /usr/lib/wsl/lib --> **/usr/lib/wsl2/lib** (new location)

1. rm /usr/lib/wsl2/lib/libcuda.so.* and sudo ldconfig

works for CUDA in WSL, but "Segmentation fault" in DirectML

## 关机

```bash
wsl --shutdown
```

# docker

## docker基本操作

创建某个镜像的容器

```bash
docker run [-it] some-image
```

列出当前运行/所有(-a)的容器

```bash
docker ps [-a]
```

删除某个容器

```bash
docker rm container-id
```

容器的启动，-i表示交互模式

```bash
docker start [-i] container-id
```

容器的停止，重启

```bash
docker stop container-id
docker restart container-id
```



## Nvidia 镜像启动

```bash
docker run -it --name="pytracking" --gpus all --ipc=host -v /mnt/z/publicData/:/publicData -v /home/lth/dockerWorkspace/:/workspace -d pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
```





# GIT

## 代理git

查看全局配置

```bash
git config --global --list
```

设置全局代理（https + http）

```bash
git config --global http.proxy "http://127.0.0.1:10809"
git config --global https.proxy "http://127.0.0.1:10809"
```

设置全局代理（socks5）

```bash
git config --global http.https://github.com.proxy socks5://127.0.0.1:10808
git config --global https.https://github.com.proxy socks5://127.0.0.1:10808
```

取消代理

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global --unset http.https://github.com.proxy
git config --global --unset https.https://github.com.proxy
```



局域网代理

```
git config --global http.proxy "http://10.106.14.29:20811"
git config --global https.proxy "http://10.106.14.29:20811"
```





## 配置

```bash
 git config --global user.email "tianhaoli1996@gmail.com"
 git config --global user.name "LTH"
```



## 操作

```bash
cd ${PROJECT_FOLDER}

#初始化
git init

#添加文件
git add .

#提交
git commit -m "注释"

#创建远程仓库 在 gitee上 
git remote add origin https://gitee.com/......

#推送
git push -u origin main/master


```



## 常用代码

```bash
# 显示所有远程仓库
git remote -v
# 添加远程版本库
git remote add [name] [url]
# 推送
git push -u [origin] [local]
# 删除远程
git remote rm [name]
# 修改仓库名
git remote rename [old-name] [new-name]

# 远程仓库下载新分支与数据
git fetch [origin]

# 本地所有分支
git branch -a
```

# Git Copilit

## 代理支持

通过VSCode的代理是支持的，但是只支持http代理，不支持https代理。

所以在VsCode中报如下错误时：

```tunneling socket could not be established, cause=Client network socket disconnected before secure TLS connection was established```

需要将proxy中https更改为http

# conda

## conda环境

```bash
conda create -n name python=3.7
conda remove -n name --all

conda search pkg_name -c channel

conda info --envs
```

## torch安装



​	https://pytorch.org/get-started/previous-versions/

```bash
#197常用
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

```



## conda换源

清华镜像 https://mirrors.tuna.tsinghua.edu.cn/

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```



```bash
# 添加指定源
conda config --add channels *（*指代你要添加的源）
# 设置安装包时，显示镜像来源，建议显示
conda config --set show_channel_urls yes 
# 删除指定源
conda config --remove channels *(*代表你要删除的源）
# 显示源
cat ~/.condarc
```



换回源

```bash
conda config --remove-key channels
```



清除缓存

```bash
# 删除没有用的包
conda clean -p
# 删除tar打包
conda clean -t
# 删除无用的包和缓存
conda clean --all
```



## conda代理

```bash
conda config --set proxy_servers.http http://10.106.14.29:20811
conda config --set proxy_servers.https https://10.106.14.29:20811
```







# apt

```bash
sudo cp /etc/apt/sources.list /etc/apt/sourcse.list.bak
```

```bash
sudo vim /etc/apt/sources.list
```

添加

```bash
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
```

```bash
sudo apt-get update
```

## libGL缺失

```bash
apt-get update && apt-get install libgl1
```



# tensorboard启动命令

```bash
tensorboard --logdir=./ --bind_all
```





# PASCAL VOC格式标注

VOC数据集的坐标起始点是1，而不是0，所以提取xml里的坐标数据时需要-1，写进xml文件时坐标需要+1

xmin = int(bbox.find('xmin').text) - 1



# 常见bug

```bash
#AttributeError: module 'setuptools._distutils' has no attribute 'version'
# solution
pip install setuptools==59.5.0

```





# Segment Anything

## wsl2上 安装

```bash
conda create -n segment_anything python=3.9
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .

pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

安装成功后，可以参考调用$scripts/amg.py$完成分割

## web demo

1. 安装npm和yarn

2. yarn翻墙代理

```bash
hostip=$(cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }')
port=10811

yarn config set https-proxy ${PROXY_HTTP}
yarn config set proxy ${PROXY_HTTP}
```

3. 导出onnx

   调用$scripts/export\_onnx\_model.py$

```bash
python scripts/export_onnx_model.py --checkpoint ../checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --output test/sam_onnx_h.onnx --opset 13
```

> --opset参数，表示onnx version，参考python中onnx版本调整。

​	调用时根据torch版本，可能会出现错误 export of `repeat_interleave` fails: 'torch._C.Value' object is not iterable.，可以参考[github bug修复](https://github.com/pytorch/pytorch/pull/73760)对源码进行修改。

4. 导出embeddings

```python
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
checkpoint = "../checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

image = cv2.imread('imgs/000185.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("imgs/000185.npy", image_embedding)

```

5. 修改网页配置

   $demo/src/App.tsx$

```python
// Define image, embedding and model paths
const IMAGE_PATH = "/assets/data/000185.jpg";
const IMAGE_EMBEDDING = "/assets/data/000185.npy";
const MODEL_DIR = "/model/sam_onnx_h.onnx";
```

> 其中/assets指目录demo/src/assets
>
> /model指目录demo/model

6. yarn start

```bash
yarn && yarn start
```

7. 访问8081端口，可以得到静态图像的demo





# UntiUav

### pip安装报bmt_clipit>=1.0 错误

在安装cv时候，采用如下命令指定url安装

```bash
pip install -r requirements/cv.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

安装后对torch和torchvision版本进行确认，必要时重新安装

```
pip install torch==1.8.1+cu102  torchvision==0.9.1+cu102 torchaudio==0.8.1  --extra-index-url https://download.pytorch.org/whl/cu102
```

### track1

直接跑通。

结果：

![image-20230211143432127](C:\Users\lth\AppData\Roaming\Typora\typora-user-images\image-20230211143432127.png)

### track2

报错

![image-20230211143543122](https://raw.githubusercontent.com/Annzstbl/image-host/main/img/image-20230211143543122.png)



![image-20230211221842465](https://raw.githubusercontent.com/Annzstbl/image-host/main/img/image-20230211221842465.png)



![image-20230218142511500](https://raw.githubusercontent.com/Annzstbl/image-host/main/img/image-20230218142511500.png)



CoCo数据集

​	[ref 知乎](https://zhuanlan.zhihu.com/p/29393415)

​    3种标注类型(JSON)： object instance object keypoints image captions

​	3中json里共享的字段：info, image, license

​	不共享的字段: annotation

### 共享字段定义：

```json
"info":{
    "description":"",
    "url":"",
    "version":"",
    "year":"",
    "contributor":"",
    “data_created:""
}
```

```json
{
	"license":3,
	"file_name":"COCO_val2014_000000391895.jpg",
	"coco_url":"http:\/\/mscoco.org\/images\/391895",
	"height":360,"width":640,"date_captured":"2013-11-14 11:18:45",
	"flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
	"id":391895
},
```

```json
{
	"url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
	"id":1,
	"name":"Attribution-NonCommercial-ShareAlike License"
},
```

### object instance

```json
{
    "info": info,
    "licenses": [license],
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}
```

images数组元素的数量  = 训练（测试）集图片数量

annotations数组数量 = 训练（测试）集bounding box数量

categories数组元素数量 = 分类数量

#### annotations:

```json
annotation{
    "id": int,    
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
```

image_id：这个标注是哪个图像

iscrowd: 表示segmentation的格式：0表示polygons格式，1表示RLE格式

#### categories

```json
{
    "id": int,
    "name": str,
    "supercategory": str,
}
```





# GlobalTrack

## $\surd$ mmdetection-最新版 2023年-3月

安装mmdetection最新版成功，但是不适合GlobalTrack

```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

#按照pytorch官网安装pytorch
conda install python=3.7 pytorch==1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y

pip install openmim
mim install mmdet
```



## $\times$ mmdetection-v1.0.0rc0 

编译失败

```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install cython
pip install pycocotools
python setup.py develop
```



## $\surd$安装

```bash
conda create -n global_track python=3.7.3 -y
conda activate global_track
# 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# 11.1：
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```



需要安装的库写入requirments.txt：

```
imageio
mmcv==0.4.3
numpy
opencv-python
opencv-python-headless
Pillow
scikit-image
scikit-learn
scipy
Shapely
matplotlib
pycocotools
terminaltables
cpython
cython
tensorboard
```



执行命令 

```bash
pip install -r requirments.txt

#编译
cd _submodules/mmdetection
python setup.py develop
```



__编译会遇到错误，需要把所有源代码中的 *AT_CHECK* 换成 *TORCH_CHECK*__



当编译失败需要clean的时候

```bash
cd _submodules/mmdetection
rm -rf build .egs mmdet.egg-info
```



__Runtime Error__

第二个错误修改

```python
#from mmcv.runner.utils import get_dist_info
from mmcv.runner import get_dist_info
```



### 软链接

```bash
cd ~
mkdir data
cd data
ln -s /data3/publicData/Datasets/COCO ./coco
ln -s /data3/publicData/Datasets/GOT-10k ./GOT-10k
ln -s /data3/publicData/Datasets/LaSOT ./LaSOTBenchmark
ln -s /data3/publicData/Datasets/OTB/OTB2015 ./OTB
cd ~/global_track
ln -s /data3/litianhao/checkpoints/globalTrack ./checkpoints
ln -s /data3/litianhao/workdir/globalTrack ./work_dirs

```



### 训练

```bash
PYTHONPATH=. python tools/train_qg_rcnn.py --config configs/qg_rcnn_r50_fpn.py --load_from checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth --gpus 4
```



### 训练部分代码

在训练开始前，会注册一系列的hook

```python
runner.register_training_hooks(cfg.lr_config, optimizer_config,
                               cfg.checkpoint_config, cfg.log_config)
```

在训练的过程中，每epoch前后，每iter前后，都会调用所有hook的相对应函数

```python
def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch') # 调用所有hook的before_train_epoch函数
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
```

#### lr_hook

​	在每个epoch、iter前更新lr

#### optimizer_hook

​	在每一个iter后，进行梯度回传、梯度剪裁，最后更新参数

```python
@HOOKS.register_module
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()

```

#### checkpoint_hook

​	每个epoch后存checkpoint



log_hook



# SLT-NET

## 1 安装

安装环境：

```bash
conda create -n slt_net python=3.8
conda activate slt_net

pip install pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install timm==0.6.12

pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmsegmentation==0.24.1

cd ./lib/ref_video/PNS
python setup.py build develop
```



## 2 报错

### 2.1 PyramidVisionTransformerV2不接受pretrained_cfg参数

修改 lib/pvt_v2.py 385行附近

```python
@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    # remove pretrained_cfg in kwargs
    if('pretrained_cfg' in kwargs.keys()):
        kwargs.pop('pretrained_cfg')
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
```

# S2ADet

## 安装

```bash
conda create -n s2adet python=3.8
conda activate s2adet
pip install pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

#降级
pip install setuptools==58.0.4
```

## HOD3K数据集

- [x] HOD3K原始数据 [百度云](https://pan.baidu.com/share/init?surl=mtXDJfU6M8F60GZinLam-w) gvbe
- [x] HOD3K处理后数据 [百度云](https://pan.baidu.com/s/1ga-YqLqTqVxTbnHHjch82g) qugy

数据是使用[hsitracking](https://www.hsitracking.com)处理的。

-  原始数据中，共有原始图像2308张，无法可视化
- 处理后数据中包含两个文件夹，一个是HSI，包含test和val，无法可视化。test中包含715张图像，val中包含219张图像。
- 另一个文件夹是hsidetection，里边又包含sa_information和se_information，均可可视化
- sa_information中test715张，train2308张，val219张
- se_information中test715张，train2308张，val219张，与sa_information一样
- 715+2308+219总计3242张。
- 总结：
  1. 原始图像train  ./raw_hsi_train/
  2. 原始图像test ./hsi_dataset/HSI/test
  3. 原始图像val ./hsi_dataset/HSI/val
  4. 处理后图像 ./hsi_dataset/hsidetection/sa_information/images
  5. 处理后图像 ./hsi_dataset/hsidetection/se_information/images



#### 调试进程

对test的第一张图 0001.png进行调试。

1. 图像的3,9,4似乎是同一波段
2. ONR的波段选择结果是：六波段情况1 2 4 6 12 15，三波段情况 2 9 15
3. TRC-OC-FDPC结果：1 2 4 5 9 10
4. NC-OC-MVPCA结果: 1 2 3 5 9 13
5. NC-OC-IE结果: 1 2 3 6 9 13
6. 图像的 6 15/13 2波段合并，与作者给出的较为一致，且能符合ONR或NC-OC-IE的结果
7. 不知道怎么处理的能得到sa图像

## HSI-1数据集 [github](https://github.com/yanlongbinluck/HSI-Object-Detection-NPU)

- [x] HSI-1数据集 [百度云](https://pan.baidu.com/s/1ga-YqLqTqVxTbnHHjch82g) 6shr

​		



## 训练

1. HOD3K训练，需要删除labels下的cache
2. 报错numpy has no attribute 'int'

```python
	np.int32#error
    int#replace
```

3. summary.py文件中报错，删除np.int32

```python
    cum_counts = np.cumsum(np.greater(counts, 0, np.int32))#error
    cum_counts = np.cumsum(np.greater(counts, 0))
```

4. 



# DiffusionDet

```
git clone https://github.com/ShoufaChen/DIffusionDet

conda create -n diffusionDet python=3.8
conda activate diffusion

#197常用
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install opencv-python

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

pip install timm

```

# MMDet & MMYOLO

```bash

mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

```bash
conda create -n openmmlab python=3.8 -y && conda activate openmmlab
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
#安装mmcv
git clone sth about mmcv
#mim install "mmcv>=2.0.0,<2.1.0"
cd mmcv
pip install -v -e .

cd mmdetection
pip install -v -e .

#软连接
ln -s /data3/litianhao/workdir/mmdet ./work_dirs
ln -s /data3/litianhao/checkpoints/mmdet ./checkpoints

#测试
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```



## Todo List

- [ ] 首层不训练问题：DiffDet训练中，ResNet的frozen_stages=1，导致首层不训练。文件地址：mmdet/models/backbones/resnet.py，行数613。
- [ ] 

