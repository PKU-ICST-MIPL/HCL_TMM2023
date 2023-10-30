# Introduction

This is the source code of our TMM 2023 paper "HCL: Hierarchical Consistency Learning for Webly Supervised Fine-Grained Recognition". Please cite the following paper if you use our code.

Hongbo Sun, Xiangteng He and Yuxin Peng, "HCL: Hierarchical Consistency Learning for Webly Supervised Fine-Grained Recognition", IEEE Transactions on Multimedia (TMM), 2023.




# Dependencies

Python 3.7.15

PyTorch 1.13.0

Torchvision 0.14.0



# Data Preparation

Download the webly supervised fine-grained datasets [web-bird](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz),  [web-car](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz) and [web-aircraft](https://wsnfg-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz), then uncompress them in the ./data folder.



# Usage

Start training by executing the following commands.

- web-bird

  ```
  bash train_cub.sh
  ```

- web-car

  ```
  bash train_car.sh
  ```

- web-aircraft

  ```
  bash train_air.sh
  ```



For any questions, feel free to contact us (sunhongbo@pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.
