#!/bin/bash
# auther:lishan
# time:2020/6/19
# funcation:搭建yolo訓練環境gpu版本,安裝darknet + opencv + cuda + cudnn
# instruction:此腳本僅供參考,可自行修改命令,特別是下載速度過慢時,可註釋掉wget開頭的命令(#爲註釋符),
#            並用其它方式將對應文件放到darknet目錄下
# attention:腳本測試時使用操作系統爲ubuntu18.04 LTS 桌面版64位,gpu爲GTX 1080ti
 
# darknet官網:https://pjreddie.com/darknet/yolo/
# opencv官網:https://opencv.org/
# nvidia官網:https://www.nvidia.cn/
# 運行腳本之前先確認顯卡驅動已經安裝成功，輸入nvidia-smi，查看是否有輸出
 
# --------------------------------------------------------------------
################## part one: darkent ######################
#更新軟件源
sudo apt update
  
#安裝必要工具
sudo apt install git gcc g++ make -y
 
#安裝文本編輯器
sudo apt install gedit -y
 
#安裝圖片查看器
sudo apt install eom -y
 
#安裝視頻播放器
sudo apt install vlc -y

# 安裝git
sudo apt install git -y

# 安裝git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

#下載darknet
cd ~
git clone https://github.com/ketty0712/yolov4.git
   
#--------------------------------------------------------------------
################## part two: opencv ######################
#安裝opencv
sudo apt install libopencv-dev -y
 
# --------------------------------------------------------------------
# ################# part three: cuda ######################
# instruction: 本機顯卡爲GTX 1080ti,下載cuda版本爲10.0
# 在安裝cuda之前,請自行安裝顯卡驅動,安裝方法請參考其他教程
# note:腳本默認已經安裝好顯卡驅動

# --------------------------------------------------------------------
# 方法1：自動安裝(目前安裝的最新版本爲9.1,所以需要自行下載cudnn7.6)
# cuda與cudnn版本需要一一對應
# cudnn 7.6.5 <--> cuda9.0、cuda9.2、cuda10.0、cuda10.1、cuda10.2
# cudnn 8.0.0 <--> cuda10.2、cuda11.0

#自動安裝
sudo apt install nvidia-cuda-toolkit -y

#默認安裝位置是/usr/lib/cuda/，需要移動到/usr/local/
sudo mv /usr/lib/cuda/ /usr/local/

#-------------------------------------------------------------------- 
# 方法2：官網下載安裝
# cuda官網下載：https://developer.nvidia.com/cuda-toolkit-archive
# 打開官網，根據自己的顯卡型號找到需要的版本鏈接，替換掉wget -c 後的鏈接地址
# 或者提前下載好文件之後放在主目錄下(~/)，並註釋wget命令

# cd ~
# 下載cuda 10.0.130_410.48_linux
# wget -c https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
 
# 下載cuda 9.0.176_384.81_linux
# wget -c https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
 
# 運行cuda安裝文件
# sudo sh cuda*linux.run
# 第一次選擇n，後面選擇y
 
#--------------------------------------------------------------------
# 添加環境變數
echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
 
# 更新環境變量
source ~/.bashrc
 
# --------------------------------------------------------------------
# ################# part four: cudnn ######################
# cuda與cudnn版本需要一一對應
# cudnn 7.6.5 <--> cuda9.0、cuda9.2、cuda10.0、cuda10.1、cuda10.2
# cudnn 8.0.0 <--> cuda10.2、cuda11.0
 
# 下載cudnn 7.6.5
# 請通過官網下載  https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz
# 並移動到主目錄(~/)下
 
# 到主目錄下解壓文件
cd ~
tar -xzvf cudnn*.tgz
 
# 複製cudnn文件到上一步安裝的cuda中 
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64

# 賦予可讀寫權限
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
 
# --------------------------------------------------------------------
# 編譯darknet
cd ~/yolov4/darknet
 
# 編輯Makefile文件
# GPU=1,CUDNN=1,OPENCV=1,OPENMP=1, LIBSO=1
gedit Makefile
 
# 編譯
make
 
# 運行yolov4進行圖片測試
./darknet detector test cfg/deron.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_2000.weights data/QRcode/image25.jpg