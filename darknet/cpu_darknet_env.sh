#更新軟件源
sudo apt update
 
# 安裝必要工具
sudo apt install git gcc g++ make -y
 
# 安裝文本編輯器
sudo apt install gedit -y
 
# 安裝圖片查看器
sudo apt install eom -y

# 安裝git
sudo apt install git -y

# 安裝git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

 
# 下載darknet
cd ~
# git clone https://github.com/AlexeyAB/darknet.git
git clone https://github.com/ketty0712/yolo-v4.git
 
# 編譯darknet
cd ~/darknet
make
 
# 下載yolov4-tiny權重文件
# 若下載過程中,網絡較差,可用其他方法下載
# wget -c https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
 
# #運行yolov4-tiny進行圖片測試
# ./darknet detector test cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights data/dog.jpg
 
# #查看檢測圖片結果
# eom predictions.jpg
 
# 安裝opencv
sudo apt install libopencv-dev -y
 
# 編輯Makefile文件
# 修改OPENCV=1，OPENMP=1
gedit Makefile
 
#重新編譯
make
 
#再次運行yolov3進行圖片測試
./darknet detector test cfg/deron.data cfg/yolov4-tiny-custom.cfg yolov4-tiny-custom_2000.weights data/QRcode/image25.jpg

