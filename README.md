# yolov4

## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip.
I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.
```
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
## Downloading
```
cd ~
git clone https://github.com/ketty0712/yolov4.git
```

### Conda (Recommended)
```
# For GPU
conda env create -f environment.yml
conda activate yolov4
```
### Pip
(Required pip up to date)
```
pip install -r requirements.txt
```
