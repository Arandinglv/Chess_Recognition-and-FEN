# Chess_Recognition-and-FEN
## Directory
**Root Directory**
```
Chess
├─icon
├─images
├─model
```
- **`model` Directory**
`rename.py` is optional, used for renaming the images.

`Data_Aug.py` is optional, used for data augmentation. Referrence: https://blog.csdn.net/qq_38973721/article/details/128700920

`add_Data.py` is optional, user for add images from other folders. 

`dataloader.py` overwrites the `Dataset` class for this dataset. 

`model.py` is the network architecture, employing modules from `modules.py`. This network fine-tuned the ResNet50. The architecture can be seen in the figure 1. 

`train.py` and `valid.py` is used for training the test the model. 

`recognize.py` contains the function of the chessboard division into `8*8` pieces for recognition by caliing the `Board` class from `read_contour.py` to get the square chessboard. Load the training weight and generate the recognition results. 

`Compare.py` is used to generate changes in the state of the chessboard before and after. 
--------------------------------------------------------------
`rename.py` 是可选的，用于重命名图像。

`Data_Aug.py` 是可选的，用于数据增强。参考：https://blog.csdn.net/qq_38973721/article/details/128700920

`add_Data.py` 是可选的，用户用于从其他文件夹添加图像。

`dataloader.py` 覆盖此数据集的 `Dataset` 类。
“model.py”是网络架构，使用“modules.py”中的模块。该网络对 ResNet50 进行了微调。其架构如图1所示。

`train.py` 和 `valid.py` 用于训练和测试模型。

`recognize.py`包含将棋盘分割成`8*8`块进行识别的功能，通过调用`read_contour.py`中的`Board`类得到正方形棋盘。加载训练权重并生成识别结果。

`Compare.py` 用于生成前后棋盘状态的变化。

```
棋盘数组为:[[0, 12, 2, 4, 12, 2, 12, 12], [5, 5, 12, 12, 1, 5, 12, 0], [1, 12, 12, 12, 12, 12, 12, 5], [12, 7, 5, 12, 11, 12, 5, 12], [12, 12, 12, 12, 12, 11, 11, 12], [12, 12, 12, 12, 11, 12, 12, 8], [11, 11, 12, 12, 12, 12, 12, 11], [6, 12, 8, 10, 12, 12, 7, 6]]
具体棋盘为:[['rook', ' ', 'bishop', 'king', ' ', 'bishop', ' ', ' '], ['soldier', 'soldier', ' ', ' ', 'knight', 'soldier', ' ', 'rook'], ['knight', ' ', ' ', ' ', ' ', ' ', ' ', 'soldier'], [' ', 'KNIGHT', 'soldier', ' ', 'SOLDIER', ' ', 'soldier', ' '], [' ', ' ', ' ', ' ', ' ', 'SOLDIER', 'SOLDIER', ' '], [' ', ' ', ' ', ' ', 'SOLDIER', ' ', ' ', 'BISHOP'], ['SOLDIER', 'SOLDIER', ' ', ' ', ' ', ' ', ' ', 'SOLDIER'], ['ROOK', ' ', 'BISHOP', 'KING', ' ', ' ', 'KNIGHT', 'ROOK']]
棋盘数组为:[[0, 12, 12, 4, 12, 2, 12, 12], [5, 5, 12, 12, 1, 5, 12, 0], [1, 12, 12, 12, 12, 12, 12, 5], [12, 7, 5, 12, 11, 12, 5, 12], [12, 12, 12, 12, 12, 11, 2, 12], [12, 12, 12, 12, 11, 12, 12, 8], [11, 11, 12, 12, 12, 12, 12, 11], [6, 12, 8, 10, 12, 12, 7, 6]]
具体棋盘为:[['rook', ' ', ' ', 'king', ' ', 'bishop', ' ', ' '], ['soldier', 'soldier', ' ', ' ', 'knight', 'soldier', ' ', 'rook'], ['knight', ' ', ' ', ' ', ' ', ' ', ' ', 'soldier'], [' ', 'KNIGHT', 'soldier', ' ', 'SOLDIER', ' ', 'soldier', ' '], [' ', ' ', ' ', ' ', ' ', 'SOLDIER', 'bishop', ' '], [' ', ' ', ' ', ' ', 'SOLDIER', ' ', ' ', 'BISHOP'], ['SOLDIER', 'SOLDIER', ' ', ' ', ' ', ' ', ' ', 'SOLDIER'], ['ROOK', ' ', 'BISHOP', 'KING', ' ', ' ', 'KNIGHT', 'ROOK']]
['c8 -> g4']
```
`FEN.py` is **imcomplete**, used for converting the chessboard data into FEN string. 
- **Dataset Directory**
All the images are divided into 13 classes, representing 12 types of pieces and empty status. Images are collected from the `chess.com` screenshots.  

所有图像分为13类，代表12种棋子和空状态。图片是从“chess.com”屏幕截图中收集的。
![resnet](https://github.com/user-attachments/assets/22d811c6-53fb-484d-bfba-0142c07532a2)

`Origin` folder stores the original images.

`Rename` folder stores the renamed images.  

`ImageDataset` folder stores the training image data after Dataset Augmentation: `./model/Data_Aug`.  

`valid_ImageDataset` stores the test image data.  
```
├─ImageDataset
│  ├─00rook black
│  ├─01knight black
│  ├─02bishop black
│  ├─03queen black
│  ├─04king black
│  ├─05sodier black
│  ├─06rook white
│  ├─07knight white
│  ├─08bishop white
│  ├─09queen white
│  ├─10king white
│  ├─11sodier white
│  └─12empty
├─Origin
│  ├─...
├─Rename
│  ├─...
└─valid_ImageDataset
```
All the datasets can be downloaded from:

**BaiduNetDisk**

URL：https://pan.baidu.com/s/19nQvXf9HiQ8flVkJFYEafw?pwd=hg04 

CODE：hg04
