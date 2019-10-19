# Text-Classification
文本情感分类（TextRNN/TextCNN/TextRCNN/Highway/Attention）

数据集：https://github.com/SophonPlus/ChineseNlpCorpus (中文命名实体识别数据集，下载后放在 data 文件夹中)

所用字向量：https://github.com/SophonPlus/ChineseWordVectors (sjl_weixin词向量，解压后放在 data 文件夹中)

## Train and Test
快速训练 + 交叉验证 + 测试，具体参数设定见main.py文件
````
python main.py -model_name TextRNN or TextCNN or TextRCNN or TextCNN_withHighway or TextRNN_Attention \
                -do_train True \
                -do_cv True \
                -do_test True
````

## Results
验证集分数为5折交叉验证的F1分数平均值，测试集为5折最优模型的投票结果，未仔细调参，最终结果仅共参考

| Model_name  | Dev F1 | Test F1 |
| ------------- | ---- | ---- | 
| TextRNN | 0.8654 | 0.8469 |
| TextCNN | 0.8738  | 0.8563 |
| TextRCNN | 0.8710  | 0.8573 |
| TextCNN_withHighway | 0.8717  | 0.8620 |
| TextRNN_Attention | 0.8672  | 0.8540 |

## Ref
https://github.com/brightmart/text_classification
https://github.com/songyingxin/TextClassification-Pytorch
https://github.com/649453932/Chinese-Text-Classification-Pytorch
