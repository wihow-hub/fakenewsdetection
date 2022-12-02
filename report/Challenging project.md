# Challenging project

2112649 卢海泉            授课教师：陈晨

## 问题描述

随着互联网和社交媒体平台的普及，虚假信息的产生和传播呈爆炸式增长，并以不可控制的速度迅速传播。虚假信息的广泛传播影响着公众舆论，并威胁着社会/政治发展，其对民主、正义和公众信任的损害已经成为一个严重的全球性问题。然而，真伪消息之间也存在着差异，我们可以根据消息的内容、出处等判断真伪。基于此情境，我们可以利用机器学习和深度学习的知识，在训练集上训练一个模型，来对测试集上的消息作出预测。

给定一个中文微信消息的数据集，其中每一条消息包含（title）标题、出处（offical account name）、相关的链接（url）、相关评论（report content）和标签（label：0/1），我们需要根据消息的title、offical account name 、url和report content判断消息的真伪。

## 数据集说明

数据集来源于WeFEND-AAAI20（weak supervision fake news detection via reinforcement learning）：https://github.com/yaqingwang/WeFEND-AAAI20

#### 我们选择了其中的train和 test数据：

![image-20221127172214744](/Users/mac/Library/Application Support/typora-user-images/image-20221127172214744.png)
![image-20221127172350787](/Users/mac/Library/Application Support/typora-user-images/image-20221127172350787.png)

### example：

![image-20221127172743868](/Users/mac/Library/Application Support/typora-user-images/image-20221127172743868.png)

#### 特点：

正负样本不平衡（fake news少）
一条消息中report content可能有多条
url中又很多重复部分

## 方法介绍

### 思路与原理：

在pytorch框架下利用预训练模型bert做分类器进行文本二分类
预训练模型选用了chinese_roberta_wwm_ext_pytorch，权重下载自中文BERT-wwm系列模型，https://github.com/ymcui/Chinese-BERT-wwm。

<img src="/Users/mac/Library/Application Support/typora-user-images/image-20221128225711472.png" alt="image-20221128225711472" style="zoom:50%;" />

![image-20221128225746826](/Users/mac/Library/Application Support/typora-user-images/image-20221128225746826.png)



#### 什么是BERT？

Bidirectional Encoder Representations from Transformer，从名字看Bert是[Transformer](https://blog.csdn.net/qq_36618444/article/details/106472126) Encoder的变种我们在大型文本[语料库](https://so.csdn.net/so/search?q=语料库&spm=1001.2101.3001.7020)上训练通用的“语言理解”模型，之后可以直接借助迁移学习的想法使用已经预训练好的模型参数，并根据自己的实际任务进行fine-tuning，然后将该模型用于我们关心的下游NLP任务，即可以做文本分类.
BERT的基础是transformer，它是多个Transformer的双向叠加，中间每一个蓝色的圈圈都是一个transformer。

### transfomer：

![image-20221128225807871](/Users/mac/Library/Application Support/typora-user-images/image-20221128225807871.png)

#### bert：

![image-20221127204104802](/Users/mac/Library/Application Support/typora-user-images/image-20221127204104802.png)

BERT的训练：通过Masked LM（学习单词的上下文关系）和Next Sentence Prediction（学习预测下一句话）

![image-20221127204600508](/Users/mac/Library/Application Support/typora-user-images/image-20221127204600508.png)

BERT用于文本分类：根据[CLS]的输出来判断句子的种类。在训练自己的数据的时候只需要进行fine-tune即可。**基于BERT的文本分类模型就是在原始的BERT模型后再加上一个分类层即可**。对于基于BERT的文本分类模型来说其输入就是BERT的输入，输出则是每个类别对应的logits值。

### 运用到的框架和库：

主要运用了pytorch和transformer。



## 具体流程和关键代码：

#### 从给定的数据集（train和test的csv文件）中读数据

利用pandas库进行读取

#### 数据预处理

（即将数据处理为模型的输入，bert模型输入为一个序列，只需要构造原始文本对应的Token序列，并在首位分别再加上一个`[CLS]`符和`[SEP]`符作为输入即可。）

###### 1.提取特征：每条消息有五个feature和一个label。基于数据特征，我先将url中重复和多余的部分出去掉，再将这几个feature拼接到一起。

###### 2.讲原始数据样本进行分字（tokenize）处理

###### 3.根据tokenize后的结果构造一个字典：使用BERT预训练时并不需要我们自己来构造这个字典，直接载入谷歌开的`vocab.txt`文件构造字典即可，因为只有`vocab.txt`中每个字的索引顺序才与开源模型中每个字的Embedding向量一一对应的。

###### 4.根据字典将tokenize后的文本序列转换为Token序列，同时在Token序列的首尾分别加上`[CLS]`和`[SEP]`符号，并进行Padding。

###### 5.根据上一步的结果生成padding mask向量。

最终得到第四步和第五步的结果即可输入模型。

##### 主要代码：预训练模型已经封装了大多数操作的函数，所以可以直接调用

```python
#初始化预训练模型和分词器
bert_path = "bert_model"
tokenizer = BertTokenizer.from_pretrained(bert_path)
maxlen=100 #最大序列长度
#encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
#根据参数会短则补齐，长则切断
encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen, padding='max_length', truncation=True)
```

#### 从训练集中按9:1切分为训练集和验证集

验证集可以在一定程度上避免过度拟合，消除一些随机性，训练资料和测试资料是不一样的 have different distribution。
具体做法：在训练集上更新参数，验证集用于调整和选择模型

```python
input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:9000]], input_ids[idxes[9000:10587]], input_ids_test
input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:9000]], input_masks[idxes[9000:10587]], input_masks_test
input_types_train, input_types_valid, input_types_test = input_types[idxes[:9000]], input_types[idxes[9000:10587]], input_types_test
```

#### 利用pytorch的dataloader加载数据

dataloader的作用是可以从数据中一个一个batch地读数据，方便后续模型训练

```python
BATCH_SIZE = 64
# 训练集
train_data = TensorDataset(torch.LongTensor(input_ids_train), 
                           torch.LongTensor(input_masks_train), 
                           torch.LongTensor(input_types_train), 
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)  
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# 验证集
valid_data = TensorDataset(torch.LongTensor(input_ids_valid), 
                          torch.LongTensor(input_masks_valid),
                          torch.LongTensor(input_types_valid), 
                          torch.LongTensor(y_valid))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

# 测试集（是没有标签的）
test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
```

#### 定义和加载bert模型

导入下载的bert的模型超参数，然后在bert模型后加入分类层就行了，可以用一些防止过度拟合的trick

```python
# 定义model
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=2):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)     # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
      
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]   # 池化后的输出 [bs, config.hidden_size]
        logit = self.fc(out_pool)   #  [bs, classes]
        return logit

    def regularization(self, coef):
            item = 0
            for param in self.net.parameters():
                item += torch.norm(param,2)
            res = coef*item
            return res
    def cal_loss(self,pred,target):
         #RMSE+L2 regularization
        loss = torch.sqrt(self.criterion(pred, target)) + self.regularization(0)
        return loss
```

#### 定义优化器（optimizer）和criterion（loss函数）

optimizer：选用AdamW并加上warmup
criterion：选用FocalLoss

```python
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4) #AdamW优化器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),num_training_steps=EPOCHS*len(train_loader))

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
```

#### 训练和验证评估

定义训练函数和验证测试函数，代码略。
训练环境：colab（google的云端开发环境），可免费使用一定gpu资源，加速训练



### 训练过程和结果

<img src="/Users/mac/Library/Application Support/typora-user-images/image-20221128205802091.png" alt="image-20221128205802091"  />

## 可用的trick：（由于计算资源的限制，有一些没有训练出来）

#### 缓解样本不平衡：

在模型层面缓解样本不均匀：加入Focal Loss学习样本

数据层面：尝试了欠采样和过采样。但效果不明显

#### 防止过度拟合：

L1和L2正则化

Dropout

Batch Normalization

Early stopping

### 对抗训练

通过在原始输入上增加对抗扰动，得到对抗样本，再利用对抗样本进行训练，从而提高模型的表现。由于自然语言文本是离散的，一般会把对抗扰动添加到嵌入层上。为了最大化对抗样本的扰动能力，利用梯度上升的方式生成对抗样本。为了避免扰动过大，将梯度做了归一化处理。

pytorch实现：

```python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) 
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```



#### 在分类层的全连接层后加入cnn层，提取更重要的特征，用于分类

pytorch实现：

```python
self.conv1D = torch.nn.Conv1d(in_channels=500, out_channels=500, kernel_size=1)
self.MaxPool1D = torch.nn.MaxPool1d(4, stride=2)
```

#### 什么是cnn：

  将神经网络应用于大图像时，输入可能有上百万个维度，如果输入层和隐含层进行“全连接”，需要训练的参数将会非常多。如果构建一个“部分联通”网络，每个隐含单元仅仅只能连接输入单元的一部分，参数数量会显著下降。卷积神经网络就是基于这个原理而构建的。这其中的思想就是，降维或者说是特征选择，通过前面的卷积层或者池化层将重要的特征选取出来，然后全连接进行分类。以提高分类效果。



## 结果分析

1.结果中，假消息的precision，recall，F1等都较低，原因是数据集中负样本太少，导致数据集不平衡

2.原因还可能是过度拟合了：在training时loss明显小于test时。model自由度很大，在没有traing data的地方，函数会出现很多奇怪的形状。