Lesson3 Homework



# 基础作业 - 完成下面两个作业


## 1. 在[茴香豆 Web 版](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)中创建自己领域的知识问答助手


- 参考视频[零编程玩转大模型，学习茴香豆部署群聊助手](https://www.bilibili.com/video/BV1S2421N7mn)
	- [x] 已完成
- 完成不少于 400 字的笔记 + 线上茴香豆助手对话截图(不少于5轮)
### 笔记

1. 什么是RAG
	RAG（Retrieval Augmented Generation）技术，通过检索与用户输入相关的信息片段，并结合**外部知识库**来生成更准确、更丰富的回答。解决 LLMs 在处理知识密集型任务时可能遇到的挑战, 如幻觉、知识过时和缺乏透明、可追溯的推理过程等。提供更准确的回答、降低推理成本、实现外部记忆。
	
	RAG 能够让基础模型**实现非参数知识更新**，无需训练就可以掌握新领域的知识。通过使用 RAG 技术，基于自己的文档，可以快速、高效的搭建自己的知识领域助手。

2. 什么是茴香豆

	茴香豆是一个基于 LLM 的**群聊**知识助手，优势：

  1. 设计拒答、响应两阶段 pipeline 应对群聊场景，解答问题同时不会消息泛滥。详情见[技术报告](https://arxiv.org/abs/2401.08772)
  2. 成本低至 1.5G 显存，无需训练适用各行业
  3. 提供一整套前后端 web、android、算法源码，工业级开源可商用的代码

4. RAG 的优势
	RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术，它在处理技术问题和其他需要广泛知识的任务中具有显著优势。以下是RAG的一些主要优势：

- **增强的知识库访问**：RAG通过检索相关信息来增强语言模型的知识库，使其能够访问和利用比模型自身训练数据更广泛、更更新的信息源。
- **提高准确性和相关性**：通过检索相关信息，RAG能够生成更准确、更相关的回答，特别是在处理特定领域的问题时，这一点尤为重要。
- **动态内容更新**：与传统的语言模型相比，RAG可以动态地利用最新的数据和信息，因为它在生成回答时可以检索和整合最新的相关内容。
- **长文本理解和生成**：RAG特别适合处理长文本的理解和生成任务，因为它可以有效地结合来自不同来源的大量信息来构建连贯和详细的回答。
- **灵活性和可扩展性**：RAG可以根据不同任务的需求进行调整和扩展，例如，可以通过更换检索源或调整生成策略来适应不同的应用场景。
- **减少知识偏差**：由于RAG可以访问最新的信息，它有助于减少因模型训练数据过时或不全面而导致的知识偏差。

在HuiXiangDou技术助手中，RAG被用于响应管道，用于寻找真正问题的答案。这种方法有助于提高助手在回答技术相关问题时的准确性和可靠性，尤其是在处理需要特定领域知识的复杂问题时。通过结合检索到的信息和语言模型的生成能力，HuiXiangDou能够提供更全面和精确的技术帮助。

### 对话截图

1. Round1

![对话截图第一轮](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-homework-Dialog01.png)


2. Round2

![对话截图第二轮](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-homework-Dialog02.png)

3. Round3

![对话截图第三轮](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-homework-Dialog03.png)

4. Round4

![对话截图第四轮](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-homework-Dialog04.png)

5. Round5

![对话截图第五轮](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-homework-Dialog05.png)



- （可选）参考 [代码](https://github.com/InternLM/HuixiangDou/tree/main/web) 在自己的服务器部署茴香豆 Web 版
 - [x] 暂时不选

## 2.在 `InternLM Studio` 上部署茴香豆技术助手


- 根据教程文档搭建 `茴香豆技术助手`，针对问题"茴香豆怎么部署到微信群？"进行提问	

- 完成不少于 400 字的笔记 + 截图


### 笔记

#### 一、 环境配置

##### 1.1 配置基础环境

1. 首先打开 `Intern Studio` 界面，点击 _**创建开发机**_ 配置开发机系统。
2. 在开发机系统配置中填写 `开发机名称` 后，点击 选择镜像 使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `30% A100 * 1` 的选项，然后立即创建开发机器。
3. 开发机创建完成后，在开发机条目右侧的操作选项中，点击 `进入开发机` 选项。
4. 进入开发机后，从官方环境复制运行 InternLM 的基础环境，命名为 `InternLM2_Huixiangdou`，在命令行模式下运行：
```shell
studio-conda -o internlm-base -t InternLM2_Huixiangdou
```
5. 复制完成后，在本地查看环境。

```shell
conda env list

```

结果如下所示。

```shell
# conda environments:
#
base                  *  /root/.conda
InternLM2_Huixiangdou                 /root/.conda/envs/InternLM2_Huixiangdou
```

6. 运行 _**conda**_ 命令，激活 `InternLM2_Huixiangdou` _**python**_ 虚拟环境:

```shell
conda activate InternLM2_Huixiangdou
```

环境激活后，命令行左边会显示当前（也就是 `InternLM2_Huixiangdou`）的环境名称

##### 1.2 下载基础文件


复制茴香豆所需模型文件，为了减少下载和避免 **HuggingFace** 登录问题，所有作业和教程涉及的模型都已经存放在 `Intern Studio` 开发机共享文件中。本教程选用 **InternLM2-Chat-7B** 作为基础模型。

```shell
# 创建模型文件夹
cd /root && mkdir models

# 复制BCE模型
ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1

# 复制大模型参数（下面的模型，根据作业进度和任务进行**选择一个**就行）
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b

```

##### 1.3 下载安装茴香豆


安装茴香豆运行所需依赖。

```shell
# 安装 python 依赖
# pip install -r requirements.txt

pip install protobuf==4.25.3 accelerate==0.28.0 aiohttp==3.9.3 auto-gptq==0.7.1 bcembedding==0.1.3 beautifulsoup4==4.8.2 einops==0.7.0 faiss-gpu==1.7.2 langchain==0.1.14 loguru==0.7.2 lxml_html_clean==0.1.0 openai==1.16.1 openpyxl==3.1.2 pandas==2.2.1 pydantic==2.6.4 pymupdf==1.24.1 python-docx==1.1.0 pytoml==0.1.21 readability-lxml==0.8.1 redis==5.0.3 requests==2.31.0 scikit-learn==1.4.1.post1 sentence_transformers==2.2.2 textract==1.6.5 tiktoken==0.6.0 transformers==4.39.3 transformers_stream_generator==0.0.5 unstructured==0.11.2

## 因为 Intern Studio 不支持对系统文件的永久修改，在 Intern Studio 安装部署的同学不建议安装 Word 依赖，后续的操作和作业不会涉及 Word 解析。
## 想要自己尝试解析 Word 文件的同学，uncomment 掉下面这行，安装解析 .doc .docx 必需的依赖
# apt update && apt -y install python-dev python libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev

```

从茴香豆官方仓库下载茴香豆。

```shell
cd /root
# 下载 repo
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout 447c6f7e68a1657fce1c4f7c740ea1700bde0440

```

茴香豆工具在 `Intern Studio` 开发机的安装工作结束。如果部署在自己的服务器上，参考上节课模型下载内容或本节 [3.4 配置文件解析](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#34-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90) 部分内容下载模型文件。

#### 二、 使用茴香豆搭建 RAG 助手

##### 2.1 修改配置文件


用已下载模型的路径替换 `/root/huixiangdou/config.ini` 文件中的默认模型，需要修改 3 处模型地址，分别是:

命令行输入下面的命令，修改用于向量数据库和词嵌入的模型

```shell
sed -i '6s#.*#embedding_model_path = "/root/models/bce-embedding-base_v1"#' /root/huixiangdou/config.ini

```

用于检索的重排序模型

```shell
sed -i '7s#.*#reranker_model_path = "/root/models/bce-reranker-base_v1"#' /root/huixiangdou/config.ini
```

和本次选用的大模型

```shell
sed -i '29s#.*#local_llm_path = "/root/models/internlm2-chat-7b"#' /root/huixiangdou/config.ini
```


配置文件具体含义和更多细节参考 [3.4 配置文件解析](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md#34-%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90)。

##### 2.2 创建知识库


本示例中，使用 **InternLM** 的 **Huixiangdou** 文档作为新增知识数据检索来源，在不重新训练的情况下，打造一个 **Huixiangdou** 技术问答助手。

首先，下载 **Huixiangdou** 语料：

```shell
cd /root/huixiangdou && mkdir repodir

git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou

```

提取知识库特征，创建向量数据库。数据库向量化的过程应用到了 **LangChain** 的相关模块，默认嵌入和重排序模型调用的网易 **BCE 双语模型**，如果没有在 `config.ini` 文件中指定本地模型路径，茴香豆将自动从 **HuggingFace** 拉取默认模型。

除了语料知识的向量数据库，茴香豆建立接受和拒答两个向量数据库，用来在检索的过程中更加精确的判断提问的相关性，这两个数据库的来源分别是：

- 接受问题列表，希望茴香豆助手回答的示例问题
    - 存储在 `huixiangdou/resource/good_questions.json` 中
- 拒绝问题列表，希望茴香豆助手拒答的示例问题
    - 存储在 `huixiangdou/resource/bad_questions.json` 中
    - 其中多为技术无关的主题或闲聊
    - 如："nihui 是谁", "具体在哪些位置进行修改？", "你是谁？", "1+1"

运行下面的命令，增加茴香豆相关的问题到接受问题示例中：

```shell
cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json

echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json

```

再创建一个测试用的问询列表，用来测试拒答流程是否起效：

```shell
cd /root/huixiangdou

echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json

```

在确定好语料来源后，运行下面的命令，创建 RAG 检索过程中使用的向量数据库：

```shell
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json

```

向量数据库的创建需要等待一小段时间，过程约占用 1.6G 显存。

完成后，**Huixiangdou** 相关的新增知识就以向量数据库的形式存储在 `workdir` 文件夹下。

检索过程中，茴香豆会将输入问题与两个列表中的问题在向量空间进行相似性比较，判断该问题是否应该回答，避免群聊过程中的问答泛滥。确定的回答的问题会利用基础模型提取关键词，在知识库中检索 `top K` 相似的 `chunk`，综合问题和检索到的 `chunk` 生成答案。

##### 2.3 运行茴香豆知识助手

我们已经提取了知识库特征，并创建了对应的向量数据库。现在，让我们来测试一下效果：

命令行运行：

```shell
# 填入问题
sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py

# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone

```

RAG 技术的优势就是非参数化的模型调优，这里使用的仍然是基础模型 `InternLM2-Chat-7B`， 没有任何额外数据的训练。面对同样的问题，我们的**茴香豆技术助理**能够根据我们提供的数据库生成准确的答案：

### 截图


![搭建茴香豆技术助手截图](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-RAG01-%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83.png)

![茴香豆怎么部署到微信群](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson3-RAG02.png)


