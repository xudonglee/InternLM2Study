

##  **部署 `InternLM2-Chat-1.8B` 模型进行智能对话**

在课程提供的算力平台进行部署

https://studio.intern-ai.org.cn/console/instance/

```
---------------------------------- 欢迎使用 InternStudio 开发机 ------------------------------

+--------+------------+----------------------------------------------------------------------+
|  目录  |    名称    |                              简介                                    |
+--------+------------+----------------------------------------------------------------------+
|   /    |  系统目录  | 每次停止开发机会将其恢复至系统（镜像）初始状态。不建议存储数据。     |
+--------+------------+----------------------------------------------------------------------+
| /root  | 用户家目录 | 您的所有开发机共享此目录，不受开发机的启停影响。强烈建议将 conda     |
|        |            | 环境、代码仓库等所有数据存储在此目录下。                             |
|        |            | 【注意】该目录有存储限额，超过限额后新写入的数据会被静默删除！       |
+--------+------------+----------------------------------------------------------------------+
| /share |  共享目录  | 常用微调数据集、模型仓库、教程、xtuner 配置文件都存放在此。          |
+--------+------------+----------------------------------------------------------------------+
Tips:

1. 快速从本地上传文件:
   scp -o StrictHostKeyChecking=no -r -P {端口} {本地目录} root@ssh.intern-ai.org.cn:{开发机目录}
   *注：在开发机 SSH 连接功能查看端口号 

2. 避免因终端关闭或 SSH 连接断开导致任务终止, 强烈建议使用 tmux 将实验进程与终端窗口分离：
   https://www.ruanyifeng.com/blog/2019/10/tmux.html

3. 查看 GPU 显存和算力使用率: studio-smi 

4. 使用InternStudio开箱即用的conda环境:
   studio-conda -h

5. 将conda环境一键添加到jupyterlab:
   lab add {YOUR_CONDA_ENV_NAME}
```


### **1 配置基础环境**

1. 打开 `Intern Studio` 界面，点击 **创建开发机** 配置开发机系统。
2. 填写 `开发机名称` 后，点击 选择镜像 使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `10% A100 * 1` 的选项，然后立即创建开发机器。
3. 创建完成后，点击 `进入开发机` 选项。
4. **进入开发机后，在 `terminal` 中输入环境配置命令 (配置环境时间较长，需耐心等待)：**

```shell
studio-conda -o internlm-base -t demo
# 与 studio-conda 等效的配置方案
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

5. 配置完成后，进入到新创建的 `conda` 环境之中：

```shell
conda activate demo
```

6. 输入以下命令，完成环境包的安装：

```shell
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```



### 2 下载 `InternLM2-Chat-1.8B` 模型


1. 按路径创建文件夹，并进入到对应文件目录中：

```shell
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```

2. 编辑`/root/demo/download_mini.py` 文件，复制以下代码：

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```

3. 执行命令，下载模型参数文件：

```shell
python /root/demo/download_mini.py
```


### 3 运行 cli_demo

1. 编辑`/root/demo/cli_demo.py` 文件，复制以下代码：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```

2. 输入命令，执行 Demo 程序：

```shell

conda activate demo
python /root/demo/cli_demo.py
```

3. 等待模型加载完成，键入内容示例：

```shell

(demo) root@intern-studio-50051794:~/demo# python /root/demo/cli_demo.py
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.79s/it]
=============Welcome to InternLM chatbot, type 'exit' to exit.=============

User  >>> 请创作一个 300 字的小故事
在一片宁静的森林里，有一只名叫小兔的小兔子。它总是喜欢在森林里四处游荡，探索新奇的事物。有一天，它发现了一颗神奇的种子，种子散发着一种奇妙的能量，可以让它变得更加聪明和勇敢。小兔子非常兴奋，决定把这个种子带回家，并开始照料它。

经过一段时间的努力，小兔子终于发现，这颗神奇的种子不仅能够让小兔子变得更加聪明，还能够帮助它克服所有的困难和挑战。小兔子开始更加努力地学习和成长，它不断地探索和学习新的知识，变得更加勇敢和坚强。

随着时间的推移，小兔子的成长速度越来越快，它变得越来越聪明和有才华。它开始帮助其他小动物解决问题，并成为了森林里最受欢迎和受人尊敬的动物之一。小兔子的故事激励了所有的小动物，让它们相信只要努力和坚持，就能取得成功。

最终，小兔子的成长和努力得到了回报，它成为了森林里最优秀的动物之一。小兔子也明白了，只有通过不断学习和成长，才能真正实现自己的梦想和目标。从此以后，小兔子一直保持着对知识的渴望和追求，并成为了其他动物们的榜样。
User  >>> 
```


##  **猪猪Chat-InternLM2**

### **1 配置基础环境**

1. 运行环境命令：

 ```shell
 conda activate demo
```

2. 使用 `git` 命令来获得仓库内的 Demo 文件：

```shell
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
```



### **2 下载运行 Chat-八戒 Demo**

1. 在 `Web IDE` 中执行 `bajie_download.py`：

```shell
python /root/Tutorial/helloworld/bajie_download.py
```

2. 待下载完成后，输入运行命令：

```shell
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```

3. 先查询端口，再根据端口键入命令

```shell
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 41617
```

> 再复制下方的密码，输入到 `password` 中，直接回车：

### **3. 测试**

打开 [http://127.0.0.1:6006](http://127.0.0.1:6006/) 后，等待加载完成即可进行对话，键入内容示例如下：

```shell
你是谁，有什么本事

R

俺老猪就是猪八戒啊！俺可是天庭的天蓬元帅，管着天河的水域，威风凛凛的。不过俺因为一次醉酒调戏嫦娥的事情被玉皇大帝赶出天界，投胎到人间，结果错投了猪胎，变成了现在这个模样。不过俺可不是普通的猪，俺有着不俗的本事和才能，比如俺的武艺高强，力大无穷，还有一把好刀——九齿钉耙，能耙地翻江，斩妖除魔，是个了不起的猪八戒！不过俺也有俺的缺点，比如贪吃懒做，喜欢安逸舒适的生活，有时候也有点嘴硬，喜欢捉弄人。不过别看俺这样，俺可是个忠诚的伙伴，时刻保护唐僧师傅，帮助你取经，真是你的好朋友！

你取经路上数次想散伙，还好意思说自己忠诚？

R

你不知我八戒是个甚么脾气？人各有志，只要取经，还愁不长久么？

胡说八道，你就差留在高老庄待一辈子了吧

R

这可不敢当，我只要跟唐僧取经，吃香喝辣，过上舒服日子，就是天上的神仙也不如我自在！
```



`Lagent` 运行 `InternLM2-Chat-7B` 模型**

###  **1 初步介绍 Lagent 相关知识**

Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。

Lagent 的特性总结如下：

- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括：
    - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
    - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
    - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。


### **2   配置基础环境（开启 30% A100 ）**

> 该实例毕竟耗资源，需要将原来的开发机关闭，重新创建 50%算力的开发机，再进行

1. 重新开启开发机，输入命令，开启 conda 环境：

```shell
conda activate demo
```

2. 打开文件子路径

```shell
cd /root/demo
```

3. 使用 git 命令下载 Lagent 相关的代码库

```shell
git clone https://gitee.com/internlm/lagent.git
# git clone https://github.com/internlm/lagent.git
cd /root/demo/lagent
git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
pip install -e . # 源码安装
```


###  **3. 使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型为内核的智能体**

1. 打开 lagent 路径：

```shell
cd /root/demo/lagent
```


2. 在 terminal 中输入指令，构造软链接快捷访问方式：

```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```

3. 修改 `lagent` 路径下 `examples/internlm2_agent_web_demo_hf.py` 文件，并修改对应位置 (71行左右) 代码：

```python
    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        # model_name = st.sidebar.selectbox('模型选择：', options=['internlm'])
        model_name = st.sidebar.text_input('模型名称：', value='internlm2-chat-7b')
        meta_prompt = st.sidebar.text_area('系统提示词', value=META_CN)
        da_prompt = st.sidebar.text_area('数据分析提示词', value=INTERPRETER_CN)
        plugin_prompt = st.sidebar.text_area('插件提示词', value=PLUGIN_CN)
        model_path = st.sidebar.text_input(
            '模型路径：', value='/root/models/internlm2-chat-7b')
```

4. 输入运行命令 - **点开 6006 链接后，大约需要 5 分钟完成模型加载：**

```shell
streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006

(demo) root@intern-studio-50051794:~/demo/lagent# streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:6006

Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards:  12%|███████▍                                                   | 1/8 [00:04<00:33,  4.72s/it]Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [01:27<00:00, 10.88s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [01:16<00:00,  9.57s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained./it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards:  62%|████████████████████████████████████▉                      | 5/8 [00:27<00:16,  5.59s/it]Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [00:42<00:00,  5.31s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [00:35<00:00,  4.47s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [00:40<00:00,  5.07s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 8/8 [00:42<00:00,  5.28s/it]
```

>  需要做本地端口映射，另外打开的网页不要总是刷新，刷新一次需要重新加载一次，当一个进度条完成 8/8时，就是加载完成了


5. 打开 [http://127.0.0.1:6006](http://127.0.0.1:6006/) 后，（会有较长的加载时间）勾上数据分析，其他的选项不要选择，进行计算方面的 Demo 对话，即完成本章节实战。键入内容示例：

```shell

请解方程 2*X=1360 之中 X 的结果

当然可以，我将使用Python来解这个方程。

首先，我们将方程2*X=1360转换为Python代码，然后求解X的值。

IPythonInterpreter

`from sympy import symbols, Eq, solve # 定义变量 X = symbols('X') # 创建方程 equation = Eq(2*X, 1360) # 解方程 solution = solve(equation, X) # 输出解 solution`

`[680]`

通过解方程2*X=1360，我们得到X的值为680。因此，方程的解是X=680。

最终答案是：680​。
```


## **实践部署 `浦语·灵笔2` 模型**



### **1 配置基础环境**

1. 选用 `50% A100` 进行开发

2. 进入开发机，启动 `conda` 环境：

```shell
conda activate demo
# 补充环境包
pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5
```

3. 下载 **InternLM-XComposer 仓库** 相关的代码资源：

```shell
cd /root/demo
git clone https://gitee.com/internlm/InternLM-XComposer.git
# git clone https://github.com/internlm/InternLM-XComposer.git
cd /root/demo/InternLM-XComposer
git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626
```

4. 在 `terminal` 中输入指令，构造软链接快捷访问方式：

```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b
```


###  **2 图文写作实战**

1. 继续输入指令，用于启动 `InternLM-XComposer`：

```shell
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006
```

>  设置端口映射

###  **3 图片理解实战**

1. 关闭并重新启动一个新的 `terminal`，继续输入指令，启动 `InternLM-XComposer2-vl`：

```shell

(base) root@intern-studio-50051794:~/demo# conda activate demo
(demo) root@intern-studio-50051794:~/demo# cd /root/demo/InternLM-XComposer
(demo) root@intern-studio-50051794:~/demo/InternLM-XComposer# python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
> --code_path /root/models/internlm-xcomposer2-vl-7b \
> --private \
> --num_gpus 1 \
> --port 6006
You are using a model of type internlmxcomposer2 to instantiate a model of type internlm. This is not supported for all configurations of models and can yield errors.
Set max length to 4096
2024-04-06 15:48:57,509 - modelscope - INFO - PyTorch version 2.0.1 Found.
2024-04-06 15:48:57,513 - modelscope - INFO - Loading ast index from /root/.cache/modelscope/ast_indexer
2024-04-06 15:49:00,255 - modelscope - INFO - Loading done! Current index file version is 1.9.5, with md5 0910f1e486bf33d495af0c29998ad3d0 and a total number of 945 components indexed
Position interpolate from 24x24 to 35x35
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████| 2/2 [01:14<00:00, 37.25s/it]
Some weights of InternLMXComposer2ForCausalLM were not initialized from the model checkpoint at /root/models/internlm-xcomposer2-vl-7b and are newly initialized: ['vit.vision_tower.vision_model.post_layernorm.weight', 'vit.vision_tower.vision_model.post_layernorm.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running on local URL:  http://127.0.0.1:6006

To create a public link, set `share=True` in `launch()`.
```


2. 打开 [http://127.0.0.1:6006](http://127.0.0.1:6006/) (上传图片后) 键入内容示例如下：

```
请分析一下图中内容
```


