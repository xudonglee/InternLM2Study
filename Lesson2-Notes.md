

##  **部署 `InternLM2-Chat-1.8B` 模型进行智能对话**

在课程提供的算力平台进行部署

https://studio.intern-ai.org.cn/console/instance/

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
