# Lesson4 Homework

本节课体验了使用Xtuner进行微调，虽然数据集很简单，但是流程是一样的。最重要的一段话如下：

> 微调也经常被戏称为是炼丹，就是说你炼丹的时候你得思考好用什么样的材料、用多大的火候、烤多久的时间以及用什么丹炉去烧。这里的丹炉其实我们可以想象为 XTuner ，只要丹炉的质量过得去，炼丹的时候不会炸，一般都是没问题的。但是假如炼丹的材料（就是数据集）本来就是垃圾，那无论怎么炼（微调参数的调整），炼多久（训练的轮数），炼出来的东西还只能且只会是垃圾。只有说用了比较好的材料，那么我们就可以考虑说要炼多久以及用什么办法去炼的问题。因此总的来说，学会如何构建一份高质量的数据集是至关重要的。

微调的工具Xtuner已经Ready了，更重要的是如何构建一份高质量的数据集。


---

# 第 4 节课作业


本节课的作业是训练自己的小助手，记录复现过程，整个训练的流程如下图所示：

- 训练小助手整体流程图 lesson4-xtuner-flow.png

![训练小助手整体流程图](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-flow.png)


## 基础作业（结营必做）

- 训练自己的小助手认知（记录复现过程并截图）


###  开发机准备

这部分内容比较简单，按照步骤操作即可，注意以下两点：

1. 创建开发机的时候，镜像不要选错，要选`Cuda11.7-conda` 镜像，然后在资源配置中，使用 `10% A100 * 1` 的选项，然后立即创建开发机器即可。

2. 一定注意使用Laucher界面中的 `Terminal` 进入终端界面进行操作！


###  环境安装

环境安装部分，需要在激活虚拟环境的前提下，下周xtuner的源码进行安装，一定要注意，**先激活环境，再下载源码安装，否则会出错**。

1. 激活环境
```shell
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 的环境：
# pytorch    2.0.1   py3.10_cuda11.7_cudnn8.5.0_0

studio-conda xtuner0.1.17
# 如果你是在其他平台：
# conda create --name xtuner0.1.17 python=3.10 -y

# 激活环境
conda activate xtuner0.1.17
```

2. 下载源码

```shell
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir -p /root/xtuner0117 && cd /root/xtuner0117

# 拉取 0.1.17 的版本源码
git clone -b v0.1.17  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd /root/xtuner0117/xtuner
```


3. 源码安装

```shell
# 从源码安装 XTuner
pip install -e '.[all]'
```


###  前期准备

该部分主要是准备好微调使用的数据集，使用的模型，以及微调使用的配置文件。

1. 数据集准备

	本实验比较简单，使用脚本生成了10000条同样的数据，导致最后微调的结果过拟合，对话测试的时候，每次不管问什么都返回同样的结果。

2. 模型准备

	为了微调的效率和尽量占用少的资源，使用 InternLM 最新推出的小模型 `InterLM-chat-1.8B` 来完成此次的微调，通过符号链接的方式链接到模型文件，这样既节省了空间，也便于管理。
	 > 注意后续使用的时候，模型的目录/root/ft/model/internlm2-chat-1_8b/，否则会提示找不到config.json文件
	 
	 如下所示：
	 ```shell
	 (xtuner0.1.17) root@intern-studio-50051794:/opt/jupyterlab# xtuner chat /root/ft/model --prompt-template internlm2_chat
[2024-04-14 16:39:21,817] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
[2024-04-14 16:40:02,214] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Traceback (most recent call last):
  File "/root/xtuner0117/xtuner/xtuner/tools/chat.py", line 491, in <module>
    main()
  File "/root/xtuner0117/xtuner/xtuner/tools/chat.py", line 235, in main
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
  File "/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 523, in from_pretrained
    config, kwargs = AutoConfig.from_pretrained(
  File "/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1138, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/transformers/configuration_utils.py", line 631, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/transformers/configuration_utils.py", line 686, in _get_config_dict
    resolved_config_file = cached_file(
  File "/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/transformers/utils/hub.py", line 369, in cached_file
    raise EnvironmentError(
OSError: /root/ft/model does not appear to have a file named config.json. Checkout 'https://huggingface.co//root/ft/model/tree/None' for available files.
(xtuner0.1.17) root@intern-studio-50051794:/opt/jupyterlab# xtuner chat /root/ft/model/internlm2-chat-1_8b --prompt-template internlm2_chat
```
	 
3. 配置文件准备
	XTuner 提供多个开箱即用的配置文件，可以通过下列命令查看：

```shell
# 列出所有内置配置文件
# xtuner list-cfg

# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
xtuner list-cfg -p internlm2_1_8b
```

由于我们是通过 `QLoRA` 的方式对 `internlm-chat-1.8b` 进行微调。而最相近的配置文件应该就是 `internlm2_1_8b_qlora_alpaca_e3` ，因此我们可以选择拷贝这个配置文件到当前目录：

```shell
# 创建一个存放 config 文件的文件夹
mkdir -p /root/ft/config

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config
```

> 这里我们就用到了 XTuner 工具箱中的第二个工具 `copy-cfg` ，该工具有两个必须要填写的参数 `{CONFIG_NAME}` 和 `{SAVE_PATH}` ，在我们的输入的这个指令中，我们的 `{CONFIG_NAME}` 对应的是上面搜索到的 `internlm2_1_8b_qlora_alpaca_e3` ,而 `{SAVE_PATH}` 则对应的是刚刚新建的 `/root/ft/config`。我们假如需要复制其他的配置文件只需要修改这两个参数即可实现。 输入后我们就能够看到在我们的 `/root/ft/config` 文件夹下有一个名为 `internlm2_1_8b_qlora_alpaca_e3_copy.py` 的文件了。

```
|-- config/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
```

- 再次强调

>其实微调也不过如此！事实上确实是这样的！其实在微调的时候最重要的还是要自己准备一份高质量的数据集，这个才是你能否真微调出效果最核心的利器。


4. 配置文件修改
###  模型训练

#### 1 常规训练


当我们准备好了配置文件好，我们只需要将使用 `xtuner train` 指令即可开始训练。

我们可以通过添加 `--work-dir` 指定特定的文件保存位置，比如说就保存在 `/root/ft/train` 路径下。假如不添加的话模型训练的过程文件将默认保存在 `./work_dirs/internlm2_1_8b_qlora_alpaca_e3_copy` 的位置，就比如说我是在 `/root/ft/train` 的路径下输入该指令，那么我的文件保存的位置就是在 `/root/ft/train/work_dirs/internlm2_1_8b_qlora_alpaca_e3_copy` 的位置下。

```shell
# 指定保存路径
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
```

- lesson4-xtuner-train-001
![训练过程1](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-train-001.png)
- lesson4-xtuner-train-002
![训练过程2](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-train-002.png)
- lesson4-xtuner-train-003
![训练过程3](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-train-003.png)
- lesson4-xtuner-train-004
![训练过程4](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-train-004.png)
- lesson4-xtuner-train-005
![训练过程5](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-train-005.png)


在输入训练完后的文件如下所示：

```
|-- train/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
    |-- iter_600.pth
    |-- last_checkpoint
    |-- iter_768.pth
    |-- iter_300.pth
    |-- 20240406_203957/
        |-- 20240406_203957.log
        |-- vis_data/
            |-- 20240406_203957.json
            |-- eval_outputs_iter_599.txt
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- eval_outputs_iter_299.txt
            |-- config.py
```

#### 2 使用 deepspeed 来加速训练


我们也可以结合 XTuner 内置的 `deepspeed` 来加速整体的训练过程，共有三种不同的 `deepspeed` 类型可进行选择，分别是 `deepspeed_zero1`, `deepspeed_zero2` 和 `deepspeed_zero3`（详细的介绍可看下拉框）。

DeepSpeed优化器及其选择方法

```shell
# 使用 deepspeed 来加速训练
xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2
```

- lesson4-xtuner-deepspeed-001

![deepspeed训练过程1](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-deepspeed-001.png)

可以看到，通过 `deepspeed` 来训练后得到的权重文件和原本的权重文件是有所差别的，原本的仅仅是一个 .pth 的文件，而使用了 `deepspeed` 则是一个名字带有 .pth 的文件夹，在该文件夹里保存了两个 .pt 文件。当然这两者在具体的使用上并没有太大的差别，都是可以进行转化并整合。

```
|-- train_deepspeed/
    |-- internlm2_1_8b_qlora_alpaca_e3_copy.py
    |-- zero_to_fp32.py
    |-- last_checkpoint
    |-- iter_600.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
    |-- 20240406_220727/
        |-- 20240406_220727.log
        |-- vis_data/
            |-- 20240406_220727.json
            |-- eval_outputs_iter_599.txt
            |-- eval_outputs_iter_767.txt
            |-- scalars.json
            |-- eval_outputs_iter_299.txt
            |-- config.py
    |-- iter_768.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
    |-- iter_300.pth/
        |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
        |-- mp_rank_00_model_states.pt
```

- lesson4-xtuner-deepspeed-002

![deepspeed训练过程2](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner_deepspeed-002.png)

###  模型验证

#### 1 模型转换


模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件，那么我们可以通过以下指令来实现一键转换。

```shell
# 创建一个保存转换后 Huggingface 格式的文件夹
mkdir -p /root/ft/huggingface

# 模型转换
# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
```


- lesson4-xtuner-convert-01

![模型转换](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-convert-01.png)


转换完成后，可以看到模型被转换为 Huggingface 中常用的 .bin 格式文件，这就代表着文件成功被转化为 Huggingface 格式了。

- lesson4-xtuner-hf
![模型转换HF格式](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-hf.png)


```
|-- huggingface/
    |-- adapter_config.json
    |-- xtuner_config.py
    |-- adapter_model.bin
    |-- README.md
```

**此时，huggingface 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”**

> 可以简单理解：LoRA 模型文件 = Adapter

除此之外，我们其实还可以在转换的指令中添加几个额外的参数，包括以下两个：

|参数名|解释|
|---|---|
|--fp32|代表以fp32的精度开启，假如不输入则默认为fp16|
|--max-shard-size {GB}|代表每个权重文件最大的大小（默认为2GB）|

假如有特定的需要，我们可以在上面的转换指令后进行添加。由于本次测试的模型文件较小，并且已经验证过拟合，故没有添加。假如加上的话应该是这样的：

```shell
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface --fp32 --max-shard-size 2GB
```



#### 2 模型整合


对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（adapter）。那么训练完的这个层最终还是要与原模型进行组合才能被正常的使用。

而对于全量微调的模型（full）其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 adapter ，因此是不需要进行模型整合的。

在 XTuner 中也是提供了一键整合的指令，但是在使用前我们需要准备好三个地址，包括原模型的地址、训练好的 adapter 层的地址（转为 Huggingface 格式后保存的部分）以及最终保存的地址。

```shell
# 创建一个名为 final_model 的文件夹存储整合后的模型文件
mkdir -p /root/ft/final_model

# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
```

- lesson4-xtuner-convert-merge

![模型整合](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-convert-merge.png)

#### 3 对话测试


在 XTuner 中也直接的提供了一套基于 transformers 的对话代码，让我们可以直接在终端与 Huggingface 格式的模型进行对话操作。我们只需要准备我们刚刚转换好的模型路径并选择对应的提示词模版（prompt-template）即可进行对话。假如 prompt-template 选择有误，很有可能导致模型无法正确的进行回复。

> 想要了解具体模型的 prompt-template 或者 XTuner 里支持的 prompt-tempolate，可以到 XTuner 源码中的 `xtuner/utils/templates.py` 这个文件中进行查找。

```shell
# 与模型进行对话
xtuner chat /root/ft/final_model --prompt-template internlm2_chat
```

我们可以通过一些简单的测试来看看微调后的模型的能力。

> 假如我们想要输入内容需要在输入文字后敲击两下回车，假如我们想清楚历史记录需要输入 RESET，假如我们想要退出则需要输入 EXIT。

- lesson4-xtuner-test-01

![模型测试1](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-test-01.png)


```
(xtuner0.1.17) root@intern-studio-50051794:/opt/jupyterlab# # 与模型进行对话
(xtuner0.1.17) root@intern-studio-50051794:/opt/jupyterlab# xtuner chat /root/ft/final_model --prompt-template internlm2_chat 
[2024-04-14 16:28:11,149] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
[2024-04-14 16:28:36,810] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Loading checkpoint shards:   0%|                                                                       | 0/2 [00:00<?, ?it/s]/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.33s/it]
Load LLM from /root/ft/final_model

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 是<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你是谁

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 请你介绍一下你自己

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你是我的小助手吗

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你是谁的帮手3

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你可以是我的帮手吗

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>exit  

我是西东东西的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦</s>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> EXIT

Log: Exit!
```

可以看到模型已经严重过拟合，回复的话就只有 “我是剑锋大佬的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦” 这句话。我们下面可以通过对比原模型的能力来看看差异。

```shell
# 同样的我们也可以和原模型进行对话进行对比
xtuner chat /root/ft/model --prompt-template internlm2_chat
```

我们可以用同样的问题来查看回复的情况。

- lesson4-xtuner-test-02

![模型测试2](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-test-02.png)


```
(xtuner0.1.17) root@intern-studio-50051794:/opt/jupyterlab# xtuner chat /root/ft/model/internlm2-chat-1_8b --prompt-template internlm2_chat
[2024-04-14 16:41:22,740] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
[2024-04-14 16:41:54,262] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:43<00:00, 21.52s/it]
Load LLM from /root/ft/model/internlm2-chat-1_8b

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你是谁

我是一个人工智能助手，由上海人工智能实验室开发。我致力于通过执行常见的基于语言的任务和提供建议来帮助人类。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但是，我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 请你介绍一下你自己

你好，我是一个人工智能助手，由上海人工智能实验室开发。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但是，我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> 你是我的小助手吗

是的，我是一款人工智能助手，旨在帮助人类完成各种基于语言的任务。我使用了Transformer模型和深度学习技术，并使用语言模型作为预训练任务。我能够回答问题、提供定义和解释、将文本从一种语言翻译成另一种语言、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码以及其他任何基于语言的任务。但是，我不能看、听、尝、触摸、闻、移动、与物理世界交互、感受情感或体验感官输入、执行需要身体能力的任务。<|im_end|>

double enter to end input (EXIT: exit chat, RESET: reset history) >>> EXIT

Log: Exit!
```

可以看到在没有进行我们数据的微调前，原模型是能够输出有逻辑的回复，并且也不会认为他是我们特有的小助手。因此我们可以很明显的看出两者之间的差异性。

那对于 `xtuner chat` 这个指令而言，还有很多其他的参数可以进行设置的，包括：

|启动参数|解释|
|---|---|
|--system|指定SYSTEM文本，用于在对话中插入特定的系统级信息|
|--system-template|指定SYSTEM模板，用于自定义系统信息的模板|
|**--bits**|指定LLM运行时使用的位数，决定了处理数据时的精度|
|--bot-name|设置bot的名称，用于在对话或其他交互中识别bot|
|--with-plugins|指定在运行时要使用的插件列表，用于扩展或增强功能|
|**--no-streamer**|关闭流式传输模式，对于需要一次性处理全部数据的场景|
|**--lagent**|启用lagent，用于特定的运行时环境或优化|
|--command-stop-word|设置命令的停止词，当遇到这些词时停止解析命令|
|--answer-stop-word|设置回答的停止词，当生成回答时遇到这些词则停止|
|--offload-folder|指定存放模型权重的文件夹，用于加载或卸载模型权重|
|--max-new-tokens|设置生成文本时允许的最大token数量，控制输出长度|
|**--temperature**|设置生成文本的温度值，较高的值会使生成的文本更多样，较低的值会使文本更确定|
|--top-k|设置保留用于顶k筛选的最高概率词汇标记数，影响生成文本的多样性|
|--top-p|设置累计概率阈值，仅保留概率累加高于top-p的最小标记集，影响生成文本的连贯性|
|--seed|设置随机种子，用于生成可重现的文本内容|

除了这些参数以外其实还有一个非常重要的参数就是 `--adapter` ，这个参数主要的作用就是可以在转化后的 adapter 层与原模型整合之前来对该层进行测试。使用这个额外的参数对话的模型和整合后的模型几乎没有什么太多的区别，因此我们可以通过测试不同的权重文件生成的 adapter 来找到最优的 adapter 进行最终的模型整合工作。

```shell
# 使用 --adapter 参数与完整的模型进行对话
xtuner chat /root/ft/model --adapter /root/ft/huggingface --prompt-template internlm2_chat
```
###  模型部署


1. 下载网页端 web demo 所需要的依赖。

```shell
pip install streamlit==1.24.0
```

2. 下载 [InternLM](https://github.com/InternLM/InternLM) 项目代码

```shell
# 创建存放 InternLM 文件的代码
mkdir -p /root/ft/web_demo && cd /root/ft/web_demo

# 拉取 InternLM 源文件
git clone https://github.com/InternLM/InternLM.git

# 进入该库中
cd /root/ft/web_demo/InternLM
```

3. 修改 `/root/ft/web_demo/InternLM/chat/web_demo.py` 中的内容

```shell
(xtuner0.1.17) root@intern-studio-50051794:~/ft/web_demo/InternLM/chat# diff web_demo.py web_demo_bak.py 
38,40c38,40
<     max_length: int = 2048
<     top_p: float = 0.75
<     temperature: float = 0.1
---
>     max_length: int = 32768
>     top_p: float = 0.8
>     temperature: float = 0.8
42c42
<     repetition_penalty: float = 1.000
---
>     repetition_penalty: float = 1.005
183c183
<     model = (AutoModelForCausalLM.from_pretrained('/root/ft/final_model',
---
>     model = (AutoModelForCausalLM.from_pretrained('internlm/internlm2-chat-7b',
186c186
<     tokenizer = AutoTokenizer.from_pretrained('/root/ft/final_model',
---
>     tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b',
196,198c196,198
<                                value=2048)
<         top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
<         temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
---
>                                value=32768)
>         top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
>         temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
216c216,218
<     meta_instruction = ('')
---
>     meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
>                         'and harmless AI assistant developed by Shanghai '
>                         'AI Laboratory (上海人工智能实验室).')
236a239,240
>     user_avator = 'assets/user.png'
>     robot_avator = 'assets/robot.png'
238c242
<     st.title('InternLM2-Chat-1.8B')
---
>     st.title('InternLM2-Chat-7B')
254c258
<         with st.chat_message('user'):
---
>         with st.chat_message('user', avatar=user_avator):
260a265
>             'avatar': user_avator
263c268
<         with st.chat_message('robot'):
---
>         with st.chat_message('robot', avatar=robot_avator):
278a284
>             'avatar': robot_avator,
```

4. 运行 `/root/personal_assistant/code/InternLM` 目录下的 `web_demo.py` 文件

```shell
(xtuner0.1.17) root@intern-studio-50051794:~/ft/web_demo/InternLM/chat# streamlit run /root/ft/web_demo/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:6006

load model begin.
Loading checkpoint shards:   0%|                                                                       | 0/2 [00:00<?, ?it/s]/root/.conda/envs/xtuner0.1.17/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████| 2/2 [00:25<00:00, 12.87s/it]
load model end.
load model begin.
load model end.
load model begin.
load model end.
load model begin.
load model end.
load model begin.
load model end.
load model begin.
load model end.
load model begin.
load model end.
```

- lesson4-xtuner-webdemo-01

![模型部署1](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-webdemo-01.png)


5. 做好本地端口映射，在浏览器中进行测试验证

在 iTerm 中输入以下内容（需要替换为自己的端口号）

```shell
# 从本地使用 ssh 连接 studio 端口
# 将下方端口号 38374 替换成自己的端口号
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
```


如下：


```shell
(base)  ✘ xudonglee@192  ~  ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p  43055
The authenticity of host '[ssh.intern-ai.org.cn]:43055 ([8.130.47.207]:43055)' can't be established.
ED25519 key fingerprint is SHA256:FHKSn+aBDe/ZqW/92VSMgbyffG0Pp9ApyCiwCidliSI.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '[ssh.intern-ai.org.cn]:43055' (ED25519) to the list of known hosts.
root@ssh.intern-ai.org.cn's password:
(base)  xudonglee@192  ~  ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p  43055
root@ssh.intern-ai.org.cn's password:
```

6. 浏览器中对话

- lesson4-xtuner-webdemo-02

![模型部署2](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson4-xtuner-webdemo-02.png)


## 进阶作业

- 将自我认知的模型上传到 OpenXLab，并将应用部署到 OpenXLab（优秀学员必做）
- 复现多模态微调（优秀学员必做）
