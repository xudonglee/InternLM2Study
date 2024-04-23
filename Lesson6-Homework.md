# Lesson6 Homework

## 1. Lagent Web Demo


### 1.1 使用 LMDeploy 部署

> 这是第一步，在第一个Terminal窗口启动api server

由于 Lagent 的 Web Demo 需要用到 LMDeploy 所启动的 api_server，因此我们首先按照下图指示在 vscode terminal 中执行如下代码使用 LMDeploy 启动一个 api_server。

```shell
(base) root@intern-studio-50051794:~# conda activate agent
(agent) root@intern-studio-50051794:~# lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
>                             --server-name 127.0.0.1 \
>                             --model-name internlm2-chat-7b \
>                             --cache-max-entry-count 0.1



[WARNING] gemm_config.in is not found; using default GEMM algo                                                 
HINT:    Please open http://127.0.0.1:23333 in a browser for detailed api usage!!!
HINT:    Please open http://127.0.0.1:23333 in a browser for detailed api usage!!!
HINT:    Please open http://127.0.0.1:23333 in a browser for detailed api usage!!!
INFO:     Started server process [7776]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:23333 (Press CTRL+C to quit)                        
```



### 1.2 启动并使用 Lagent Web Demo

> 这是第二步，在第二个Terminal窗口启动 Lagent Web Demo

接下来我们按照下图指示新建一个 terminal 以启动 Lagent Web Demo。在新建的 terminal 中执行如下指令：

```shell

(base) root@intern-studio-50051794:~# conda activate agent
s
streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
(agent) root@intern-studio-50051794:~# cd /root/agent/lagent/examples
(agent) root@intern-studio-50051794:~/agent/lagent/examples# streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:7860

```

### 1.3  设置端口映射

在等待 LMDeploy 的 api_server 与 Lagent Web Demo 完全启动后（如下图所示），在**本地**进行端口映射，将 LMDeploy api_server 的23333端口以及 Lagent Web Demo 的7860端口映射到本地。可以执行：

```shell
(base)  ✘ xudonglee@xudongdeMacBook-Pro  ~  ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 44157
The authenticity of host '[ssh.intern-ai.org.cn]:44157 ([8.130.47.207]:44157)' can't be established.
ED25519 key fingerprint is SHA256:FHKSn+aBDe/ZqW/92VSMgbyffG0Pp9ApyCiwCidliSI.
This host key is known by the following other names/addresses:
    ~/.ssh/known_hosts:30: [ssh.intern-ai.org.cn]:43055
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '[ssh.intern-ai.org.cn]:44157' (ED25519) to the list of known hosts.
root@ssh.intern-ai.org.cn's password:
```

接下来在本地的浏览器页面中打开 [http://localhost:7860](http://localhost:7860/) 以使用 Lagent Web Demo。

## 2. AgentLego

### 2. 1首先下载 demo 文件

```shell
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
```

由于 AgentLego 在安装时并不会安装某个特定工具的依赖，因此我们接下来准备安装目标检测工具运行时所需依赖。

### 2. 2 安装依赖包

AgentLego 所实现的目标检测工具是基于 mmdet (MMDetection) 算法库中的 RTMDet-Large 模型，因此我们首先安装 mim，然后通过 mim 工具来安装 mmdet。这一步所需时间可能会较长，请耐心等待。

```shell
conda activate agent
pip install openmim==0.3.9
mim install mmdet==3.3.0
```


### 2. 3 运行检测程序

3. 然后通过 `touch /root/agent/direct_use.py`（大小写敏感）的方式在 /root/agent 目录下新建 direct_use.py 以直接使用目标检测工具，

# 作业


## 基础作业


1. 完成 Lagent Web Demo 使用，并在作业中上传截图。文档可见 [Lagent Web Demo](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md#1-lagent-web-demo)


![ Lagent Web Demo  服务运行](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson6-lagent-web-demo.png)

![Lagent Web Demo 使用](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson6-homework-webdemo.png)


2. 完成 AgentLego 直接使用部分，并在作业中上传截图。文档可见 [直接使用 AgentLego](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#1-%E7%9B%B4%E6%8E%A5%E4%BD%BF%E7%94%A8-agentlego)。

![AgentLego直接使用，GPU资源占用](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson6-homework-agentlego01.png)

![AgentLego直接使用](https://github.com/xudonglee/InternLM2Study/blob/main/images/lesson6-homework-agentlego02.png)

## 进阶作业


1. 完成 AgentLego WebUI 使用，并在作业中上传截图。文档可见 [AgentLego WebUI](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#2-%E4%BD%9C%E4%B8%BA%E6%99%BA%E8%83%BD%E4%BD%93%E5%B7%A5%E5%85%B7%E4%BD%BF%E7%94%A8)。
2. 使用 Lagent 或 AgentLego 实现自定义工具并完成调用，并在作业中上传截图。文档可见：
    - [用 Lagent 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/lagent.md#2-%E7%94%A8-lagent-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B7%A5%E5%85%B7)
    - [用 AgentLego 自定义工具](https://github.com/InternLM/Tutorial/blob/camp2/agent/agentlego.md#3-%E7%94%A8-agentlego-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B7%A5%E5%85%B7)
