# RAG(检索增强生成)应用demo

## 项目介绍

本demo对模型的使用采用api调用形式，但api调用会限制上下文长度，导致RAG输入文档不能太大，有条件的可以本地部署模型

## Get start

- 环境同步
  ```shell
  uv sync
  ```

- 执行代码
  ```shell
  uv run python main.py
  ```

- 效果展示
  可以查看[效果展示输出日志](assets/demo.log)  
  <br>

- 模型加载问题
  如果执行代码过程中，涉及到huggingface上模型下载慢等原因，可以设置下面的环境变量
  ```shell
  export HF_TOKEN="xxxxxxxxxxxx" # 你的HF-token
  export HF_ENDPOINT="https://hf-mirror.com"
  ```

## 模型选型

baseline模型：https://modelscope.cn/models/Qwen/Qwen3-0.6B  
嵌入层：https://huggingface.co/infgrad/stella-base-zh-v3-1792d  

## 参考

baseline：https://github.com/Infrasys-AI/AIInfra/tree/main/07Application/03RAG