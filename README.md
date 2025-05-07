# 基于大模型API的检索增强生成(RAG)系统实现

## 简介
本项目旨在研究、实现并评估一个检索增强生成（Retrieval-Augmented Generation, RAG）系统。该系统利用自定义的本地知识库，并通过大语言模型API来生成针对特定领域问题的回答。核心目标是探索RAG如何提升LLM在不重新训练的情况下处理特定知识的能力，并展示其在回答模型本身可能未知晓的问题时的优势。

## 特性
- 支持自定义本地知识库（如 `.txt`, `.md`, `.pdf` 文件）。
- 集成LLM API 进行文本生成。
- 使用 LangChain 框架简化RAG流程的构建和管理。
- 提供命令行工具 (CLI) 进行知识库的初始化和交互式问答。
- 通过示例对比展示RAG系统在提升答案准确性和领域相关性方面的效果。
- 包含对RAG相关技术和概念的总结。

## 技术栈
- **编程语言:** Python 3.10
- **核心框架/库:**
    - `LangChain`: 用于构建RAG流程。
    - `sentence-transformers`: 用于文本嵌入。
    - `FAISS` (Facebook AI Similarity Search): 用于高效向量相似度搜索。
    - `requests`: 用于与大模型的 API通信 (若无直接LangChain集成)。
    - `python-dotenv`: 管理API密钥等配置。
    - `unstructured`, `pypdf`: 文档加载。
- **LLM API:** 暂用免费的智谱bigmodel的推理模型glm-z1-flash API
- **知识库:** 本地文本文件。

## 目录结构说明
```
rag-lab/
├── .env                     # 存储API密钥和配置 (由用户创建)
├── .env.example             # .env文件的示例模板
├── data/                    # 存放知识库源文件 (例如: my_document.txt)
├── external_models/         # 存放从Hugging Face下载的模型文件
├── vector_store/            # 存储FAISS索引文件
├── src/
│   ├── __init__.py
│   ├── config.py            # 配置模块 (API密钥, 模型名称, 路径等)
│   ├── data_processor.py    # 文档加载、预处理、文本分割
│   ├── embedding_utils.py   # 嵌入模型加载和文本嵌入生成
│   ├── vector_store_manager.py # 向量数据库的创建、加载和检索
│   ├── llm_interface.py     # 与LLM API交互的接口
│   ├── rag_pipeline.py      # RAG核心逻辑，结合检索和生成
│   └── main.py              # 项目入口，命令行交互界面
├── download_nltk.py         # 下载NLTK资源
├── requirements.txt         # 项目依赖
└── README.md                # 本项目说明文档
```

## 安装与配置
1.  **克隆仓库** (如果您是从git仓库获取):
    ```bash
    git clone <repository-url>
    cd rag-lab
    ```
2.  **创建并激活Python虚拟环境**:
    ```bash
    conda create -n rag python=3.10 -y
    conda activate rag
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **下载NLTK资源 (重要)**:
    `unstructured`库（用于文档加载）的某些功能依赖NLTK包及其资源。在安装完上述依赖后，请运行以下命令下载必要的NLTK资源。
    在激活了 `rag` 环境的终端中运行它：
    ```bash
    python download_nltk.py
    ```
    *注意：此步骤需要稳定的网络连接。如果下载失败，请检查网络或稍后再试。*
5.  **Hugging Face模型下载配置 (重要)**:
    本项目使用的默认嵌入模型 (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) 需要从Hugging Face Hub下载。首次运行时，如果您的网络无法流畅访问 `huggingface.co`，可能会遇到模型下载失败或超时的问题。

    **解决方案:**
    *   **确保网络通畅**: 尝试直接在浏览器访问 `https://huggingface.co`。如果无法访问，请检查您的网络连接、防火墙或尝试使用网络代理。
    *   **配置代理 (如果需要)**: 如果您在需要代理的网络环境下，可能需要为Python配置 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量。例如，在PowerShell中：
        ```powershell
        $env:HTTP_PROXY="http://your_proxy_address:port"
        $env:HTTPS_PROXY="http://your_proxy_address:port"
        # 或者，如果代理需要认证:
        # $env:HTTPS_PROXY="http://username:password@your_proxy_address:port"
        ```
        请根据您的实际代理服务器信息进行替换。设置后，重新运行相关命令。
    *   **使用Hugging Face镜像/加速器**: 您可以尝试配置Hugging Face的国内镜像。例如，设置环境变量 `HF_ENDPOINT`：
        Windows
        ```powershell
        $env:HF_ENDPOINT="https://hf-mirror.com" # 请替换为实际可用的镜像地址
        ```
        Linux
        ```bash
        export HF_ENDPOINT=https://hf-mirror.com
        ```
    *   **手动预下载模型并从本地加载 (推荐的离线方案)**:
        1.  **下载模型**: 在网络条件好的环境中，使用以下Python代码（或其他工具如`git clone`）下载模型文件到本地指定目录。例如，创建一个 `external_models/` 目录存放：
            ```python
            # (在一个单独的Python脚本或解释器中运行)
            from huggingface_hub import snapshot_download
            model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            local_model_dir = "./external_models/paraphrase-multilingual-MiniLM-L12-v2" # 指定本地保存路径
            snapshot_download(repo_id=model_id, local_dir=local_model_dir, local_dir_use_symlinks=False)
            print(f"模型已下载到: {local_model_dir}")
            ```
        2.  **修改配置**: 在您的 `.env` 文件中，将 `EMBEDDING_MODEL_NAME` 变量的值修改为指向您本地存放模型文件的**绝对路径或相对项目根目录的路径**。
            例如，在 `.env` 文件中添加或修改：
            ```env
            EMBEDDING_MODEL_NAME="./external_models/text2vec-base-chinese"
            ```
            或者使用绝对路径：
            ```env
            EMBEDDING_MODEL_NAME="/path/to/your/project/rag/external_models/text2vec-base-chinese"
            ```
            代码中的 `src/embedding_utils.py` 会使用这个路径来加载本地模型，从而避免从网络下载。
6.  **配置API密钥等信息**:
    *   复制 `.env.example` 文件为 `.env`:
        ```bash
        # Windows
        copy .env.example .env
        # macOS/Linux
        cp .env.example .env
        ```
    *   编辑 `.env` 文件，填入您的 `LLM_API_KEY` 和其他可能需要的配置（如模型名称，以及上面提到的 `EMBEDDING_MODEL_NAME` 如果使用本地模型）。
7.  **准备知识库**:
    *   在 `data/` 目录下放入您的知识库源文件 (例如 `mydoc.txt`, `report.pdf`)。如果目录不存在，请创建它。

## 使用方法

### 1. 初始化/更新知识库
该命令会处理 `data/` 目录下的文档，生成文本嵌入，并创建/更新本地的FAISS向量索引。
```bash
python -m src.main --init-kb
```
如果需要强制重新创建知识库（即使已存在索引），使用:
```bash
python -m src.main --init-kb --force-recreate-kb
```

### 2. 使用RAG系统提问
```bash
python -m src.main --query "PostgreSQL 16中引入的哪个订阅参数可以防止双向逻辑复制中的无限循环，它可以取哪两个值？"
```
能回答正确并给出参考文档。
```
--- RAG System Answer ---
<think>
好的，我现在需要回答用户的问题：“PostgreSQL 16中引入的哪个订阅参数可以防止双向逻辑复制中的无限循环，它可以取哪两个值 ？”首先，我要仔细查看提供的上下文信息。

根据上下文，Bidirectional Logical Replication在PostgreSQL 16中引入了一个新的订阅参数origin。目的是解决旧版本中的无限 循环问题。当origin设置为none时，发布端只会发送本机写入，没有来自其他节点的变更，从而避免循环。而origin的默认值是any，此时发布端会发送所有变更，但在双向拓扑下可能导致回环。

用户的问题有两个部分：一是询问引入的参数名称，二是该参数可以取的两个值。根据上下文信息，参数名称是origin，两个值分别 是none和any。需要确认上下文是否明确说明这两个值是参数的可能取值。上下文提到当origin=none时和origin=any时的情况，因此 这两个值是正确的。

需要确保回答完全基于上下文，不添加额外信息。因此，正确的回答应该是：引入的参数是origin，取值none和any。要检查是否有其他可能的参数或值，但上下文没有提到其他值，所以确定这两个是正确的。
</think>
根据提供的上下文信息，PostgreSQL 16 中引入的订阅参数是 **origin**，它可以取以下两个值：
1. **none**：发布端仅发送本机写入（无 replication origin 的变更），避免循环。
2. **any**（默认）：发布端发送所有变更（本机 + 其他节点），需注意双向拓扑下可能引发回环。

参考：MyDBOps 博客《Bidirectional Logical Replication in PostgreSQL 16》，2023‑09‑02。

--- Retrieved Documents (Snippets) ---

[Document 1] Source: data\bidirectional_replication_pg16.md
  Snippet: Bidirectional Logical Replication – PostgreSQL 16

从 PostgreSQL 16 开始，逻辑复制支持 双向同步。 为了解决旧版本中无限循环的问题，引入了一个新的订阅参数 origin。       

当 origin = none 时，发布端 只会发送本机写入（没有 replication origin 的变更），从而避免循环。

当 origin = an...

------------------------------
```

系统将首先从知识库中检索相关信息，然后结合这些信息让LLM生成答案。

### 3. 仅使用LLM提问 (用于对比)
```bash
python -m src.main --query-llm-only "PostgreSQL 16中引入的哪个订阅参数可以防止双向逻辑复制中的无限循环，它可以取哪两个值？"
```
即使是reasoning模型有思考过程，但还是回答错误。
```
--- LLM-Only Answer ---
<think>
嗯，用户问的是PostgreSQL 16中引入的哪个订阅参数可以防止双向逻辑复制中的无限循环，并且这个参数可以取哪两个值。我需要先回忆一下PostgreSQL 16的新特性，特别是关于逻辑复制和订阅的部分。

首先，我记得逻辑复制订阅在PostgreSQL中是用来从主节点复制数据到从节点的，但有时候如果配置不当，可能会导致主从节点之间 互相发送变更，形成循环，导致数据不一致或者无限循环的问题。所以在PostgreSQL 16中应该引入了某个参数来解决这个问题。    

然后，我需要确认具体的参数名称。可能涉及到的是订阅的参数，比如*log replication*相关的设置。之前学过的内容中，有一个参数叫做*log rep心跳间隔*，或者类似的名字。不过更准确地说，应该是*log rep订阅的同步行为*相关的参数。或者是*log rep订阅 的同步参数*，比如*log rep sync*？

不过根据我的记忆，PostgreSQL 16引入的参数是*log rep sync*，它控制逻辑复制订阅的同步行为。这个参数可以设置为两种值：* 同步*（synchronous）和*异步*（asynchronous）。当设置为同步时，主节点会等待订阅的从节点确认收到变更后再继续发送，这样 可以防止循环，但可能会影响性能。而异步模式则允许主节点继续发送变更，减少延迟，但需要依赖订阅的可靠性。

不过需要确认这个参数的正确名称和值。可能我记错了参数的名字，比如是不是*log rep sync*还是*log rep syncron*？或者有没有其他名称？比如*log rep subscription*的某个参数？

再查一下资料，PostgreSQL 16的文档中提到，新增了*log rep sync*参数，用于逻辑复制订阅。这个参数的值可以是synchronous或asynchronous。当设置为synchronous时，主节点会等待订阅确认收到变更，从而避免双向同步中的循环。而asynchronous则允许主节 点继续发送变更，但需要确保订阅能够及时处理，避免循环。

所以正确的参数名称是*log rep sync*，取值是synchronous和asynchronous。用户的问题中提到的是“订阅参数”，所以应该就是这个参数。需要确保回答准确，参数名称和值正确无误。
</think>
PostgreSQL 16 中引入的订阅参数是 `log_rep_sync`，它用于控制逻辑复制订阅的同步行为。该参数可以取以下两个值：
1. **synchronous**（同步）：主节点会等待订阅的从节点确认收到变更后再继续发送，防止双向逻辑复制中的无限循环。
2. **asynchronous**（异步）：主节点不等待确认即可继续发送变更，适用于对延迟不敏感的场景，但需依赖从节点的可靠性以避免循环。

通过合理配置此参数，可有效避免主从节点间的逻辑复制循环问题。

------------------------------
```

系统将直接把问题发送给LLM，不经过知识库检索。这有助于对比RAG系统的效果。


## RAG技术总结 (待补充)
此部分将根据后续对RAG相关论文和资料的研究进行补充，主要包括：
- RAG的定义和核心组件（索引、检索、生成）。
- RAG技术的不同类别（例如：Naive RAG, Advanced RAG, Modular RAG）。
- RAG如何帮助LLM克服局限（如幻觉、知识过时）。
- RAG面临的挑战与未来发展方向。 

- 国际化课程 可能需要改成英文