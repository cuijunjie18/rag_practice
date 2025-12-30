import json
import jieba
import pdfplumber
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

def split_text_fixed_size(text, chunk_size=100, overlap_size=5):
    """
    对长文本按固定长度切分，保留重叠部分以避免上下文断裂
    参数：
        text: 待切分的长文本
        chunk_size: 每个片段的最大长度（参考文章设为 100，适配汽车手册的密集信息）
        overlap_size: 片段间的重叠长度（设为 5，确保切分处的语义连贯）
    返回：
        new_text: 切分后的文本片段列表
    """
    new_text = []
    # 循环切分文本，步长=chunk_size（但重叠部分会覆盖前一个片段的末尾）
    for i in range(0, len(text), chunk_size):
        if i == 0:
            # 第一个片段：从开头取 chunk_size 长度
            new_text.append(text[0:chunk_size])
        else:
            # 后续片段：从 i-overlap_size 开始，取 chunk_size 长度（包含前一个片段的末尾 5 个字符）
            new_text.append(text[i - overlap_size : i + chunk_size])
    return new_text

def read_car_data(query_data_path, knowledge_data_path):
    """
    读取汽车知识问答数据集：问题集（JSON）和知识库（PDF）
    参数：
        query_data_path: questions.json 路径（用户问题）
        knowledge_data_path: 初赛训练数据集.pdf 路径（汽车知识库）
    返回：
        questions: 问题列表（每个元素是含"question"键的字典）
        pdf_content: 知识库片段列表（每个元素含"page"页码和"content"文本）
    """
    # 1. 读取 JSON 格式的问题集
    with open(query_data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)  # 格式示例：[{"question": "如何打开前机舱盖？"}, ...]
    
    # 2. 读取 PDF 格式的知识库，按页处理并切分
    pdf = pdfplumber.open(knowledge_data_path)
    pdf_content = []  # 存储（页码，文本片段）对
    
    for page_idx in range(len(pdf.pages)):
        # 提取当前页的文本（跳过空页）
        page_text = pdf.pages[page_idx].extract_text()
        if not page_text:
            continue
        
        # 调用切分函数，将当前页文本切成小片段
        text_chunks = split_text_fixed_size(page_text, chunk_size=100, overlap_size=5)
        
        # 记录每个片段的页码（page_1 表示第 1 页，符合用户阅读习惯）
        for chunk in text_chunks:
            pdf_content.append({
                "page": f"page_{page_idx + 1}",  # 页码从 1 开始（符合用户阅读习惯）
                "content": chunk.strip()  # 去除前后空格，避免冗余
            })
    
    pdf.close()  # 关闭 PDF 文件，释放资源
    return questions, pdf_content

def build_retrieval_libraries(pdf_content):
    """
    构建两种检索库：BM25 文本检索库 + 语义向量检索库
    参数：
        pdf_content: 知识库片段列表（含"page"和"content"）
    返回：
        bm25: BM25 检索实例
        sent_model: 语义嵌入模型（stella_base_zh_v3_1792d）
        pdf_embeddings: 知识库片段的语义向量（n 个片段 × 1792 维）
        pdf_texts: 知识库片段的文本列表（与向量一一对应）
    """
    # ------------------- 1. 构建 BM25 文本检索库 -------------------
    # BM25 需要输入“词列表”（每个片段按词分割），用 jieba 分词（中文适配）
    pdf_words = [jieba.lcut(chunk["content"]) for chunk in pdf_content]
    # 初始化 BM25 实例（用 BM25Okapi 算法，参考文章同款）
    bm25 = BM25Okapi(pdf_words)
    
    # ------------------- 2. 构建语义向量检索库 -------------------
    # 加载参考文章推荐的中文语义嵌入模型：stella_base_zh_v3_1792d（1792 维向量，语义捕捉能力强）
    sent_model = SentenceTransformer("infgrad/stella-base-zh-v3-1792d")
    # 提取所有知识库片段的文本（用于后续生成向量）
    pdf_texts = [chunk["content"] for chunk in pdf_content]
    # 生成语义向量（normalize_embeddings=True：归一化向量，加速余弦相似度计算）
    pdf_embeddings = sent_model.encode(
        pdf_texts,
        normalize_embeddings=True,
        show_progress_bar=True  # 显示进度条，方便观察
    )
    
    return bm25, sent_model, pdf_embeddings, pdf_texts

def rerank_results(question, candidate_chunks, pdf_content):
    """
    用 BAAI/bge-reranker-base 模型对候选片段重排，选最优片段
    参数：
        question: 用户问题
        candidate_chunks: 候选片段的索引列表（来自 BM25 和语义检索）
        pdf_content: 知识库片段列表（含"page"和"content"）
    返回：
        best_chunk: 重排后得分最高的片段（含"page"和"content"）
    """
    # 加载参考文章用的重排模型：BAAI/bge-reranker-base（中文匹配任务最优模型之一）
    rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    # 重排模型用 GPU 加速（若没有 GPU，可删去.cuda()，改用 CPU）
    # rerank_model = rerank_model.cuda()
    
    # 1. 准备重排输入：（问题，候选片段文本）对
    pairs = []
    candidate_indices = list(set(candidate_chunks))  # 去重，避免重复计算
    for idx in candidate_indices:
        chunk_text = pdf_content[idx]["content"]
        pairs.append([question, chunk_text])  # 格式：[问题, 片段文本]
    
    # 2. 重排模型推理（计算每个配对的得分）
    # 对输入文本编码（padding=True：自动补全，truncation=True：截断过长文本）
    inputs = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512  # 重排模型最大支持 512 tokens，足够汽车场景
    )
    # 模型推理（关闭梯度计算，节省内存）
    with torch.no_grad():
        # inputs = {k: v.cuda() for k, v in inputs.items()}  # 输入移到 GPU
        inputs = {k: v for k, v in inputs.items()}  # 输入移到 CPU
        outputs = rerank_model(**inputs)
        # 提取得分（重排模型的输出 logits 就是匹配得分）
        scores = outputs.logits.view(-1).cpu().numpy()  # 移回 CPU，转成 numpy 数组
    
    # 3. 选得分最高的片段
    best_idx = scores.argmax()  # 得分最高的配对索引
    best_chunk_idx = candidate_indices[best_idx]  # 对应知识库片段的索引
    best_chunk = pdf_content[best_chunk_idx]  # 得分最高的片段（含页码和文本）
    
    return best_chunk