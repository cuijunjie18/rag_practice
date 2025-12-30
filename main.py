# 导入所有需要的库
from openai import OpenAI
from dotenv import load_dotenv
from rag import *
import os

load_dotenv()

client = OpenAI(
    base_url = os.getenv("LLM_BASE_URL"),
    api_key = os.getenv("LLM_API_KEY")
)
    
def qwen3_without_rag(question):
    """
    无 RAG 的 Qwen3 直接生成回答（仅依赖模型自身训练数据）
    参数：
        question: 用户问题
    返回：
        answer: 模型直接生成的回答
    """
    # 构建基础 prompt（无参考资料，仅问题）
    messages = [
        {
            "role": "system",
            "content": "你是汽车知识问答助手，请回答用户关于汽车操作、手册的问题。"
                      "如果不知道具体信息，直接说明；不要编造内容。"
        },
        {
            "role": "user",
            "content": question
        }
    ]
    response = client.chat.completions.create(
        model='Qwen/Qwen3-0.6B', # ModelScope Model-Id
        messages = messages,
        stream = True,
    )
    answer = ""
    for chunk in response:
        if chunk.choices:
            print(chunk.choices[0].delta.content, end='', flush=True)
            answer += chunk.choices[0].delta.content
    return answer

def qwen3_with_rag(question, pdf_content, bm25, sent_model, pdf_embeddings):
    """
    有 RAG 的 Qwen3 生成回答（基于汽车手册片段生成）
    参数：
        question: 用户问题
        其他参数：前面构建的检索库、模型等
    返回：
        answer: 基于手册的回答
        reference_page: 参考的手册页码
    """
    # 步骤 1：检索候选片段（BM25+语义检索各 top10）
    question_words = jieba.lcut(question)
    bm25_scores = bm25.get_scores(question_words)
    bm25_top10 = bm25_scores.argsort()[-10:]
    question_emb = sent_model.encode(question, normalize_embeddings=True)
    semantic_scores = question_emb @ pdf_embeddings.T
    semantic_top10 = semantic_scores.argsort()[-10:]
    
    # 步骤 2：重排选最优片段
    candidate_indices = list(set(bm25_top10) | set(semantic_top10))
    best_chunk = rerank_results(question, candidate_indices, pdf_content)
    reference_page = best_chunk["page"]
    reference_text = best_chunk["content"]
    
    # 步骤 3：构建带参考资料的 prompt
    messages = [
        {
            "role": "system",
            "content": "你是汽车知识问答助手，必须基于给定的参考资料回答问题。"
                      "如果资料中没有答案，输出“结合给定的资料，无法回答问题”；"
                      "如果有答案，需包含参考的手册页码（如“参考 page_307”），不要编造内容。"
        },
        {
            "role": "user",
            "content": f"参考资料：{reference_text}\n 用户问题：{question}"
        }
    ]
    
    # 步骤 4：模型生成回答
    response = client.chat.completions.create(
        model='Qwen/Qwen3-0.6B', # ModelScope Model-Id
        messages = messages,
        stream = True,
    )
    answer = ""
    for chunk in response:
        if chunk.choices:
            print(chunk.choices[0].delta.content, end='', flush=True)
            answer += chunk.choices[0].delta.content
    
    return answer, reference_page

if __name__ == "__main__":
    # =============== step1：读取数据集 ===========================
    questions, pdf_content = read_car_data(
        query_data_path = os.path.join("data", "questions.json"),
        knowledge_data_path = os.path.join("data", "初赛训练数据集.pdf")
    )
    
    # 打印读取结果，验证是否成功
    print(f"共读取到 {len(questions)} 个问题")
    print(f"共生成 {len(pdf_content)} 个知识库片段")
    print("\n 前 2 个问题示例：")
    for i in range(2):
        print(f"问题{i+1}：{questions[i]['question']}")
    print("\n 前 2 个知识库片段示例：")
    for i in range(2):
        print(f"{pdf_content[i]['page']}：{pdf_content[i]['content'][:50]}...")
        
        
    # ================ step2: 构建检索向量库 ========================
    bm25, sent_model, pdf_embeddings, pdf_texts = build_retrieval_libraries(pdf_content)
    print(f"BM25 检索库构建完成（共{len(pdf_texts)}个片段）")
    print(f"语义向量库构建完成（向量维度：{pdf_embeddings.shape[1]}）")
    
    
    # ================ (测试step): 结果重排 ==============================
    
    # 选一个测试问题，先获取候选片段，再重排
    test_question = "如何打开前机舱盖？"

    # 1. BM25 检索 top10 片段（用 jieba 分词问题，获取得分，取前 10 个索引）
    question_words = jieba.lcut(test_question)
    bm25_scores = bm25.get_scores(question_words)
    bm25_top10 = bm25_scores.argsort()[-10:]  # 得分从低到高排序，取后 10 个（top10）

    # 2. 语义检索 top10 片段（生成问题向量，计算与所有片段的余弦相似度）
    question_emb = sent_model.encode(test_question, normalize_embeddings=True)
    # 余弦相似度 = 点积（归一化后）
    semantic_scores = question_emb @ pdf_embeddings.T
    semantic_top10 = semantic_scores.argsort()[-10:]  # 取 top10 索引

    # 3. 合并候选片段（去重），重排选最优
    candidate_indices = list(set(bm25_top10) | set(semantic_top10))
    best_chunk = rerank_results(test_question, candidate_indices, pdf_content)
    print(f"问题：{test_question}")
    print(f"最优参考片段（{best_chunk['page']}）：{best_chunk['content']}")
    
    
    # ================= step3: 对比实验 ===========================
    
    # 选择 3 个典型测试问题（来自 questions.json，索引可根据实际数据调整）
    test_questions = [
        questions[0]["question"],  # 问题 1：操作步骤类（如“如何打开前机舱盖？”）
        questions[5]["question"],  # 问题 2：页码定位类（如“儿童安全座椅固定装置在手册第几页？”）
        questions[10]["question"]  # 问题 3：法规要求类（如“根据国家环保法，车辆在什么情况下需要报废？”）
    ]

    # 运行对比实验
    print("="*80)
    print("Qwen3-0.6B 无 RAG vs 有 RAG 对比实验")
    print("="*80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n【测试问题{i}】：{question}")
        print("-"*60)

        # 1. 无 RAG 的生成结果
        print("1. 无 RAG（仅依赖模型自身知识）：")
        answer_without_rag = qwen3_without_rag(question)
        print(f"   回答：{answer_without_rag}")

        # 2. 有 RAG 的生成结果
        print("2. 有 RAG（基于汽车手册片段）：")
        answer_with_rag, ref_page = qwen3_with_rag(question, pdf_content, bm25, sent_model, pdf_embeddings)
        print(f"   回答：{answer_with_rag}")
        print(f"   参考手册页码：{ref_page}")
        print("-"*60)
