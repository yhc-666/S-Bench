# Search-R1 Evaluation Framework

Evaluate LLMs with two search methods: tag-based (`<search>` tags) and function-based (direct function calls).

## Setup

### 1. Install Dependencies

```bash
pip install -r evaluations/requirements.txt
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

### 3. Start RAG Server

```bash
# In retriever environment
conda activate retriever
bash retrieval_launch.sh
```

### 4. (Optional) Start vLLM Server

For open-source models:
```bash
# Reads model config from evaluations/config/models.yaml
evaluations/scripts/start_vllm.sh qwen-7b

# Or for LLaMA
evaluations/scripts/start_vllm.sh llama-8b
```

## Run Evaluation

From the **Search-R1 root directory**:

```bash
# Tag-based search
python evaluations/run_evaluation.py --model qwen-7b --method tag

# Function-based search (with specialized search functions)
python evaluations/run_evaluation.py --model gpt-4 --method function

# Specific datasets
python evaluations/run_evaluation.py --model gpt-4 --method function --datasets nq popqa
```

## Configuration

All configs in `evaluations/config/`:

- **models.yaml**: Set `active_model` and API keys
- **datasets.yaml**: Set `active_datasets`
- **search_engines.yaml**: Set `search_method` (tag/function)
- **prompts.yaml**: Customize prompts

## Datasets

- **NQ**: Natural Questions
- **PopQA**: Popular QA
- **Musique**: Multi-hop reasoning
- **Bamboogle**: Complex reasoning

Data cached in `evaluations/data/` after first download.

## Results

Tag-based方式：
  {
      'id': str,           # 示例ID
      'question': str,     # 原始问题
      'gold_answer': str,  # 标准答案
      'prediction': str,   # 模型预测
      'response': str      # 完整响应文本
  }

  Function-based方式：
  {
      'id': str,           # 示例ID
      'question': str,     # 原始问题  
      'gold_answer': str,  # 标准答案
      'prediction': str,   # 模型预测
      'messages': List     # 完整对话历史
  }

## Search Methods

### Tag-based
- For Search-R1 trained models
- Uses `<search>query</search>` tags
- Direct text generation

### Function-based
- For any model supporting function/tool calling
- Specialized search functions (food, cloth, people, location, general)
- Direct execution without subprocess overhead

## Metrics

- **Exact Match (EM)**: Exact answer matching
- **F1 Score**: Token-level overlap
- **Search Stats**: Queries per question, iterations

## Scripts

- **`scripts/start_vllm.sh <model>`**: Start vLLM server with model from config

Scripts automatically read configurations from `config/*.yaml` files.

