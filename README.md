# 🦙 Text2SQL LLaMA Analyst

*Transforming natural language into SQL queries with a fine-tuned LLaMA-2 model.*

My project fine-tunes the powerful `LLaMA-2-7B` model to translate natural language questions into SQL, a real-world application of adapting large language models (LLMs) for domain-specific reasoning. By combining LoRA (Low-Rank Adaptation), 4-bit quantization, and lightweight training, my project delivers strong results while maintaining efficient compute usage (even on limited resources like Google Colab).

This was an insightful exploration of how to make LLMs more useful, efficient, and task-aware. If you're curious about NLP applications in databases, fine-tuning with PEFT (Parameter-Efficient Fine-Tuning) techniques, or just want to see what LLaMA can do when tuned for a specific task, you're in the right place.

⚙️ [Live Colab Demo](https://colab.research.google.com/drive/1ISQ9jpYCEMQQ6WSKRqF60p3ZIEsJuuFo?usp=sharing)

## 🗺️ Motivation & Learning Journey

I’ve always been fascinated by the challenge of bridging natural language and structured data. SQL is one of the most widely used query languages, but crafting coherent queries can be intimidating for non-technical users. One of my goals was to explore how a large language model like `LLaMA-2-7B` could be fine-tuned to generate coherent SQL queries from plain English questions, making database access as simple as asking a question.

My project started with questions like **How can we take a general-purpose LLM and specialize it efficiently for a niche but practical task like Text-to-SQL?**. I experimented with fine-tuning techniques, including **LoRA (Low-Rank Adaptation)**, which allowed me to train efficiently on limited hardware, and **quantization**, which reduced memory usage without significantly hurting performance. These techniques are essential for lightweight training of massive LLMs.

Along the way, I gained a deeper understanding of:
- **Prompt-to-SQL pipelines**: how models translate natural language into structured outputs.
- **Parameter-efficient fine-tuning**: adapting large models without retraining billions of parameters.
- **Evaluation challenges**: ensuring the generated SQL queries are reasonable and coherent.

This project gave me a glimpse into practical and real-world applications of fine-tuned LLMs in domain-specific tasks, while also emphasizing the importance of efficiency in handling large models. 

## 🛠️ Technical Highlights

To fine-tune `LLaMA-2-7B` for translating natural language questions into SQL queries, I combined several cutting-edge techniques that made this project efficient and scalable:

### ⚡ LoRA (Low-Rank Adaptation)
Instead of updating every parameter in the 7B-parameter LLaMA model, I used **LoRA** to inject a small number of trainable matrices into key attention and feed-forward layers. This drastically reduced training time and memory usage while still enabling the model to learn the task effectively.

- Targeted modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Benefits: Trainability on limited hardware, no full model overwrite

### 🧠 Quantization (4-bit)
To further reduce the model's memory footprint, I loaded the model using **4-bit quantization (NF4)**. This allowed me to load and fine-tune LLaMA-2 on a free Colab GPU without running out of memory.

- Compute dtype: `float16` (balance between speed and precision)
- Storage: 4-bit weights using `bitsandbytes` backend
- Trade-off: Slight degradation in raw accuracy but vastly improved accessibility

### 📊 Dataset & Task
I fine-tuned on the `ChrisHayduk/Llama-2-SQL-Dataset`, which includes natural language questions alongside their target SQL queries. The task is **causal language modeling (CLM)**, where the model learns to predict SQL completions based on the input prompt.

- Input format: `"### Instruction: ... \n### Input: ... \n### Response: "`
- Trained using: `Trainer`, `Transformers`, `PEFT`, `LoRA`, `BitsAndBytes`

My approach highlights the potential of **task-specific fine-tuning** for enabling users to interact with structured data using natural language.

## 🚀 Getting Started

Whether you're here to explore my demo or dive into my codebase, I’ve made it easy to get started.

### 🔗 Option 1: Try the Demo Instantly (No Setup Required)

If you'd like to **try the model without installing anything**, open my interactive Colab demo below:

👉🏻 [Open my Colab Demo](https://colab.research.google.com/drive/1ISQ9jpYCEMQQ6WSKRqF60p3ZIEsJuuFo?usp=sharing)

This notebook lets you:
- Run inference with my fine-tuned model hosted on Hugging Face 
- Compare model predictions to ground truth SQL queries from the evaluation dataset 
- See how natural language is translated into structured database queries 


### 💻 Option 2: Run Locally or Customize the Project

If you want to explore the code or retrain the model:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Akhan521/Text2SQL-LLaMA-Analyst.git
   cd Text2SQL-LLaMA-Analyst
   ```
2. **(Optional) Create + Activate a Virtual Environment**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Mac: source .venv/bin/activate
    ```
3. **Install Dependencies**
   ```bash
   # Necessary modules:
   pip install -r requirements.txt
   ```
4. **Run Inference**
   ```bash
   # Test the model's SQL completions:
   python -m src.inference
   ```
   > If you're loading your locally fine-tuned model from `logs/`, be sure to set use_local_finetuned to True in `inference.py`.
5. **(Optional) Fine-Tune Model Locally**
   ```bash
   # This will fine-tune llama-2 on the SQL dataset
   # and save your model + tokenizer inside the logs/ directory.
   python -m src.train
   ```
   > You may need a GPU and are free to adjust the training arguments in `train.py`.

All core functionality (loading the tokenizer, quantized model loading, data preprocessing, and text generation) is modular and found in the `src/` directory. Feel free to reuse or adapt them in your own projects!

## 📌 Key Takeaways

Here are the 3 biggest things I learned from this project:

1. **Fine-Tuning at Scale Is Possible**  
   Using LoRA and quantization, I successfully fine-tuned a 7B-parameter LLaMA-2 model on limited hardware, unlocking powerful capabilities with limited compute.

2. **Accuracy Still Needs Work**  
   While the model outperforms the base version, it often generates incomplete or incorrect SQL queries. Fine-tuning helps, but smart evaluation and data quality matter just as much.

3. **Building for Others Is Powerful**  
   Creating an interactive Colab demo taught me how to make complex ML topics accessible by turning raw models into tools others can actually use.

## 🤝🏻 Get in Touch

Thank you for reading about my project and sharing your time with me! I'd love to hear your feedback, ideas, or advice you may have for me.

- 📬 **GitHub**: View my GitHub profile [here](https://github.com/Akhan521).
- 💼 **LinkedIn**: [Connect with me](https://www.linkedin.com/in/aamir-khan-aak521/) to chat about this project or others like it.
- 📓 **Portfolio**: Explore more of my work at [https://aamir-khans-portfolio.vercel.app/](https://aamir-khans-portfolio.vercel.app/).


