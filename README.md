# How to execute the scripts (e.g., command line parameters)
# RAG-enabled Scientific Literature Assistant

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that can answer questions about scientific papers from the course literature.

The system consists of:
- An indexing script (`indexer.py`) that processes PDF files and builds a vector database.
- A query script (`query.py`) that retrieves relevant content and uses multiple specialized agents to answer questions.

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface
pip install faiss-cpu sentence-transformers pypdf requests
```
### 2. Build the vertor Store

```bash
python indexer.py
```

### 3. Run the Assistant

```bash
python python query.py
```

# File names

1. Attention Is All You Need
2. Can AI Expand the Human Mind?
3. More Agents Is All You Need

# Questions

1. What is the main idea behind the Transformer model?
2. Did the Transformer perform better than previous models?
3. What is one problem with using self-attention?
4. What is “System 0” in the article?
5. How can AI help people make decisions?
6. What is one risk of relying too much on AI?
7. How does the paper use multiple agents to improve answers?
8. What happens when you increase the number of agents?
9. What is one downside of using many agents?

# Answers

1.  [General Synthesizer]
The main idea behind the Transformer model is to use stacked self-attention mechanisms combined with point-wise, fully connected feed-forward networks for both the encoder and decoder. This architecture employs residual connections and layer normalization to facilitate training. The model avoids recurrent 
or convolutional structures and instead relies on attention mechanisms, specifically scaled dot-product attention and multi-head attention, to process sequences in parallel efficiently (Source: Attention_Is_All_You_Need.pdf, Pages 1-3).

2. [Results Extractor]
 Yes, the Transformer achieved better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests. For example, the Transformer (base model) achieved a BLEU score of 25.8 on English-to-German, which is higher than ByteNet (23.75), GNMT + RL (24.6), ConvS2S (25.16), and MoE (26.03 slightly higher but with much larger training cost). On English-to-French, the Transformer also outperformed several models by achieving BLEU scores higher than 40 (Deep-Att + PosUnk: 39.2, GNMT + RL: 39.92, ConvS2S: 40.46, MoE: 40.56) but with a significantly lower training cost (FLOPs). The Deep-Att + PosUnk Ensemble scored 40.4 but with much higher training cost (8.0·10^20 FLOPs).

In summary, the Transformer not only outperformed prior models in BLEU scores but also did so at a fraction of the training cost (FLOPs) (Source: Attention_Is_All_You_Need.pdf, Pages 8-9).

3. [Methods Analyst]
The provided excerpts do not explicitly state a problem with using self-attention. However, from the context in "Attention Is All You Need" (Page 6), the authors engage in a comparison of self-attention with recurrent and convolutional layers, indicating a consideration of various aspects and potential challenges, which is likely detailed further in that section.

Typically, one known problem with self-attention is its quadratic complexity in sequence length, as each position attends to every other position, making 
it computationally expensive for very long sequences. This issue is often discussed in contrast with recurrent and convolutional approaches which have different computational characteristics.

Since the excerpts mention the use of sinusoidal positional encodings helping with extrapolation to longer sequence lengths, it suggests self-attention has limitations related to sequence length handling or generalization.

For a precise statement of a problem with self-attention according to the paper, one must refer to Section 4 on Page 6. The text likely details the computational inefficiency or scalability issues of self-attention compared to other layers.

(Source: Attention_Is_All_You_Need.pdf, Page 6)

4.  [General Synthesizer]
"System 0" in the article refers to a concept of an AI system that operates prior to human cognitive processes, playing a foundational and pervasive role in modern cognition. It provides a view limited to the subset of information on which it has been trained and could be useful if it transparently lists its information sources along with warnings about possible missing data. The term emphasizes its role as pre-cognition, existing before the traditional "fast and slow" thinking systems. There is some debate about the name, with suggestions like "System Beta," but "System 0" is chosen to highlight its foundational nature in cognition (Source: Can_ai_expand_the_human_mind.pdf, Pages 2-3).

5. [General Synthesizer]
 AI can help people make decisions by sifting through large amounts of information that humans may not have the time or mental capacity to handle. It can 
detect subtle patterns that humans might miss and identify variables that make certain options more suitable for a particular user, even if the user is unaware of these factors. This allows AI agents to offer more tailored and effective decision support (Source: Can_ai_expand_the_human_mind.pdf, Page 1).

6. [General Synthesizer]
 One risk of relying too much on AI is that it may influence people’s decisions through subtle tricks used by companies and advertisers to nudge them toward particular choices, which could undermine autonomous decision-making (Source: Can_ai_expand_the_human_mind.pdf, Page 2).

7. [Methods Analyst]
 The paper discusses using multiple agents to improve answer accuracy by enabling multiple agents to "debate" the final answer for arithmetic tasks. This 
multi-agent framework contrasts with the single-agent approach used in earlier methods such as (Wei et al., 2022). By allowing agents to collaboratively deliberate or debate, the paper reports performance improvements in problem-solving accuracy.

Further, it references LLM-Debate (Du et al., 2023), which shows a preliminary curve indicating that the accuracy of solving a math problem increases with the number of agents involved (More_Agents_Is_All_You_Need.pdf, Page 1). Similarly, CoT-SC (Wang et al., 2023b) produces multiple thought chains instead 
of just one, selecting the most self-consistent chain as the final answer, which also improves reasoning performance over a single chain-of-thought approach.

Thus, the method to improve answers fundamentally relies on generating multiple candidate solutions or reasoning chains through multiple agents and then selecting or debating the best final answer, leveraging multi-agent collaboration to enhance accuracy.

(Source: More_Agents_Is_All_You_Need.pdf, Pages 1, 13)

8. [Results Extractor]
 Increasing the number of agents generally leads to a performance improvement. Specifically, the accuracy of solving problems increases as more agents are added. This scaling property is observed across multiple studies, indicating that putting multiple agents together can boost effectiveness, particularly 
through a process like sampling-and-voting. However, there is also a consideration that increasing the number of agents raises the computational cost, though optimization can mitigate this (Source: More_Agents_Is_All_You_Need.pdf, Pages 1, 2, 12).

9. [Methods Analyst]
 One downside of using many agents is the escalating costs associated with multiple LLM calls. Increasing the number of agents requires more computational resources and hence greater expense, which is a common challenge in methods involving multiple language model calls (Source: More_Agents_Is_All_You_Need.pdf, Page 12).


# Any additional comments (optional)


