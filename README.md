# NLP Paper
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

List the papers you need to study natural language processing.  

## Contents
* [Bert Series](#Bert-Series)   
* [Transformer Series](#Transformer-Series)  
* [Transfer Learning](#Transfer-Learning)  
* [Text Summarization](#Text-Summarization)  
* [Sentiment Analysis](#Sentiment-Analysis)  
* [Question Answering](#Question-Answering)  
* [Machine Translation](#Machine-Translation)
* [Surver paper](#survey-paper)  
* [Downstream task](#downstream-task) 
* [Generation](#generation) 
* [Quality evaluator](#quality-evaluator) 
* [Modification (multi-task, masking strategy, etc.)](#modification-multi-task-masking-strategy-etc) 
* [Probe](#probe) 
* [Multi-lingual](#multi-lingual) 
* [Other than English models](#other-than-english-models) 
* [Domain specific](#domain-specific) 
* [Multi-modal](#multi-modal) 
* [Model compression](#model-compression) 
* [Misc](#misc) 


### Bert Series
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  - NAACL 2019)](https://arxiv.org/abs/1810.04805)  
* [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding - arXiv 2019)](https://arxiv.org/abs/1907.12412)  
* [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding - arXiv 2019)](https://arxiv.org/abs/1908.04577)  
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach  - arXiv 2019)](https://arxiv.org/abs/1907.11692)  
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations  - arXiv 2019)](https://arxiv.org/abs/1909.11942)  
* [Multi-Task Deep Neural Networks for Natural Language Understanding  - arXiv 2019)](https://arxiv.org/abs/1901.11504)  

### Transformer Series
* [Attention Is All You Need - arXiv 2017)](https://arxiv.org/abs/1706.03762)  
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context - arXiv 2019)](https://arxiv.org/abs/1901.02860)  
* [Universal Transformers - ICLR 2019)](https://arxiv.org/abs/1807.03819) 
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer - arXiv 2019)](https://arxiv.org/abs/1910.10683) 
* [Reformer: The Efficient Transformer - ICLR 2020)](https://arxiv.org/abs/2001.04451) 


### Transfer Learning
* [Deep contextualized word representations - NAACL 2018)](https://arxiv.org/abs/1802.05365)  
* [Universal Language Model Fine-tuning for Text Classification  - ACL 2018)](https://arxiv.org/abs/1801.06146)  
* [Improving Language Understanding by Generative Pre-Training  - Alec Radford)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  - NAACL 2019)](https://arxiv.org/abs/1810.04805)  
* [Cloze-driven Pretraining of Self-attention Networks - arXiv 2019)](https://arxiv.org/abs/1903.07785)  
* [Unified Language Model Pre-training for Natural Language Understanding and Generation - arXiv 2019)](https://arxiv.org/abs/1905.03197)  
* [MASS: Masked Sequence to Sequence Pre-training for Language Generation - ICML 2019)](https://arxiv.org/abs/1905.02450)  



### Text Summarization
* [Positional Encoding to Control Output Sequence Length - Sho Takase(2019)](https://arxiv.org/pdf/1904.07418.pdf)  
* [Fine-tune BERT for Extractive Summarization - Yang Liu(2019)](https://arxiv.org/pdf/1903.10318.pdf)  
* [Language Models are Unsupervised Multitask Learners - Alec Radford(2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)   
* [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss - Wan-Ting Hsu(2018)](https://arxiv.org/pdf/1805.06266.pdf)   
* [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents - Arman Cohan(2018)](https://arxiv.org/pdf/1801.10198.pdf)   
* [GENERATING WIKIPEDIA BY SUMMARIZING LONG SEQUENCES - Peter J. Liu(2018)](https://arxiv.org/pdf/1801.10198.pdf)   
* [Get To The Point: Summarization with Pointer-Generator Networks - Abigail See(2017)](https://arxiv.org/pdf/1704.04368.pdf) * [A Neural Attention Model for Sentence Summarization - Alexander M. Rush(2015)](https://www.aclweb.org/anthology/D15-1044)   


### Sentiment Analysis
* [Multi-Task Deep Neural Networks for Natural Language Understanding - Xiaodong Liu(2019)](https://arxiv.org/pdf/1901.11504.pdf)  
* [Aspect-level Sentiment Analysis using AS-Capsules - Yequan Wang(2019)](http://coai.cs.tsinghua.edu.cn/hml/media/files/WWW19WangY.pdf) 
* [On the Role of Text Preprocessing in Neural Network Architectures:
An Evaluation Study on Text Categorization and Sentiment Analysis - Jose Camacho-Collados(2018)](https://arxiv.org/pdf/1704.01444.pdf) 
* [Learned in Translation: Contextualized Word Vectors - Bryan McCann(2018)](https://arxiv.org/pdf/1708.00107.pdf) 
* [Universal Language Model Fine-tuning for Text Classification - Jeremy Howard(2018)](https://arxiv.org/pdf/1801.06146.pdf) 
* [Convolutional Neural Networks with Recurrent Neural Filters - Yi Yang(2018)](https://aclweb.org/anthology/D18-1109) 
* [Information Aggregation via Dynamic Routing for Sequence Encoding - Jingjing Gong(2018)](https://arxiv.org/pdf/1806.01501.pdf) 
* [Learning to Generate Reviews and Discovering Sentiment - Alec Radford(2017)](https://arxiv.org/pdf/1704.01444.pdf) 
* [A Structured Self-attentive Sentence Embedding - Zhouhan Lin(2017)](https://arxiv.org/pdf/1703.03130.pdf) 

### Question Answering  
* [Language Models are Unsupervised Multitask Learners - Alec Radford(2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
* [Improving Language Understanding by Generative Pre-Training - Alec Radford(2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
* [Bidirectional Attention Flow for Machine Comprehension - Minjoon Seo(2018)](https://arxiv.org/pdf/1611.01603.pdf) 
* [Reinforced Mnemonic Reader for Machine Reading Comprehension - Minghao Hu(2017)](https://arxiv.org/pdf/1705.02798.pdf)  
* [Neural Variational Inference for Text Processing - Yishu Miao(2015)](https://arxiv.org/pdf/1511.06038.pdf)  

### Machine Translation    
* [The Evolved Transformer - David R. So(2019)](https://arxiv.org/pdf/1901.11117.pdf)  

### Surver paper    
* [Evolution of transfer learning in natural language processing](https://arxiv.org/abs/1910.07370)
* [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)
* [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278)

### Downstream task    
* [A BERT Baseline for the Natural Questions](https://arxiv.org/abs/1901.08634)
* [MultiQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension](https://arxiv.org/abs/1905.13453) (ACL2019)
* [Unsupervised Domain Adaptation on Reading Comprehension](https://arxiv.org/abs/1911.06137)
* [BERTQA -- Attention on Steroids](https://arxiv.org/abs/1912.10435)
* [A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning](https://arxiv.org/abs/1908.05514) (EMNLP2019)
* [SDNet: Contextualized Attention-based Deep Network for Conversational Question Answering](https://arxiv.org/abs/1812.03593)
* [Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)
* [Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)
* [Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering](https://arxiv.org/abs/1909.07598) (EMNLP2019 WS)
* [End-to-End Open-Domain Question Answering with BERTserini](https://arxiv.org/abs/1902.01718) (NAALC2019)
* [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300) (ACL2019)
* [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167) (EMNLP2019)
* [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470) (ICLR2020)
* [Learning to Ask Unanswerable Questions for Machine Reading Comprehension](https://arxiv.org/abs/1906.06045) (ACL2019)
* [Unsupervised Question Answering by Cloze Translation](https://arxiv.org/abs/1906.04980) (ACL2019)
* [Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation](https://arxiv.org/abs/1908.04942)
* [A Recurrent BERT-based Model for Question Generation](https://www.aclweb.org/anthology/D19-5821/) (EMNLP2019 WS)
* [Learning to Answer by Learning to Ask: Getting the Best of GPT-2 and BERT Worlds](https://arxiv.org/abs/1911.02365)
* [Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension](https://www.aclweb.org/anthology/papers/P/P19/P19-1226/) (ACL2019)
* [Incorporating Relation Knowledge into Commonsense Reading Comprehension with Multi-task Learning](https://arxiv.org/abs/1908.04530) (CIKM2019)
* [SG-Net: Syntax-Guided Machine Reading Comprehension](https://arxiv.org/abs/1908.05147)
* [MMM: Multi-stage Multi-task Learning for Multi-choice Reading Comprehension](https://arxiv.org/abs/1910.00458)
* [Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning](https://arxiv.org/abs/1909.00277) (EMNLP2019)
* [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://arxiv.org/abs/2002.04326) (ICLR2020)
* [Robust Reading Comprehension with Linguistic Constraints via Posterior Regularization](https://arxiv.org/abs/1911.06948)
* [BAS: An Answer Selection Method Using BERT Language Model](https://arxiv.org/abs/1911.01528)
* [Beat the AI: Investigating Adversarial Human Annotations for Reading Comprehension](https://arxiv.org/abs/2002.00293)
* [A Simple but Effective Method to Incorporate Multi-turn Context with BERT for Conversational Machine Comprehension](https://arxiv.org/abs/1905.12848) (ACL2019 WS)
* [FlowDelta: Modeling Flow Information Gain in Reasoning for Conversational Machine Comprehension](https://arxiv.org/abs/1908.05117) (ACL2019 WS)
* [BERT with History Answer Embedding for Conversational Question Answering](https://arxiv.org/abs/1905.05412) (SIGIR2019)
* [GraphFlow: Exploiting Conversation Flow with Graph Neural Networks for Conversational Machine Comprehension](https://arxiv.org/abs/1908.00059) (ICML2019 WS)
* [Beyond English-only Reading Comprehension: Experiments in Zero-Shot Multilingual Transfer for Bulgarian](https://arxiv.org/abs/1908.01519) (RANLP2019)
* [XQA: A Cross-lingual Open-domain Question Answering Dataset](https://www.aclweb.org/anthology/P19-1227/) (ACL2019)
* [Cross-Lingual Machine Reading Comprehension](https://arxiv.org/abs/1909.00361) (EMNLP2019)
* [Zero-shot Reading Comprehension by Cross-lingual Transfer Learning with Multi-lingual Language Representation Model](https://arxiv.org/abs/1909.09587)
* [Multilingual Question Answering from Formatted Text applied to Conversational Agents](https://arxiv.org/abs/1910.04659)
* [BiPaR: A Bilingual Parallel Dataset for Multilingual and Cross-lingual Reading Comprehension on Novels](https://arxiv.org/abs/1910.05040) (EMNLP2019)
* [MLQA: Evaluating Cross-lingual Extractive Question Answering](https://arxiv.org/abs/1910.07475)
* [Investigating Prior Knowledge for Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679) (TACL)
* [SberQuAD - Russian Reading Comprehension Dataset: Description and Analysis](https://arxiv.org/abs/1912.09723)
* [Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://arxiv.org/abs/1909.00109) (EMNLP2019)
* [BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1907.03040) (Interspeech2019)
* [Dialog State Tracking: A Neural Reading Comprehension Approach](https://arxiv.org/abs/1908.01946) 
* [A Simple but Effective BERT Model for Dialog State Tracking on Resource-Limited Systems](https://arxiv.org/abs/1910.12995) (ICASSP2020)
* [Fine-Tuning BERT for Schema-Guided Zero-Shot Dialogue State Tracking](https://arxiv.org/abs/2002.00181)
* [Goal-Oriented Multi-Task BERT-Based Dialogue State Tracker](https://arxiv.org/abs/2002.02450)
* [Domain Adaptive Training BERT for Response Selection](https://arxiv.org/abs/1908.04812)
* [BERT Goes to Law School: Quantifying the Competitive Advantage of Access to Large Legal Corpora in Contract Understanding](https://arxiv.org/abs/1911.00473)


### Generation    
### Quality evaluator    
### Modification (multi-task, masking strategy, etc.)    
### Probe    
### Multi-lingual    
### Other than English models    
### Domain specific    
### Multi-modal    
### Model compression    
### Misc       

