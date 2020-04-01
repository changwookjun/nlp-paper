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
- [Evolution of transfer learning in natural language processing](https://arxiv.org/abs/1910.07370)
- [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)
- [A Survey on Contextual Embeddings](https://arxiv.org/abs/2003.07278)

### Machine Translation    
### Downstream task    
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

