# NLP Paper
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

natural language processing paper list

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
   * [QA MC Dialogue](#QA-MC-Dialogue) 
   * [Slot filling](#Slot-filling)    
   * [Analysis](#Analysis) 
   * [Word segmentation parsing NER](#Word-segmentation-parsing-NER)    
   * [Pronoun coreference resolution](#Pronoun-coreference-resolution) 
   * [Word sense disambiguation](#Word-sense-disambiguation) 
   * [Sentiment analysis](#Sentiment-analysis) 
   * [Relation extraction](#Relation-extraction)    
   * [Knowledge base](#Knowledge-base)     
   * [Text classification](#Text-classification)         
   * [WSC WNLI NLI](#WSC-WNLI-NLI) 
   * [Commonsense](#Commonsense) 
   * [Extractive summarization](#Extractive-summarization)
   * [IR](#IR)   
* [Generation](#generation) 
* [Quality evaluator](#quality-evaluator) 
* [Modification (multi-task, masking strategy, etc.)](#modification-multi-task-masking-strategy-etc) 
* [Probe](#probe) 
* [Multi-lingual](#multi-lingual) 
* [Other than English models](#other-than-english-models) 
* [Domain specific](#domain-specific) 
* [Multi-modal](#multi-modal) 
* [Model compression](#model-compression)
* [LLM](#LLM) 
* [Misc](#misc) 

### Bert Series
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  - NAACL 2019)](https://arxiv.org/abs/1810.04805)  
* [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding - arXiv 2019)](https://arxiv.org/abs/1907.12412)  
* [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding - arXiv 2019)](https://arxiv.org/abs/1908.04577)  
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach  - arXiv 2019)](https://arxiv.org/abs/1907.11692)  
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations  - arXiv 2019)](https://arxiv.org/abs/1909.11942)  
* [Multi-Task Deep Neural Networks for Natural Language Understanding  - arXiv 2019)](https://arxiv.org/abs/1901.11504)  
* [What does BERT learn about the structure of language?](https://hal.inria.fr/hal-02131630/document) (ACL2019)
* [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) (ACL2019) [[github](https://github.com/lena-voita/the-story-of-heads)]
* [Open Sesame: Getting Inside BERT's Linguistic Knowledge](https://arxiv.org/abs/1906.01698) (ACL2019 WS)
* [Analyzing the Structure of Attention in a Transformer Language Model](https://arxiv.org/abs/1906.04284) (ACL2019 WS)
* [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341) (ACL2019 WS)
* [Do Attention Heads in BERT Track Syntactic Dependencies?](https://arxiv.org/abs/1911.12246)
* [Blackbox meets blackbox: Representational Similarity and Stability Analysis of Neural Language Models and Brains](https://arxiv.org/abs/1906.01539) (ACL2019 WS)
* [Inducing Syntactic Trees from BERT Representations](https://arxiv.org/abs/1906.11511) (ACL2019 WS)
* [A Multiscale Visualization of Attention in the Transformer Model](https://arxiv.org/abs/1906.05714) (ACL2019 Demo)
* [Visualizing and Measuring the Geometry of BERT](https://arxiv.org/abs/1906.02715)
* [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://arxiv.org/abs/1909.00512) (EMNLP2019) 
* [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) (NeurIPS2019)
* [On the Validity of Self-Attention as Explanation in Transformer Models](https://arxiv.org/abs/1908.04211)
* [Visualizing and Understanding the Effectiveness of BERT](https://arxiv.org/abs/1908.05620) (EMNLP2019)
* [Attention Interpretability Across NLP Tasks](https://arxiv.org/abs/1909.11218)
* [Revealing the Dark Secrets of BERT](https://arxiv.org/abs/1908.08593) (EMNLP2019)
* [Investigating BERT's Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597) (EMNLP2019)
* [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://arxiv.org/abs/1909.01380) (EMNLP2019) 
* [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/abs/2002.12327)
* [Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://arxiv.org/abs/1909.07940) (EMNLP2019)
* [How Does BERT Answer Questions? A Layer-Wise Analysis of Transformer Representations](https://arxiv.org/abs/1909.04925) (CIKM2019)
* [Whatcha lookin' at? DeepLIFTing BERT's Attention in Question Answering](https://arxiv.org/abs/1910.06431)
* [What does BERT Learn from Multiple-Choice Reading Comprehension Datasets?](https://arxiv.org/abs/1910.12391)
* [Calibration of Pre-trained Transformers](https://arxiv.org/abs/2003.07892)
* [exBERT: A Visual Analysis Tool to Explore Learned Representations in Transformers Models](https://arxiv.org/abs/1910.05276) [[github](https://github.com/bhoov/exbert)]  
* [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/pdf/2004.02984.pdf) [[github](https://github.com/google-research/google-research/tree/master/mobilebert)]   
* [Measuring and Reducing Gendered Correlations in Pre-trained Models](https://arxiv.org/pdf/2010.06032.pdf)  


### Transformer Series
* [Attention Is All You Need - arXiv 2017)](https://arxiv.org/abs/1706.03762)  
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context - arXiv 2019)](https://arxiv.org/abs/1901.02860)  
* [Universal Transformers - ICLR 2019)](https://arxiv.org/abs/1807.03819) 
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer - arXiv 2019)](https://arxiv.org/abs/1910.10683) 
* [Reformer: The Efficient Transformer - ICLR 2020)](https://arxiv.org/abs/2001.04451) 
* [Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799) (ACL2019)
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) (ACL2019) [[github](https://github.com/kimiyoung/transformer-xl)]
* [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
* [Adaptively Sparse Transformers](https://arxiv.org/abs/1909.00015) (EMNLP2019)
* [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)
* [The Evolved Transformer](https://arxiv.org/abs/1901.11117) (ICML2019)
* [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (ICLR2020) [[github](https://github.com/google/trax/tree/master/trax/models/reformer)]
* [GRET: Global Representation Enhanced Transformer](https://arxiv.org/abs/2002.10101) (AAAI2020)
* [Transformer on a Diet](https://arxiv.org/abs/2002.06170) [[github](https://github.com/cgraywang/transformer-on-diet)]
* [Efficient Content-Based Sparse Attention with Routing Transformers](https://openreview.net/forum?id=B1gjs6EtDr)
* [BP-Transformer: Modelling Long-Range Context via Binary Partitioning](https://arxiv.org/abs/1911.04070)  
* [Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf)   
* [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf)  
* [UnifiedQA: Crossing Format Boundaries With a Single QA System](https://arxiv.org/pdf/2005.00700.pdf) [[github](https://github.com/allenai/unifiedqa)]  
* [Big Bird: Transformers for Longer Sequences](https://arxiv.org/pdf/2007.14062.pdf) 


### Transfer Learning
* [Deep contextualized word representations - NAACL 2018)](https://arxiv.org/abs/1802.05365)  
* [Universal Language Model Fine-tuning for Text Classification  - ACL 2018)](https://arxiv.org/abs/1801.06146)  
* [Improving Language Understanding by Generative Pre-Training  - Alec Radford)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  - NAACL 2019)](https://arxiv.org/abs/1810.04805)  
* [Cloze-driven Pretraining of Self-attention Networks - arXiv 2019)](https://arxiv.org/abs/1903.07785)  
* [Unified Language Model Pre-training for Natural Language Understanding and Generation - arXiv 2019)](https://arxiv.org/abs/1905.03197)  
* [MASS: Masked Sequence to Sequence Pre-training for Language Generation - ICML 2019)](https://arxiv.org/abs/1905.02450)  
* [MPNet: Masked and Permuted Pre-training for Language Understanding)](https://arxiv.org/pdf/2004.09297.pdf)[[github](https://github.com/microsoft/MPNet)]    
* [UNILMv2: Pseudo-Masked Language Models for
Unified Language Model Pre-Training)](https://arxiv.org/pdf/2002.12804.pdf)[[github](https://github.com/microsoft/unilm)]    

### Text Summarization
* [Positional Encoding to Control Output Sequence Length - Sho Takase(2019)](https://arxiv.org/pdf/1904.07418.pdf)  
* [Fine-tune BERT for Extractive Summarization - Yang Liu(2019)](https://arxiv.org/pdf/1903.10318.pdf)  
* [Language Models are Unsupervised Multitask Learners - Alec Radford(2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)   
* [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss - Wan-Ting Hsu(2018)](https://arxiv.org/pdf/1805.06266.pdf)   
* [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents - Arman Cohan(2018)](https://arxiv.org/pdf/1804.05685.pdf)   
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
#### QA MC Dialogue
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
* [A BERT Baseline for the Natural Questions](https://arxiv.org/abs/1911.00473) 
* [Wizard of Wikipedia](https://arxiv.org/pdf/1811.01241.pdf) 

#### Slot filling
* [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
* [Multi-lingual Intent Detection and Slot Filling in a Joint BERT-based Model](https://arxiv.org/abs/1907.02884)
* [A Comparison of Deep Learning Methods for Language Understanding](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1262.html) (Interspeech2019)

#### Analysis
* [Fine-grained Information Status Classification Using Discourse Context-Aware Self-Attention](https://arxiv.org/abs/1908.04755)
* [Neural Aspect and Opinion Term Extraction with Mined Rules as Weak Supervision](https://arxiv.org/abs/1907.03750) (ACL2019) 
* [BERT-based Lexical Substitution](https://www.aclweb.org/anthology/P19-1328) (ACL2019) 
* [Assessing BERT’s Syntactic Abilities](https://arxiv.org/abs/1901.05287)
* [Does BERT agree? Evaluating knowledge of structure dependence through agreement relations](https://arxiv.org/abs/1908.09892)
* [Simple BERT Models for Relation Extraction and Semantic Role Labeling](https://arxiv.org/abs/1904.05255)
* [LIMIT-BERT : Linguistic Informed Multi-Task BERT](https://arxiv.org/abs/1910.14296)
* [A Simple BERT-Based Approach for Lexical Simplification](https://arxiv.org/abs/1907.06226)
* [Multi-headed Architecture Based on BERT for Grammatical Errors Correction](https://www.aclweb.org/anthology/papers/W/W19/W19-4426/) (ACL2019 WS) 
* [Towards Minimal Supervision BERT-based Grammar Error Correction](https://arxiv.org/abs/2001.03521)
* [BERT-Based Arabic Social Media Author Profiling](https://arxiv.org/abs/1909.04181)
* [Sentence-Level BERT and Multi-Task Learning of Age and Gender in Social Media](https://arxiv.org/abs/1911.00637)
* [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)
* [NegBERT: A Transfer Learning Approach for Negation Detection and Scope Resolution](https://arxiv.org/abs/1911.04211)
* [xSLUE: A Benchmark and Analysis Platform for Cross-Style Language Understanding and Evaluation](https://arxiv.org/abs/1911.03663)
* [TabFact: A Large-scale Dataset for Table-based Fact Verification](https://arxiv.org/abs/1909.02164)
* [Rapid Adaptation of BERT for Information Extraction on Domain-Specific Business Documents](https://arxiv.org/abs/2002.01861)
* [LAMBERT: Layout-Aware language Modeling using BERT for information extraction](https://arxiv.org/abs/2002.08087)
* [Keyphrase Extraction from Scholarly Articles as Sequence Labeling using Contextualized Embeddings](https://arxiv.org/abs/1910.08840) (ECIR2020) [[github](https://github.com/midas-research/keyphrase-extraction-as-sequence-labeling-data)]
* [Keyphrase Extraction with Span-based Feature Representations](https://arxiv.org/abs/2002.05407)
* [What do you mean, BERT? Assessing BERT as a Distributional Semantics Model](https://arxiv.org/abs/1911.05758)

#### Word segmentation parsing NER  
* [BERT Meets Chinese Word Segmentation](https://arxiv.org/abs/1909.09292)
* [Toward Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning](https://arxiv.org/abs/1903.04190)
* [Establishing Strong Baselines for the New Decade: Sequence Tagging, Syntactic and Semantic Parsing with BERT](https://arxiv.org/abs/1908.04943)
* [Evaluating Contextualized Embeddings on 54 Languages in POS Tagging, Lemmatization and Dependency Parsing](https://arxiv.org/abs/1908.07448) 
* [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204)
* [Deep Contextualized Word Embeddings in Transition-Based and Graph-Based Dependency Parsing -- A Tale of Two Parsers Revisited](https://arxiv.org/abs/1908.07397) (EMNLP2019)
* [Is POS Tagging Necessary or Even Helpful for Neural Dependency Parsing?](https://arxiv.org/abs/2003.03204)
* [Parsing as Pretraining](https://arxiv.org/abs/2002.01685) (AAAI2020)
* [Cross-Lingual BERT Transformation for Zero-Shot Dependency Parsing](https://arxiv.org/abs/1909.06775)
* [Recursive Non-Autoregressive Graph-to-Graph Transformer for Dependency Parsing with Iterative Refinement](https://arxiv.org/abs/2003.13118)
* [Named Entity Recognition -- Is there a glass ceiling?](https://arxiv.org/abs/1910.02403) (CoNLL2019)
* [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)
* [Training Compact Models for Low Resource Entity Tagging using Pre-trained Language Models](https://arxiv.org/abs/1910.06294)
* [Robust Named Entity Recognition with Truecasing Pretraining](https://arxiv.org/abs/1912.07095) (AAAI2020)
* [LTP: A New Active Learning Strategy for Bert-CRF Based Named Entity Recognition](https://arxiv.org/abs/2001.02524)
* [MT-BioNER: Multi-task Learning for Biomedical Named Entity Recognition using Deep Bidirectional Transformers](https://arxiv.org/abs/2001.08904)
* [Portuguese Named Entity Recognition using BERT-CRF](https://arxiv.org/abs/1909.10649)
* [Towards Lingua Franca Named Entity Recognition with BERT](https://arxiv.org/abs/1912.01389)

#### Pronoun coreference resolution
* [Resolving Gendered Ambiguous Pronouns with BERT](https://arxiv.org/abs/1906.01161) (ACL2019 WS)
* [Anonymized BERT: An Augmentation Approach to the Gendered Pronoun Resolution Challenge](https://arxiv.org/abs/1905.01780) (ACL2019 WS)
* [Gendered Pronoun Resolution using BERT and an extractive question answering formulation](https://arxiv.org/abs/1906.03695) (ACL2019 WS)
* [MSnet: A BERT-based Network for Gendered Pronoun Resolution](https://arxiv.org/abs/1908.00308) (ACL2019 WS)
* [Fill the GAP: Exploiting BERT for Pronoun Resolution](https://www.aclweb.org/anthology/papers/W/W19/W19-3815/) (ACL2019 WS)
* [On GAP Coreference Resolution Shared Task: Insights from the 3rd Place Solution](https://www.aclweb.org/anthology/W19-3816/) (ACL2019 WS)
* [Look Again at the Syntax: Relational Graph Convolutional Network for Gendered Ambiguous Pronoun Resolution](https://arxiv.org/abs/1905.08868) (ACL2019 WS)
* [BERT Masked Language Modeling for Co-reference Resolution](https://www.aclweb.org/anthology/papers/W/W19/W19-3811/) (ACL2019 WS)
* [Coreference Resolution with Entity Equalization](https://www.aclweb.org/anthology/P19-1066/) (ACL2019)
* [BERT for Coreference Resolution: Baselines and Analysis](https://arxiv.org/abs/1908.09091) (EMNLP2019) [[github](https://github.com/mandarjoshi90/coref)]
* [WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://arxiv.org/abs/1908.08025) (EMNLP2019)
* [Ellipsis and Coreference Resolution as Question Answering](https://arxiv.org/abs/1908.11141)
* [Coreference Resolution as Query-based Span Prediction](https://arxiv.org/abs/1911.01746)
* [Multi-task Learning Based Neural Bridging Reference Resolution](https://arxiv.org/abs/2003.03666)

#### Word sense disambiguation  
* [GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://arxiv.org/abs/1908.07245) (EMNLP2019)
* [Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](https://arxiv.org/abs/1910.00194)  (EMNLP2019)
* [Using BERT for Word Sense Disambiguation](https://arxiv.org/abs/1909.08358)
* [Language Modelling Makes Sense: Propagating Representations through WordNet for Full-Coverage Word Sense Disambiguation](https://www.aclweb.org/anthology/P19-1569.pdf) (ACL2019)
* [Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings](https://arxiv.org/abs/1909.10430) (KONVENS2019)

#### Sentiment analysis  
* [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588) (NAACL2019)
* [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/abs/1904.02232) (NAACL2019)
* [Exploiting BERT for End-to-End Aspect-based Sentiment Analysis](https://arxiv.org/abs/1910.00883) (EMNLP2019 WS)
* [Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification](https://arxiv.org/abs/1908.11860) 
* [An Investigation of Transfer Learning-Based Sentiment Analysis in Japanese](https://arxiv.org/abs/1905.09642) (ACL2019)
* ["Mask and Infill" : Applying Masked Language Model to Sentiment Transfer](https://arxiv.org/abs/1908.08039)
* [Adversarial Training for Aspect-Based Sentiment Analysis with BERT](https://arxiv.org/abs/2001.11316)
* [Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis and Natural Language Inference](https://arxiv.org/abs/2002.04815)

#### Relation extraction
* [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/abs/1906.03158) (ACL2019)
* [BERT-Based Multi-Head Selection for Joint Entity-Relation Extraction](https://arxiv.org/abs/1908.05908) (NLPCC2019)
* [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)
* [Span-based Joint Entity and Relation Extraction with Transformer Pre-training](https://arxiv.org/abs/1909.07755)
* [Fine-tune Bert for DocRED with Two-step Process](https://arxiv.org/abs/1909.11898)
* [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://arxiv.org/abs/1909.03546) (EMNLP2019)

#### Knowledge base
* [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/abs/1909.03193)
* [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066) (EMNLP2019) [[github](https://github.com/facebookresearch/LAMA)]
* [BERT is Not a Knowledge Base (Yet): Factual Knowledge vs. Name-Based Reasoning in Unsupervised QA](https://arxiv.org/abs/1911.03681)
* [Inducing Relational Knowledge from BERT](https://arxiv.org/abs/1911.12753) (AAAI2020)
* [Latent Relation Language Models](https://arxiv.org/abs/1908.07690) (AAAI2020)
* [Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model](https://openreview.net/forum?id=BJlzm64tDH) (ICLR2020)
* [Zero-shot Entity Linking with Dense Entity Retrieval](https://arxiv.org/abs/1911.03814)
* [Investigating Entity Knowledge in BERT with Simple Neural End-To-End Entity Linking](https://www.aclweb.org/anthology/K19-1063/) (CoNLL2019)
* [Improving Entity Linking by Modeling Latent Entity Type Information](https://arxiv.org/abs/2001.01447) (AAAI2020)
* [PEL-BERT: A Joint Model for Protocol Entity Linking](https://arxiv.org/abs/2002.00744)
* [How Can We Know What Language Models Know?](https://arxiv.org/abs/1911.12543)
* [REALM: Retrieval-Augmented Language Model Pre-Training](https://kentonl.com/pub/gltpc.2020.pdf)


#### Text classification
* [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583)
* [X-BERT: eXtreme Multi-label Text Classification with BERT](https://arxiv.org/abs/1905.02331)
* [DocBERT: BERT for Document Classification](https://arxiv.org/abs/1904.08398)
* [Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402)
* [Classification and Clustering of Arguments with Contextualized Word Embeddings](https://arxiv.org/abs/1906.09821) (ACL2019)
* [BERT for Evidence Retrieval and Claim Verification](https://arxiv.org/abs/1910.02655)
* [Stacked DeBERT: All Attention in Incomplete Data for Text Classification](https://arxiv.org/abs/2001.00137)
* [Cost-Sensitive BERT for Generalisable Sentence Classification with Imbalanced Data](https://arxiv.org/abs/2003.11563)

#### WSC WNLI NLI
* [Exploring Unsupervised Pretraining and Sentence Structure Modelling for Winograd Schema Challenge](https://arxiv.org/abs/1904.09705)
* [A Surprisingly Robust Trick for the Winograd Schema Challenge](https://arxiv.org/abs/1905.06290)
* [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641) (AAAI2020)
* [Improving Natural Language Inference with a Pretrained Parser](https://arxiv.org/abs/1909.08217)
* [Adversarial NLI: A New Benchmark for Natural Language Understanding](https://arxiv.org/abs/1910.14599)
* [Adversarial Analysis of Natural Language Inference Systems](https://arxiv.org/abs/1912.03441) (ICSC2020)
* [HypoNLI: Exploring the Artificial Patterns of Hypothesis-only Bias in Natural Language Inference](https://arxiv.org/abs/2003.02756) (LREC2020)
* [Evaluating BERT for natural language inference: A case study on the CommitmentBank](https://www.aclweb.org/anthology/D19-1630/) (EMNLP2019)

#### Commonsense
* [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937) (NAACL2019)
* [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) (ACL2019) [[website](https://rowanzellers.com/hellaswag/)]
* [Story Ending Prediction by Transferable BERT](https://arxiv.org/abs/1905.07504) (IJCAI2019)
* [Explain Yourself! Leveraging Language Models for Commonsense Reasoning](https://arxiv.org/abs/1906.02361) (ACL2019)
* [Align, Mask and Select: A Simple Method for Incorporating Commonsense Knowledge into Language Representation Models](https://arxiv.org/abs/1908.06725)
* [Informing Unsupervised Pretraining with External Linguistic Knowledge](https://arxiv.org/abs/1909.02339)
* [Commonsense Knowledge + BERT for Level 2 Reading Comprehension Ability Test](https://arxiv.org/abs/1909.03415)
* [BIG MOOD: Relating Transformers to Explicit Commonsense Knowledge](https://arxiv.org/abs/1910.07713)
* [Commonsense Knowledge Mining from Pretrained Models](https://arxiv.org/abs/1909.00505) (EMNLP2019)
* [KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning](https://arxiv.org/abs/1909.02151) (EMNLP2019)
* [Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations](https://www.aclweb.org/anthology/D19-6001/) (EMNLP2019 WS)
* [Do Massively Pretrained Language Models Make Better Storytellers?](https://arxiv.org/abs/1909.10705) (CoNLL2019)
* [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641v1) (AAAI2020)
* [Evaluating Commonsense in Pre-trained Language Models](https://arxiv.org/abs/1911.11931) (AAAI2020)
* [Why Do Masked Neural Language Models Still Need Common Sense Knowledge?](https://arxiv.org/abs/1911.03024)
* [Do Neural Language Representations Learn Physical Commonsense?](https://arxiv.org/abs/1908.02899) (CogSci2019)

#### Extractive summarization
* [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/abs/1905.06566) (ACL2019)
* [Deleter: Leveraging BERT to Perform Unsupervised Successive Text Compression](https://arxiv.org/abs/1909.03223)
* [Discourse-Aware Neural Extractive Model for Text Summarization](https://arxiv.org/abs/1910.14142)  
* [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf)[[github](https://github.com/google-research/pegasus)]    
* [Discourse-Aware Neural Extractive Text Summarization](https://arxiv.org/pdf/1910.14142.pdf)[[github](https://github.com/jiacheng-xu/DiscoBERT)]      


#### IR
* [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
* [Investigating the Successes and Failures of BERT for Passage Re-Ranking](https://arxiv.org/abs/1905.01758)
* [Understanding the Behaviors of BERT in Ranking](https://arxiv.org/abs/1904.07531)
* [Document Expansion by Query Prediction](https://arxiv.org/abs/1904.08375)
* [CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094) (SIGIR2019)
* [Deeper Text Understanding for IR with Contextual Neural Language Modeling](https://arxiv.org/abs/1905.09217) (SIGIR2019)
* [FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance](https://arxiv.org/abs/1905.02851) (SIGIR2019)
* [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424)       
* [REALM: Retrieval-Augmented Language Model Pre-Training](https://kentonl.com/pub/gltpc.2020.pdf)       
* [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/pdf/2002.08910.pdf) [[github](https://github.com/google-research/google-research/tree/master/t5_closed_book_qa)]         
* [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf) [[github](https://github.com/facebookresearch/DPR)]         


### Generation    
* [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model](https://arxiv.org/abs/1902.04094) (NAACL2019 WS)
* [Pretraining-Based Natural Language Generation for Text Summarization](https://arxiv.org/abs/1902.09243)
* [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (EMNLP2019) [[github (original)](https://github.com/nlpyang/PreSumm)] [[github (huggingface)](https://github.com/huggingface/transformers/tree/master/examples/summarization)]
* [Multi-stage Pretraining for Abstractive Summarization](https://arxiv.org/abs/1909.10599)
* [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
* [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450) (ICML2019) [[github](https://github.com/microsoft/MASS)], [[github](https://github.com/microsoft/MASS/tree/master/MASS-fairseq)]
* [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197) [[github](https://github.com/microsoft/unilm)] (NeurIPS2019)
* [UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804) [[github](https://github.com/microsoft/unilm)]
* [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)
* [Towards Making the Most of BERT in Neural Machine Translation](https://arxiv.org/abs/1908.05672)
* [Improving Neural Machine Translation with Pre-trained Representation](https://arxiv.org/abs/1908.07688)
* [On the use of BERT for Neural Machine Translation](https://arxiv.org/abs/1909.12744) (EMNLP2019 WS)
* [Incorporating BERT into Neural Machine Translation](https://openreview.net/forum?id=Hyl7ygStwB) (ICLR2020)
* [Recycling a Pre-trained BERT Encoder for Neural Machine Translation](https://www.aclweb.org/anthology/D19-5603/)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
* [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) (EMNLP2019)
* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
* [ERNIE-GEN: An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)
* [Cross-Lingual Natural Language Generation via Pre-Training](https://arxiv.org/abs/1909.10481) (AAAI2020) [[github](https://github.com/CZWin32768/XNLG)]
* [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)
* [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://arxiv.org/abs/1910.07931)
* [Unsupervised Pre-training for Natural Language Generation: A Literature Review](https://arxiv.org/abs/1911.06171)  
* [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  
* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
* [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)  

### Quality evaluator   
* [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) (ICLR2020)
* [Machine Translation Evaluation with BERT Regressor](https://arxiv.org/abs/1907.12679)
* [SumQE: a BERT-based Summary Quality Estimation Model](https://arxiv.org/abs/1909.00578) (EMNLP2019)
* [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/abs/1909.02622) (EMNLP2019) [[github](https://github.com/AIPHES/emnlp19-moverscore)]
* [BERT as a Teacher: Contextual Embeddings for Sequence-Level Reward](https://arxiv.org/abs/2003.02738)

### Modification (multi-task, masking strategy, etc.)    
* [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504) (ACL2019)
* [The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/2002.07972)
* [BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://arxiv.org/abs/1902.02671) (ICML2019)
* [Unifying Question Answering and Text Classification via Span Extraction](https://arxiv.org/abs/1904.09286)
* [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129) (ACL2019)
* [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)
* [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412) (AAAI2020)
* [Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)
* [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529) [[github](https://github.com/facebookresearch/SpanBERT)]
* [Blank Language Models](https://arxiv.org/abs/2002.03079)
* [Efficient Training of BERT by Progressively Stacking](http://proceedings.mlr.press/v97/gong19a.html) (ICML2019) [[github](https://github.com/gonglinyuan/StackingBERT)]
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) [[github](https://github.com/pytorch/fairseq/tree/master/examples/roberta)]
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (ICLR2020)
* [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) (ICLR2020) [[github](https://github.com/google-research/electra)] [[blog](https://ai.googleblog.com/2020/03/more-efficient-nlp-model-pre-training.html)]
* [FreeLB: Enhanced Adversarial Training for Language Understanding](https://openreview.net/forum?id=BygzbyHFvB) (ICLR2020)
* [KERMIT: Generative Insertion-Based Modeling for Sequences](https://arxiv.org/abs/1906.01604)
* [DisSent: Sentence Representation Learning from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334) (ACL2019)
* [StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577) (ICLR2020)
* [Syntax-Infused Transformer and BERT models for Machine Translation and Natural Language Understanding](https://arxiv.org/abs/1911.06156)
* [SenseBERT: Driving Some Sense into BERT](https://arxiv.org/abs/1908.05646)
* [Semantics-aware BERT for Language Understanding](https://arxiv.org/abs/1909.02209) (AAAI2020)
* [K-BERT: Enabling Language Representation with Knowledge Graph](https://arxiv.org/abs/1909.07606)
* [Knowledge Enhanced Contextual Word Representations](https://arxiv.org/abs/1909.04164) (EMNLP2019)
* [KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://arxiv.org/abs/1911.06136)
* [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) (EMNLP2019)
* [SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models](https://arxiv.org/abs/2002.06652)
* [Universal Text Representation from BERT: An Empirical Study](https://arxiv.org/abs/1910.07973)
* [Symmetric Regularization based BERT for Pair-wise Semantic Reasoning](https://arxiv.org/abs/1909.03405)
* [Transfer Fine-Tuning: A BERT Case Study](https://arxiv.org/abs/1909.00931) (EMNLP2019)
* [Improving Pre-Trained Multilingual Models with Vocabulary Expansion](https://arxiv.org/abs/1909.12440) (CoNLL2019)
* [SesameBERT: Attention for Anywhere](https://arxiv.org/abs/1910.03176)
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) [[github](https://github.com/google-research/text-to-text-transfer-transformer)]
* [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization](https://arxiv.org/abs/1911.03437)

### Probe   
* [A Structural Probe for Finding Syntax in Word Representations](https://aclweb.org/anthology/papers/N/N19/N19-1419/) (NAACL2019)
* [Linguistic Knowledge and Transferability of Contextual Representations](https://arxiv.org/abs/1903.08855) (NAACL2019) [[github](https://github.com/nelson-liu/contextual-repr-analysis)]
* [Probing What Different NLP Tasks Teach Machines about Function Word Comprehension](https://arxiv.org/abs/1904.11544) (*SEM2019)
* [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) (ACL2019)
* [Probing Neural Network Comprehension of Natural Language Arguments](https://arxiv.org/abs/1907.07355) (ACL2019)
* [Cracking the Contextual Commonsense Code: Understanding Commonsense Reasoning Aptitude of Deep Contextual Representations](https://arxiv.org/abs/1910.01157) (EMNLP2019 WS)
* [What do you mean, BERT? Assessing BERT as a Distributional Semantics Model](https://arxiv.org/abs/1911.05758)
* [Quantity doesn't buy quality syntax with neural language models](https://arxiv.org/abs/1909.00111) (EMNLP2019)
* [Are Pre-trained Language Models Aware of Phrases? Simple but Strong Baselines for Grammar Induction](https://openreview.net/forum?id=H1xPR3NtPB) (ICLR2020)
* [oLMpics -- On what Language Model Pre-training Captures](https://arxiv.org/abs/1912.13283)
* [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](http://colinraffel.com/publications/arxiv2020how.pdf)
* [What Does My QA Model Know? Devising Controlled Probes using Expert Knowledge](https://arxiv.org/abs/1912.13337)


### Multi-lingual  
* [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760) (ACL2019)
* [Language Model Pretraining](https://arxiv.org/abs/1901.07291) (NeurIPS2019) [[github](https://github.com/facebookresearch/XLM)]
* [75 Languages, 1 Model: Parsing Universal Dependencies Universally](https://arxiv.org/abs/1904.02099) (EMNLP2019) [[github](https://github.com/hyperparticle/udify)]
* [Zero-shot Dependency Parsing with Pre-trained Multilingual Sentence Representations](https://arxiv.org/abs/1910.05479) (EMNLP2019 WS)
* [Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT](https://arxiv.org/abs/1904.09077) (EMNLP2019)
* [How multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) (ACL2019)
* [How Language-Neutral is Multilingual BERT?](https://arxiv.org/abs/1911.03310)
* [Is Multilingual BERT Fluent in Language Generation?](https://arxiv.org/abs/1910.03806)
* [Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks](https://www.aclweb.org/anthology/D19-1252/) (EMNLP2019)
* [BERT is Not an Interlingua and the Bias of Tokenization](https://www.aclweb.org/anthology/D19-6106/) (EMNLP2019 WS)
* [Cross-Lingual Ability of Multilingual BERT: An Empirical Study](https://openreview.net/forum?id=HJeT3yrtDr) (ICLR2020)
* [Multilingual Alignment of Contextual Word Representations](https://arxiv.org/abs/2002.03518) (ICLR2020)
* [On the Cross-lingual Transferability of Monolingual Representations](https://arxiv.org/abs/1910.11856)
* [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
* [Emerging Cross-lingual Structure in Pretrained Language Models](https://arxiv.org/abs/1911.01464)
* [Can Monolingual Pretrained Models Help Cross-Lingual Classification?](https://arxiv.org/abs/1911.03913)
* [Fully Unsupervised Crosslingual Semantic Textual Similarity Metric Based on BERT for Identifying Parallel Data](https://www.aclweb.org/anthology/K19-1020/) (CoNLL2019)
* [What the \[MASK\]? Making Sense of Language-Specific BERT Models](https://arxiv.org/abs/2003.02912)
* [XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization](https://arxiv.org/abs/2003.11080)

### Other than English models    
* [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894)
* [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372)
* [Multilingual is not enough: BERT for Finnish](https://arxiv.org/abs/1912.07076)
* [BERTje: A Dutch BERT Model](https://arxiv.org/abs/1912.09582)
* [RobBERT: a Dutch RoBERTa-based Language Model](https://arxiv.org/abs/2001.06286)
* [Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language](https://arxiv.org/abs/1905.07213)
* [AraBERT: Transformer-based Model for Arabic Language Understanding](https://arxiv.org/abs/2003.00104)
* [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744) 
* [CLUECorpus2020: A Large-scale Chinese Corpus for Pre-training Language Model](https://arxiv.org/abs/2003.01355)


### Domain specific 
* [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://arxiv.org/abs/1901.08746)
* [Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets](https://arxiv.org/abs/1906.05474) (ACL2019 WS) 
* [BERT-based Ranking for Biomedical Entity Normalization](https://arxiv.org/abs/1908.03548)
* [PubMedQA: A Dataset for Biomedical Research Question Answering](https://arxiv.org/abs/1909.06146) (EMNLP2019)
* [Pre-trained Language Model for Biomedical Question Answering](https://arxiv.org/abs/1909.08229)
* [How to Pre-Train Your Model? Comparison of Different Pre-Training Models for Biomedical Question Answering](https://arxiv.org/abs/1911.00712)
* [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)
* [Publicly Available Clinical BERT Embeddings](https://arxiv.org/abs/1904.03323) (NAACL2019 WS)
* [Progress Notes Classification and Keyword Extraction using Attention-based Deep Learning Models with BERT](https://arxiv.org/abs/1910.05786)
* [SciBERT: Pretrained Contextualized Embeddings for Scientific Text](https://arxiv.org/abs/1903.10676) [[github](https://github.com/allenai/scibert)]
* [PatentBERT: Patent Classification with Fine-Tuning a pre-trained BERT Model](https://arxiv.org/abs/1906.02124)


### Multi-modal    
* [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766) (ICCV2019)
* [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265) (NeurIPS2019)
* [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557)
* [Selfie: Self-supervised Pretraining for Image Embedding](https://arxiv.org/abs/1906.02940)
* [ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data](https://arxiv.org/abs/2001.07966)
* [Contrastive Bidirectional Transformer for Temporal Representation Learning](https://arxiv.org/abs/1906.05743)
* [M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787)
* [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) (EMNLP2019)
* [Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054) (EMNLP2019)
* [BERT representations for Video Question Answering](http://openaccess.thecvf.com/content_WACV_2020/html/Yang_BERT_representations_for_Video_Question_Answering_WACV_2020_paper.html) (WACV2020)
* [Unified Vision-Language Pre-Training for Image Captioning and VQA](https://arxiv.org/abs/1909.11059) [[github](https://github.com/LuoweiZhou/VLP)]
* [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline](https://arxiv.org/abs/1912.02379)
* [VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530) (ICLR2020)
* [Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066)
* [UNITER: Learning UNiversal Image-TExt Representations](https://arxiv.org/abs/1909.11740)
* [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950)
* [Weak Supervision helps Emergence of Word-Object Alignment and improves Vision-Language Tasks](https://arxiv.org/abs/1912.03063)
* [BERT Can See Out of the Box: On the Cross-modal Transferability of Text Representations](https://arxiv.org/abs/2002.10832)
* [BERT for Large-scale Video Segment Classification with Test-time Augmentation](https://arxiv.org/abs/1912.01127) (ICCV2019WS)
* [SpeechBERT: Cross-Modal Pre-trained Language Model for End-to-end Spoken Question Answering](https://arxiv.org/abs/1910.11559)
* [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations](https://arxiv.org/abs/1910.05453)
* [Effectiveness of self-supervised pre-training for speech recognition](https://arxiv.org/abs/1911.03912)
* [Understanding Semantics from Speech Through Pre-training](https://arxiv.org/abs/1909.10924)
* [Towards Transfer Learning for End-to-End Speech Synthesis from Deep Pre-Trained Language Models](https://arxiv.org/abs/1906.07307) 


### Model compression    
* [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
* [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355) (EMNLP2019)
* [Small and Practical BERT Models for Sequence Labeling](https://arxiv.org/abs/1909.00100) (EMNLP2019)
* [Pruning a BERT-based Question Answering Model](https://arxiv.org/abs/1910.06360)
* [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) [[github](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)]
* [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) (NeurIPS2019 WS) [[github](https://github.com/huggingface/transformers/tree/master/examples/distillation)]
* [Knowledge Distillation from Internal Representations](https://arxiv.org/abs/1910.03723) (AAAI2020)
* [PoWER-BERT: Accelerating BERT inference for Classification Tasks](https://arxiv.org/abs/2001.08950)
* [WaLDORf: Wasteless Language-model Distillation On Reading-comprehension](https://arxiv.org/abs/1912.06638)
* [Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687)
* [BERT-of-Theseus: Compressing BERT by Progressive Module Replacing](https://arxiv.org/abs/2002.02925)
* [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307)
* [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
* [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/abs/2002.11985)
* [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)
* [MobileBERT: Task-Agnostic Compression of BERT by Progressive Knowledge Transfer](https://openreview.net/forum?id=SJxjVaNKwB)
* [Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT](https://arxiv.org/abs/1909.05840)
* [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188) (NeurIPS2019 WS)


### LLM
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf)
* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)
* [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html)
* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)
* [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
* [Language models are few-shot learners](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
* [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)
* [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374.pdf)
* [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)
* [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/pdf/2112.06905.pdf)
* [WebGPT: Browser-assisted question-answering with human feedback](https://www.semanticscholar.org/paper/WebGPT%3A-Browser-assisted-question-answering-with-Nakano-Hilton/2f3efe44083af91cef562c1a3451eee2f8601d22)
* [Improving language models by retrieving from trillions of tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)
* [Scaling Language Models: Methods, Analysis &amp; Insights from Training Gopher](https://arxiv.org/pdf/2112.11446.pdf)
* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)
* [LaMDA: Language Models for Dialog Applications](https://arxiv.org/pdf/2201.08239.pdf)
* [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)
* [Using Deep and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/pdf/2201.11990.pdf)
* [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)
* [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311.pdf)
* [An empirical analysis of compute-optimal large language model training](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)
* [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)
* [Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131v1)
* [Emergent Abilities of Large Language Models](https://openreview.net/pdf?id=yzkSU5zdwD)
* [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://github.com/google/BIG-bench)
* [Language Models are General-Purpose Interfaces](https://arxiv.org/pdf/2206.06336.pdf)
* [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/pdf/2209.14375.pdf)
* [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)
* [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/pdf/2210.02414.pdf)
* [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110.pdf)
* [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/pdf/2211.05100.pdf)
* [Galactica: A Large Language Model for Science](https://arxiv.org/pdf/2211.09085.pdf)
* [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/pdf/2212.12017)
* [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)
* [LLaMA: Open and Efficient Foundation Language Models](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/)
* [Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045)
* [PaLM-E: An Embodied Multimodal Language Model](https://palm-e.github.io)
* [GPT-4 Technical Report](https://openai.com/research/gpt-4)
* [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
* [Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision](https://arxiv.org/abs/2305.03047)
* [PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)
* [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf)
* [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/pdf/2401.02385.pdf)



### Misc
* [jiant: A Software Toolkit for Research on General-Purpose Text Understanding Models](https://arxiv.org/abs/2003.02249) [[github](https://github.com/nyu-mll/jiant/)]
* [Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/abs/1903.07785)
* [Learning and Evaluating General Linguistic Intelligence](https://arxiv.org/abs/1901.11373)
* [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://arxiv.org/abs/1903.05987) (ACL2019 WS)
* [Learning to Speak and Act in a Fantasy Text Adventure Game](https://www.aclweb.org/anthology/D19-1062/) (EMNLP2019)
* [Conditional BERT Contextual Augmentation](https://arxiv.org/abs/1812.06705)
* [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)
* [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962) (ICLR2020)
* [Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models](https://openreview.net/forum?id=HkgaETNtDB) (ICLR2020)
* [A Mutual Information Maximization Perspective of Language Representation Learning](https://openreview.net/forum?id=Syx79eBKwr) (ICLR2020)
* [Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment](https://arxiv.org/abs/1907.11932) (AAAI2020)
* [Thieves on Sesame Street! Model Extraction of BERT-based APIs](https://arxiv.org/abs/1910.12366) (ICLR2020)
* [Graph-Bert: Only Attention is Needed for Learning Graph Representations](https://arxiv.org/abs/2001.05140)
* [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
* [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305)
* [Extending Machine Language Models toward Human-Level Language Understanding](https://arxiv.org/abs/1912.05877)
* [Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/abs/1901.10125)
* [Back to the Future -- Sequential Alignment of Text Representations](https://arxiv.org/abs/1909.03464)
* [Improving Cuneiform Language Identification with BERT](https://www.aclweb.org/anthology/papers/W/W19/W19-1402/) (NAACL2019 WS)
* [BERT has a Moral Compass: Improvements of ethical and moral values of machines](https://arxiv.org/abs/1912.05238)
* [SMILES-BERT: Large Scale Unsupervised Pre-Training for Molecular Property Prediction](https://dl.acm.org/citation.cfm?id=3342186) (ACM-BCB2019)
* [On the comparability of Pre-trained Language Models](https://arxiv.org/abs/2001.00781)
* [Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771) 
* [Jukebox: A Generative Model for Music](https://cdn.openai.com/papers/jukebox.pdf)  
* [WT5?! Training Text-to-Text Models to Explain their
Predictions](https://arxiv.org/pdf/2004.14546.pdf)  
* [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/pdf/2004.02349.pdf) [[github](https://github.com/google-research/tapas)]  
* [TABERT: Pretraining for Joint Understanding of
Textual and Tabular Data](https://arxiv.org/pdf/2005.08314.pdf)    


# Author
ChangWookJun / @changwookjun (changwookjun@gmail.com)
