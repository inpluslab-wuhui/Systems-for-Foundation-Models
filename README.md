# Big Model paper

A curated big model related academic papers. All papers are sorted based on the conference/journal name and published year. Welcome developers or researchers to add more published paper to this list. 

## Table of Listed Conferences

|     Security & Privacy & Crypto     |               Networking & Database               | Software Engineering & Programming Language | System Architecture  |
| :---------------------------------: | :-----------------------------------------------: | :-----------------------------------------: | :------------------: |
|          [CRYPTO](##dddd)           | [SIGMETRICS & Performance](#sigmetricperformance) |                [ICSE](#icse)                | [EuroSys](#eurosys)  |
|       [EUROCRYPT](#eurocrypt)       |                   [ICDE](#icde)                   |            [ESEC/FSE](#esecfse)             |  [ACM SOSP](#sosp)   |
| [USENIX Security](#usenix-security) |                   [VLDB](#vldb)                   |                 [ASE](#ase)                 |  [EuroS&P](#eurosp)  |
|           [IEEE S&P](#sp)           |               [ACM SIGMOD](#sigmod)               |              [ACM PLDI](#pldi)              |    [SRDS](#srds)     |
|            [NDSS](#ndss)            |             [IEEE INFOCOM](#infocom)              |            [ACM OOPSLA](#oopsla)            |  [ACM PODC](#podc)   |
|           [ACM CCS](#ccs)           |                   [NSDI](#nsdi)                   |                [ACM EC](#ec)                | [IEEE IPDPS](#ipdps) |
|          [IEEE DSN](#dsn)           |               [ACM CoNEXT](#conext)               |               [ISSTA](#issta)               | [IEEE ICDCS](#icdcs) |
|              [FC](#fc)              |              [ACM MobiHoc](#mobihoc)              |              [ACM POPL](#popl)              |  [ACM SOCC](#socc)   |
|             [IMC](#imc)             |                                                   |                                             |                      |


## Table of Listed Journals
- [IEEE Transaction on Knowledger and Data Engineering(TKDE)](#tkde)
- [ACM Computing Surveys (ACM CSUR)](#acmcsur)
- [ACM Distributed Ledger Technologies: Research and Practice (ACM DLT)](#acmdlt)
- [IEEE Journal on Selected Areas in Communications](#jsac)
- [IEEE Transactions on Network Science and Engineering](#tnse)

Key Words: Topics: System Architecture, Consensus(Proof-of-X and BFT), Layer 2 (Off-chain, Payment Networks, Sidechain, Crosschain), Network, Smart Contracts, Application (Trasactions), Cryptograph, Storage (light client); Contents: privacy, security, economics (incentive).

## Conferences

## NIPS

[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. NIPS'17

[GPipe: efficient training of giant neural networks using pipeline parallelism](https://dl.acm.org/doi/pdf/10.5555/3454287.3454297). Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen. NIPS'19

[Language Models are Few-Shot Learners](https://dl.acm.org/doi/pdf/10.5555/3495724.3495883). Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. NIPS'20

[COMPACTER:Efficient Low-Rank Hypercomplex Adapter Layers](https://arxiv.org/pdf/2106.04647.pdf). Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder. NIPS'21

## SC

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf). Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. SC'20

[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209). Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia. SC'21

[Chimera: efficiently training large-scale neural networks with bidirectional pipelines](https://dl.acm.org/doi/pdf/10.1145/3458817.3476145). Shigang Li, Torsten Hoefler. SC'21

[ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning](https://arxiv.org/pdf/2104.07857.pdf). Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. SC'21

## ATC

[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/system/files/atc21-ren-jie.pdf). Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. ATC'21

[Whale: Efficient Giant Model Training over Heterogeneous GPUs](https://www.usenix.org/system/files/atc22-jia-xianyan.pdf). Xianyan Jia, Le Jiang, Ang Wang, and Wencong Xiao, Alibaba Group; Ziji Shi, National University of Singapore & Alibaba Group; Jie Zhang, Xinyuan Li, Langshi Chen, Yong Li, Zhen Zheng, Xiaoyong Liu, and Wei Lin, Alibaba Group. ATC'22

## ASPLOS

[Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers](https://dl.acm.org/doi/pdf/10.1145/3575693.3575703). Yangyang Feng, Minhui Xie, Zijie Tian, Shuo Wang, Youyou Lu, Jiwu Shu. ASPLOS'23  

[Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning](https://dl.acm.org/doi/pdf/10.1145/3575693.3575725). Shuangyan Yang, Minjia Zhang, Wenqian Dong, Dong Li. ASPLOS'23 

[ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning](https://dl.acm.org/doi/10.1145/3575693.3575721). Diandian Gu, Yihao Zhao, Yinmin Zhong, Yifan Xiong, Zhenhua Han, Peng Cheng, Fan Yang, Gang Huang, Xin Jin, Xuanzhe Liu. ASPLOS'23 

[Lucid: A Non-intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs](https://dl.acm.org/doi/pdf/10.1145/3575693.3575705). Qinghao Hu, Meng Zhang, Peng Sun, Yonggang Wen, Tianwei Zhang. ASPLOS'23

[Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). Shibo Wang, Jinliang Wei, Amit Sabne, Andy Davis, Berkin Ilbeyi, Blake Hechtman, Dehao Chen, Karthik Srinivasa Murthy, Marcello Maggioni, Qiao Zhang, Sameer Kumar, Tongfei Guo, Yuanzhong Xu, Zongwei Zhou. ASPLOS'23

[Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression](https://arxiv.org/pdf/2301.09830.pdf). Jaeyong Song, Jinkyu Yim, Jaewon Jung, Hongsun Jang, Hyung-Jin Kim, Youngsok Kim, Jinho Lee. ASPLOS'23

## NAACL-HLT

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf). Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. NAACL-HLT'19

##  ICML

[BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://arxiv.org/pdf/1902.02671.pdf). Asa Cooper Stickland, Iain Murray. ICML'19

[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf). Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly. ICML'19

[DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf). Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He.  ICML'22

## EUROSYS

[Varuna: scalable, low-cost training of massive deep learning models](https://dl.acm.org/doi/pdf/10.1145/3492321.3519584). Sanjith Athlur, Nitika Saran, Muthian Sivathanu, Ramachandran Ramjee, Nipun Kwatra. EUROSYS'22

## KDD

[DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://dl.acm.org/doi/10.1145/3394486.3406703). Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, Yuxiong He. KDD'20

## OSDI

[Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads](https://www.usenix.org/system/files/osdi21-thorpe.pdf). John Thorpe, Yifan Qiao, Jonathan Eyolfson, Shen Teng, Guanzhou Hu, Zhihao Jia, Jinliang Wei, Keval Vora, Ravi Netravali, Miryung Kim, Guoqing Harry Xu. osdi'21

[Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf). Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica. osdi'22

[Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/system/files/osdi22-unger.pdf). Colin Unger, Stanford University; Zhihao Jia, Carnegie Mellon University and Meta; Wei Wu, Los Alamos National Laboratory and NVIDIA; Sina Lin, Microsoft; Mandeep Baines, Carlos Efrain Quintero Narvaez, Vinay Ramakrishnaiah, Nirmal Prajapati, Pat McCormick, Jamaludin Mohd-Yusof, Xi Luo, Dheevatsa Mudigere, Jongsoo Park, Misha Smelyanskiy, Alex Aiken, osdi'22

[Walle: An End-to-End, General-Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning](https://www.usenix.org/system/files/osdi22-lv.pdf). Chengfei Lv, Zhejiang University and Alibaba Group; Chaoyue Niu, Shanghai Jiao Tong University and Alibaba Group; Renjie Gu, Xiaotang Jiang, Zhaode Wang, Bin Liu, Ziqi Wu, Qiulin Yao, Congyu Huang, Panos Huang, Tao Huang, Hui Shu, Jinde Song, Bin Zou, Peng Lan, and Guohuan Xu, Alibaba Group; Fei Wu, Zhejiang University; Shaojie Tang, University of Texas at Dallas; Fan Wu and Guihai Chen, Shanghai Jiao Tong University. osdi'22

## SOSP

[PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/10.1145/3341301.3359646). Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, Matei Zaharia. SOSP'19

## ICPP

[Tesseract: Parallelize the Tensor Parallelism Efficiently](https://arxiv.org/pdf/2105.14500.pdf). Boxiang Wang, Qifan Xu, Zhengda Bian, Yang You. ICPP'22

## EMNLP 

[AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779.pdf). Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, Iryna Gurevych. EMNLP'20

[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691). Brian Lester, Rami Al-Rfou, Noah Constant. EMNLP'21

## ACL

[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190). Xiang Lisa Li, Percy Liang. ALC'21

[Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf). Ziyun Xu, Chengyu Wang, Minghui Qiu, Fuli Luo, Runxin Xu, Songfang Huang, Jun Huang. ACL'21

[Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks](https://arxiv.org/pdf/2106.04489.pdf). Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, James Henderson. ACL'21

[MSP: Multi-Stage Prompting for Making Pre-trained Language Models Better Translators](https://arxiv.org/pdf/2110.06609.pdf). Zhixing Tan, Xiangwen Zhang, Shuo Wang, Yang Liu. ACL'22

[Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/pdf/2108.02035.pdf). Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li, Wei Wu, Maosong Sun. ACL'22

## Journals

## VLDB

[PyTorch distributed: experiences on accelerating data parallel training](https://arxiv.org/pdf/2006.15704.pdf). Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, Soumith Chintala. VLDB'20

[Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism](https://arxiv.org/abs/2211.13878). Xupeng Miao, Yujie Wang, Youhe Jiang, Chunan Shi, Xiaonan Nie, Hailin Zhang, Bin Cui. VLDB'22

## IEEE Transactions on Parallel and Distributed Systems

[Parallel Training of Pre-Trained Models via Chunk-Based Dynamic Memory Management](https://arxiv.org/pdf/2108.05818.pdf). Jiarui Fang, Zilin Zhu, Shenggui Li, Hui Su, Yang Yu, Jie Zhou, Yang You. IEEE Transactions on Parallel and Distributed Systems'23

## ACM Computing Surveys

[Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://dl.acm.org/doi/pdf/10.1145/3560815). Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig. ACM Computing Surveys. 23

## CoRR

[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf). Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. CoRR'19

[Training Large Neural Networks with Constant Memory using a New Execution Algorithm](https://arxiv.org/pdf/2002.05645.pdf). Bharadwaj Pudipeddi, Maral Mesmakhosroshahi, Jinwen Xi, Sujeeth Bharadwaj. CoRR'20

[An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/pdf/2104.05343.pdf). Qifan Xu, Shenggui Li, Chaoyu Gong, Yang You. CoRR'21

[Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/pdf/2105.14450.pdf). Zhengda Bian, Qifan Xu, Boxiang Wang, Yang You. CoRR'21

[GPT Understands, Too](https://arxiv.org/pdf/2103.10385). Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang. CoRR'21

[On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Ré, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang et al. CoRR'21

[Decentralized Training of Foundation Models in Heterogeneous Environments](https://arxiv.org/pdf/2206.01288.pdf). Binhang Yuan, Yongjun He, Jared Quincy Davis, Tianyi Zhang, Tri Dao, Beidi Chen, Percy Liang, Christopher Re, Ce Zhang. CoRR'22

[Survey on Large Scale Neural Network Training](https://arxiv.org/pdf/2202.10435.pdf). Julia Gusak, Daria Cherniuk, Alena Shilova, Alexander Katrutsa, Daniel Bershatsky, Xunyi Zhao, Lionel Eyraud-Dubois, Oleg Shlyazhko, Denis Dimitrov, Ivan Oseledets, Olivier Beaumont. CoRR'22

[Transformer models: an introduction and catalog](https://arxiv.org/pdf/2302.07730.pdf). Xavier Amatriain. CoRR'23

## Research

[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever. https://openai.com/research/language-unsupervised. 18

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)


This list is released into the public domain.