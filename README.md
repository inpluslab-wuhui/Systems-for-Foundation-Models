# Systems for Foundation Models

A curated collection of academic papers related to foundation models. All papers in this collection are sorted based on the conference/journal name and the year of publication. Developers and researchers are welcome to contribute by adding more published papers to this list.

Key Words: foundation model, large-scale models, model training, model inference, pipeline parallelism, model parallelism, tensor parallelism, data parallelism, pre-training, fine-tuning, zero-shot, model compression, data compression, gradient compression, memory footprint reduction, batching, heterogeneous system, distributed system,  network architecture

## Table of Listed Conferences

- [ACM SIGPLAN Symposium on Principles & Practice of Parallel Programming (PPoPP)](#PPoPP)

- [Conference on Neural Information Processing Systems (NeurIPS)](#NIPS)

- [International Conference for High Performance Computing, Networking, Storage and Analysis (SC)](#SC)

- [USENIX Annual Technical Conference (USENIX ATC)](#ATC)

- [International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)](#ASPLOS)

- [North American Chapter of the Association for Computational Linguistics (NAACL)](#NAACL-HLT)

- [International Conference on Machine Learning (ICML)](#ICML)

- [European Conference on Computer Systems (EuroSys)](#EUROSYS)

- [ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)](#KDD)

- [USENIX Symposium on Operating Systems Design and Implementation (OSDI)](#OSDI)

- [Symposium on Operating Systems Principles (SOSP)](#SOSP)

- [International Conference on Parallel Processing (ICPP)](#ICPP)

- [Conference on Empirical Methods in Natural Language Processing (EMNLP)](#EMNLP)

- [Annual Meeting of the Association for Computational Linguistics (ACL)](#ACL)

- [International Conference on Learning Representations (ICLR)](#ICLR)

- [Very Large Data Bases Conference (VLDB)](#VLDB)

- [Symposium on Network System Design and Implementation(NSDI)](#NSDI)

- [Association for the Advancement of Artificial Intelligence(AAAI)](#AAAI)

- [IEEE International Conference on Computer Vision(ICCV)](#ICCV)

- [Conference on File and Storage Technologies(FAST)](#FAST)

- [International Symposium on Computer Architecture(ISCA)](#ISCA)

- [High Performance Computer Architecture(HPCA)](#HPCA)

- [IEEE International Conference on Computer Communications(INFOCOM)](#INFOCOM)

- [IEEE International Parallel & Distributed Processing Symposium(IPDPS)](#IPDPS)

- [Machine Learning and Systems(MLSys)](#MLSys)

- [ACM Special Interest Group on Data Communication(SIGCOMM)](#SIGCOMM)

   


## Table of Listed Journals
- [IEEE Transactions on Parallel and Distributed Systems (TPDS)](#TPDS)
- [ACMComputingSurveys](#ACMComputingSurveys)
- [JournalofMachineLearningResearch](#JournalofMachineLearningResearch)
- [Transactions on Machine Learning Research(TMLR)](#TMLR)

## Conferences

## PPoPP

[dynamic GPU memory scheduling] [Superneurons: dynamic GPU memory management for training deep neural networks](https://dl.acm.org/doi/pdf/10.1145/3200691.3178491). Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuaiwen Leon Song, Zenglin Xu, Tim Kraska. PPoPP'18

[training on supercomputer] [BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/ppopp22-bagualu/ppopp22-bagualu.pdf). Zixuan Ma1 , Jiaao He1 , Jiezhong Qiu1,4, Huanqi Cao1 , Yuanwei Wang1 , Zhenbo Sun1 , Liyan Zheng1 , Haojie Wang1 , Shizhi Tang1 , Tianyu Zheng3 , Junyang Lin2 , Guanyu Feng1 , Zeqiang Huang3 , Jie Gao3 , Aohan Zeng1,4, Jianwei Zhang2 , Runxin Zhong1 , Tianhui Shi1 , Sha Liu3 , Weimin Zheng1 , Jie Tang1,4, Hongxia Yang2 , Xin Liu3 , Jidong Zhai1 , Wenguang Chen1. PPoPP'22

[distributed MoE model training] [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/ppopp22-fastermoe/ppopp22-fastermoe.pdf). Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, Qin Li. PPoPP'22

[pipeline parallelism] [Elastic Averaging for Efficient Pipelined DNN Training](https://doi.org/10.1145/3572848.3577484). Zihao Chen, Chen Xu, Weining Qian, Aoying Zhou. PPoPP'23

[sparse attention] [Dynamic N M Fine-grained Structured Sparse Attention Mechanism](https://dl.acm.org/doi/pdf/10.1145/3572848.3577500). Zhaodong Chen, Zheng Qu, Yuying Quan, Liu Liu, Yufei Ding, Yuan Xie. PPoPP'23

[failure recovery] [POSTER Swift Expedited Failure Recovery for Large-scale DNN Training](https://arxiv.org/pdf/2302.06173.pdf). Yuchen Zhong, Guangming Sheng, Juncheng Liu, Jinhui Yuan, and Chuan Wu. PPoPP'23

## NIPS

[network architecture] [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. NIPS'17

[parallel decoding] [Blockwise Parallel Decoding for Deep Autoregressive Models](https://proceedings.neurips.cc/paper_files/paper/2018/file/c4127b9194fe8562c64dc0f5bf2c93bc-Paper.pdf). Mitchell Stern, Noam Shazeer, Jakob Uszkoreit. NIPS'18

[pipeline parallelism] [GPipe: efficient training of giant neural networks using pipeline parallelism](https://dl.acm.org/doi/pdf/10.5555/3454287.3454297). Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen. NIPS'19

[pre-training] [Language Models are Few-Shot Learners](https://dl.acm.org/doi/pdf/10.5555/3495724.3495883). Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. NIPS'20

[fine-tuning] [COMPACTER:Efficient Low-Rank Hypercomplex Adapter Layers](https://arxiv.org/pdf/2106.04647.pdf). Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder. NIPS'21

[reinforcement learning] [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345). 	Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch. NIPS'21

[vision moe model] [Scaling Vision with Sparse Mixture of Experts](https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf). Carlos Riquelme, Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, Neil Houlsby. NIPS'22

[length generalization] [Exploring Length Generalization in Large Language Models](https://arxiv.org/pdf/2207.04901). Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay V. Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, Behnam Neyshabur. NIPS'22

[model compression] [XTC: Extreme Compression for Pre-trained Transformers Made Simple and Efficient](https://arxiv.org/pdf/2206.01859). Xiaoxia Wu, Zhewei Yao, Minjia Zhang, Conglong Li, Yuxiong He. NIPS'22

[zero-shot] [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/pdf/2202.04538). Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han. NIPS'22

[memory footprint reduction] [Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction](https://arxiv.org/pdf/2210.10246). Muralidhar Andoorveedu, Zhanda Zhu, Bojian Zheng, Gennady Pekhimenko. NIPS'22

[model compression] [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/pdf/2206.01861). 	Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He. NIPS'22

[rlhf] [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://arxiv.org/pdf/2306.01693.pdf). Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, Hannaneh Hajishirzi. NIPS'23

[rlhf] [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/pdf/2304.05302.pdf). Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang. NIPS'23

## SC

[adaptive batching] [BATCH: Machine Learning Inference Serving on Serverless Platforms with Adaptive Batching](https://www.cse.unr.edu/~fyan/Paper/Feng-SC20-BATCH.pdf). Ahsan Ali, Riccardo Pinciroli, Feng Yan, Evgenia Smirni. SC'20

[memory optimizations] [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf). Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. SC'20

[pipelining schedule] [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209). Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia. SC'21

[pipeline parallelism] [Chimera: efficiently training large-scale neural networks with bidirectional pipelines](https://dl.acm.org/doi/pdf/10.1145/3458817.3476145). Shigang Li, Torsten Hoefler. SC'21

[heterogeneous system] [ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning](https://arxiv.org/pdf/2104.07857.pdf). Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He. SC'21

[parallel matrix multiplication] [CA3DMM: A New Algorithm Based on a Unified View of Parallel Matrix Multiplication](https://dl.acm.org/doi/pdf/10.5555/3571885.3571922). Hua Huang, Edmond Chow. SC'22

[GNN training] [CoGNN: Efficient Scheduling for Concurrent GNN Training on GPUs](https://dl.acm.org/doi/pdf/10.5555/3571885.3571936). Qingxiao Sun, Yi Liu, Hailong Yang, Ruizhe Zhang, Ming Dun, Mingzhen Li, Xiaoyan Liu, Wencong Xiaoy, Yong Liy, Zhongzhi Luan, Depei Qian. SC'22

[model inference] [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://dl.acm.org/doi/pdf/10.5555/3571885.3571946). Reza Yazdani Aminabadi, Samyam Rajbhandari, Minjia Zhang, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Jeff Rasley, Shaden Smith, Olatunji Ruwase, Yuxiong He, Microsoft Corporation. SC'22

[large-scale recommendation model training] [EL-Rec: Efficient Large-Scale Recommendation Model Training via Tensor-Train Embedding Table](https://dl.acm.org/doi/pdf/10.5555/3571885.3571978). Zheng Wang, Yuke Wang, Boyuan Feng, Dheevatsa Mudigere, Bharath Muthiah, Yufei Ding. SC'22

[network topology] [HammingMesh: A Network Topology for Large-Scale Deep Learning](https://dl.acm.org/doi/pdf/10.5555/3571885.3571899). Torsten Hoeflery, Tommaso Bonato, Daniele De Sensi, Salvatore Di Girolamo, Shigang Li, Marco Heddesy, Jon Belky, Deepak Goely, Miguel Castroy, and Steve Scotty. SC'22

[accelerate training] [LightSeq2: Accelerated Training for Transformer-Based Models on GPUs](https://dl.acm.org/doi/pdf/10.5555/3571885.3571935). Xiaohui Wang, Yang Wei, Ying Xiong, Guyue Huang, Xian Qian, Yufei Ding, Mingxuan Wang, Lei Li. SC'22

[variability in accelerator-rich systems] [Not All GPUs Are Created Equal: Characterizing Variability in Large-Scale, Accelerator-Rich Systems](https://dl.acm.org/doi/pdf/10.5555/3571885.3571971). Prasoon Sinha, Akhil Guliani, Rutwik Jain, Brandon Tran, Matthew D. Sinclair and Shivaram Venkataraman. SC'22

[DNN model training] [STRONGHOLD: Fast and Affordable Billion-Scale Deep Learning Model Training](https://dlnext.acm.org/doi/pdf/10.5555/3571885.3571979). Xiaoyang Sun, Wei Wang, Shenghao Qiu, Renyu Yang, Songfang Huang, Jie Xu, Zheng Wang. SC'22

[GNN training] [WholeGraph: A Fast Graph Neural Network Training Framework with Multi-GPU Distributed Shared Memory Architecture](https://dl.acm.org/doi/pdf/10.5555/3571885.3571956). Dongxu Yang, Junhong Liu,  Jiaxing Qi,  Junjie Lai. SC'22

[pipeline parallelism] [Hanayo: Harnessing Wave-like Pipeline Parallelism for Enhanced Large Model Training Efficiency](https://dl.acm.org/doi/pdf/10.1145/3581784.3607073). Ziming Liu, Shenggan Cheng, Haotian Zhou, Yang You. SC'23

## ATC

[fine tuning] [Cavs: An Efficient Runtime System for Dynamic Neural Networks](https://www.usenix.org/system/files/conference/atc18/atc18-xu-shizhen.pdf). Shizhen Xu, Carnegie Mellon University, Tsinghua University; Hao Zhang, Graham Neubig, and Wei Dai, Carnegie Mellon University, Petuum Inc.; Jin Kyu Kim, Carnegie Mellon University; Zhijie Deng, Tsinghua University; Qirong Ho, Petuum Inc.; Guangwen Yang, Tsinghua University; Eric P. Xing, Petuum Inc. ATC'18

[cpu speedup] [DeepCPU: Serving RNN-based Deep Learning Models 10x Faster](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-minjia.pdf). Minjia Zhang, Samyam Rajbhandari, Wenhan Wang Yuxiong He. ATC'18

[network resource share] [DynaMix: Dynamic Mobile Device Integration for Efficient Cross-device Resource Sharing](https://www.usenix.org/system/files/conference/atc18/atc18-chae.pdf). Dongju Chae, POSTECH;Joonsung Kim and Gwangmu Lee, Seoul National University; Hanjun Kim, POSTECH; Kyung-Ah Chang and Hyogun Lee, Samsung Electronics;Jangwoo
Kim, Seoul National University. ATC'18

[IO mitigating in vCPU+cpu resource share] [Effectively Mitigating I/O Inactivity in vCPU Scheduling](https://www.usenix.org/system/files/conference/atc18/atc18-jia.pdf). Weiwei Jia, The University of Hong Kong, New Jersey Institute of Technology; Cheng Wang and Xusheng Chen, The University of Hong Kong; Jianchen Shan and Xiaowei Shang, New Jersey Institute of Technology; Heming Cui, The University of Hong Kong; Xiaoning Ding, New Jersey Institute of Technology; Luwei Cheng, Facebook; Francis C. M. Lau and Yuexuan Wang, The University of Hong Kong; Yuangang Wang, Huawei. ATC'18

[ML distributed inference] [Litz: Elastic Framework for High-Performance Distributed Machine Learning](https://www.usenix.org/system/files/conference/atc18/atc18-qiao.pdf). Aurick Qiao, Petuum, Inc. and Carnegie Mellon University; Abutalib Aghayev, Carnegie Mellon University; Weiren Yu, Petuum, Inc. and Beihang University; Haoyang Chen and Qirong Ho, Petuum, Inc.; Garth A. Gibson, Carnegie Mellon 
University and Vector Institute; Eric P. Xing, Petuum, Inc. and Carnegie Mellon University. ATC'18

[finetuning] [Mainstream: Dynamic Stem-Sharing for Multi-Tenant Video Processing](https://www.usenix.org/system/files/conference/atc18/atc18-jiang.pdf). Angela H. Jiang, Daniel L.K. Wong, Christopher Canel, Lilia Tang, and Ishan Misra, Carnegie Mellon University; Michael Kaminsky, Michael A. Kozuch, and Padmanabhan Pillai, Intel Labs; David G. Andersen and Gregory R. Ganger, Carnegie Mellon University. ATC'18

[instance placement] [Placement of Virtual Containers on NUMA systems: A Practical and Comprehensive Model](https://www.usenix.org/system/files/conference/atc18/atc18-funston.pdf). Justin Funston, Maxime Lorrillere, and Alexandra Fedorova, University of British Columbia; Baptiste Lepers, EPFL; David Vengerov and Jean-Pierre Lozi, Oracle Labs; Vivien Quéma, IMAG. ATC'18

[sparse matrix operation] [Locality-Aware Software Throttling for Sparse Matrix Operation on GPUs](https://www.usenix.org/sites/default/files/conference/protected-files/atc18_slides_chen.pdf). Yanhao Chen and Ari B. Hayes, Rutgers University; Chi Zhang, University of Pittsburgh; Timothy Salmon and Eddy Z. Zhang, Rutgers University. ATC'18

[data compression] [TerseCades: Efficient Data Compression in Stream Processing](https://www.usenix.org/system/files/conference/atc18/atc18-pekhimenko.pdf). Gennady Pekhimenko, University of Toronto; Chuanxiong Guo, Bytedance Inc.; Myeongjae Jeon, Microsoft Research; Peng Huang, Johns Hopkins University; Lidong Zhou, Microsoft Research. ATC'18

[spot resource usage] [Tributary: spot-dancing for elastic services with latency SLOs](https://www.usenix.org/system/files/conference/atc18/atc18-harlap.pdf). Aaron Harlap and Andrew Chung, Carnegie Mellon University; Alexey Tumanov, 
UC Berkeley; Gregory R. Ganger and Phillip B. Gibbons, Carnegie Mellon University. ATC'18

[load balancing] [The Battle of the Schedulers: FreeBSD ULE vs. Linux CFS](https://www.usenix.org/sites/default/files/conference/protected-files/atc18_slides_bouron.pdf).  Justinien Bouron, Sebastien Chevalley, Baptiste Lepers, and Willy Zwaenepoel, EPFL; Redha Gouicem, Julia Lawall, Gilles Muller, and Julien Sopena, Sorbonne University/Inria/LIP6. ATC'18

[GPU analysis] [Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](https://www.usenix.org/system/files/atc19-jeon.pdf). Myeongjae Jeon, UNIST and Microsoft Research; Shivaram Venkataraman, University of Wisconsin and Microsoft Research; Amar Phanishayee and Junjie Qian, Microsoft Research; Wencong Xiao, Beihang University and Microsoft Research; Fan Yang, Microsoft Research. ATC'19

[model inference] [Optimizing CNN Model Inference on CPUs](https://www.usenix.org/system/files/atc19-liu-yizhi.pdf). Yizhi Liu, Yao Wang, Ruofei Yu, Mu Li, Vin Sharma, and Yida Wang, Amazon. ATC'19

[schedule in CPU-GPU framework] [FineStream: Fine-Grained Window-Based Stream Processing on CPU-GPU Integrated Architectures](https://www.usenix.net/system/files/atc20-zhang-feng.pdf). Feng Zhang and Lin Yang, Renmin University of China; Shuhao Zhang, Technische Universität Berlin and National University of Singapore; Bingsheng He, National University of Singapore; Wei Lu and Xiaoyong Du, Renmin University of China. ATC'20

[offload workload between CPU and GPU] [Offload Annotations: Bringing Heterogeneous Computing to Existing Libraries and Workloads](https://www.usenix.org/system/files/atc20-yuan.pdf). Gina Yuan, Shoumik Palkar, Deepak Narayanan, and Matei Zaharia, Stanford University. ATC'20

[DNN deployment system] [ALERT: Accurate Learning for Energy and Timeliness](https://www.usenix.org/system/files/atc20-wan.pdf). Chengcheng Wan, Muhammad Santriaji, Eri Rogers, Henry Hoffmann, Michael Maire, and Shan Lu, University of Chicago. ATC'20

[DNN running time predict] [Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training](https://www.usenix.org/system/files/atc20-zhu-hongyu.pdf). Hongyu Zhu, University of Toronto & Vector Institute; Amar Phanishayee, Microsoft Research; Gennady Pekhimenko, University of Toronto & Vector Institute. ATC'20

[DNN traning in GPU] [HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism](https://www.usenix.org/system/files/atc20-paper1132-slides-park.pdf). Jay H. Park, Gyeongchan Yun, Chang M. Yi, Nguyen T. Nguyen, and Seungmin Lee, UNIST; Jaesik Choi, KAIST; Sam H. Noh and Young-ri Choi, UNIST. ATC'20

[multi DNN deployment platform] [NeuOS: A Latency-Predictable Multi-Dimensional Optimization Framework for DNN-driven Autonomous Systems](https://www.usenix.org/system/files/atc20-bateni.pdf). Soroush Bateni and Cong Liu, University of Texas at Dallas. ATC'20

[pipeline model parallelism] [Fine-tuning giant neural networks on commodity  hardware with automatic pipeline model parallelism](https://www.usenix.org/system/files/atc21-eliad.pdf). Saar Eliad, Ido Hakimi, and Alon De Jagger, Department of Computer Science, Technion - Israel Institute of Technology; Mark Silberstein, Department of Computer 
Science and Department of Electrical Engineering, Technion - Israel Institute of Technology; Assaf Schuster, Department of Computer Science, Technion - Israel Institute of Technology. ATC'21

[model-less inference serving] [INFaaS: Automated Model-less Inference Serving](https://www.usenix.org/system/files/atc21-romero.pdf). Francisco Romero, Qian Li, Neeraja J. Yadwadkar, and Christos Kozyrakis, Stanford University. ATC'21

[model running time predict] [Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training](https://www.usenix.org/system/files/atc21-yu.pdf). Geoffrey X. Yu, University of Toronto/Vector Institute; Yubo Gao, University of Toronto; Pavel Golikov and Gennady Pekhimenko, University of Toronto/Vector Institute. ATC'21

[giant model training by offloading data and compute to cpu] [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/system/files/atc21-ren-jie.pdf). Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, Yuxiong He. ATC'21

[ML training in GPU with GPU share] [Zico: Efficient GPU Memory Sharing for Concurrent DNN Training](https://www.usenix.org/system/files/atc21-lim.pdf). Gangmuk Lim, UNIST; Jeongseob Ahn, Ajou University; Wencong Xiao, Alibaba Group; Youngjin Kwon, KAIST; Myeongjae Jeon, UNIST. ATC'21

[gaint model training in GPU] [Whale: Efficient Giant Model Training over Heterogeneous GPUs](https://www.usenix.org/system/files/atc22-jia-xianyan.pdf). Xianyan Jia, Le Jiang, Ang Wang, and Wencong Xiao, Alibaba Group; Ziji Shi, National University of Singapore & Alibaba Group; Jie Zhang, Xinyuan Li, Langshi Chen, Yong Li, Zhen Zheng, Xiaoyong Liu, and Wei Lin, Alibaba Group. ATC'22

[ML failure recover in GPU] [Sibylla: To Retry or Not To Retry on Deep Learning Job Failure](https://www.usenix.org/system/files/atc22-kim-taeyoon.pdf). Taeyoon Kim, Suyeon Jeong, Jongseop Lee, Soobee Lee, and Myeongjae Jeon, UNIST. ATC'22

[mixed precision training] [Campo: Cost-Aware Performance Optimization for Mixed-Precision Neural Network Training](https://www.usenix.org/system/files/atc22-he.pdf). Xin He, CSEE, Hunan University & Xidian University; Jianhua Sun and Hao Chen, CSEE, Hunan University; Dong Li, University of California, Merced. ATC'22

[DNN in GPU with batch] [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs](https://www.usenix.org/system/files/atc22-cui.pdf). Weihao Cui, Han Zhao, Quan Chen, Hao Wei, and Zirui Li, Shanghai Jiao Tong University; Deze Zeng, China University of Geosciences; Chao Li and Minyi Guo, Shanghai Jiao Tong University. ATC'22

[transformer in GPU] [Faith: An Efficient Framework for Transformer Verification on GPUs](https://www.usenix.org/system/files/atc22-feng.pdf). Boyuan Feng, Tianqi Tang, Yuke Wang, Zhaodong Chen, Zheng Wang, Shu Yang, Yuan Xie, and Yufei Ding, University of California, Santa Barbara. ATC'22

[transformer inference] [PetS: A Unified Framework for Parameter-Efficient Transformers Serving](PetS: A Unified Framework for Parameter-Efficient Transformers Serving). Zhe Zhou, Peking University; Xuechao Wei, Peking University and Alibaba Group; Jiejing Zhang, Alibaba Group; Guangyu Sun, Peking University. ATC'22

[GPU share] [PilotFish: Harvesting Free Cycles of Cloud Gaming with Deep Learning Training](https://www.usenix.org/system/files/atc22-zhang-wei.pdf). Wei Zhang and Binghao Chen, Shanghai Jiao Tong University; Zhenhua Han, Microsoft Research Asia; Quan Chen, Shanghai Jiao Tong University; Peng Cheng, Fan Yang, Ran Shu, and Yuqing Yang, Microsoft Research; Minyi Guo, Shanghai Jiao Tong University. ATC'22

[temporal sharing-GPU share with SLO] [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing](https://www.usenix.org/system/files/atc22-choi-seungbeom.pdf). Seungbeom Choi, Sunho Lee, Yeonjae Kim, Jongse Park, Youngjin Kwon, and Jaehyuk Huh, KAIST. ATC'22

[accelerate moe] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/system/files/atc23-li-jiamin.pdf). Jiamin Li, City University of Hong Kong; Yimin Jiang, ByteDance Inc.; Yibo Zhu, unaffiliated; Cong Wang, City University of Hong Kong;Hong Xu, The Chinese University of Hong Kong. ATC'23

[accelerating billion-scale GNN training] [Legion Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training](https://www.usenix.org/system/files/atc23-sun.pdf). Jie Sun, Collaborative Innovation Center of Artificial Intelligence, Zhejiang University, China; Li Su, Alibaba Group; Zuocheng Shi, Collaborative Innovation Center of Artificial Intelligence,  Zhejiang University, China; Wenting Shen, Alibaba Group; Zeke Wang, Collaborative Innovation  Center of Artificial Intelligence, Zhejiang University, China; Lei Wang, Alibaba Group; Jie Zhang,  Collaborative Innovation Center of Artificial Intelligence, Zhejiang University, China; Yong Li,  Wenyuan Yu, and Jingren Zhou, Alibaba Group; Fei Wu, Collaborative Innovation Center of  Artificial Intelligence, Zhejiang University, China and Shanghai Institute for Advanced Study of  Zhejiang University, China. ATC'23 

[auto-parallelization in moe] [SMARTMOE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization](https://www.usenix.org/system/files/atc23-zhai.pdf). Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, Jidong Zhai. ATC'23 

[accelerate MoE] [Accelerating Distributed MoE Training and Inference with Lina](https://www.usenix.org/system/files/atc23-li-jiamin.pdf). Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, Hong Xu. ATC'23

## ASPLOS

[GPU memory management] [Capuchin: Tensor-based GPU Memory Management for Deep Learning](http://alchem.usc.edu/portal/static/download/capuchin.pdf). Xuan Peng, Xuanhua Shi, Hulin Dai, Hai Jin, Weiliang Ma, Qian Xiong, Fan Yang,  Xuehai Qian. ASPLOS'20

[swapping between GPU and CPU memory] [SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping](https://dl.acm.org/doi/pdf/10.1145/3373376.3378530). Chien-Chin Huang, Gu Jin, Jinyang Li. ASPLOS'20

[distributed supernet training system] [NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism](https://dl.acm.org/doi/pdf/10.1145/3503222.3507735). Shixiong Zhao, Fanxin Li, Xusheng Chen, Tianxiang Shen, Li Chen, Sen Wang, Nicholas Zhang, Cheng Li, Heming Cui. ASPLOS'22

[deep learning workload scheduler] [Lucid: A Non-intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs](https://dl.acm.org/doi/pdf/10.1145/3575693.3575705). Qinghao Hu, Meng Zhang, Peng Sun, Yonggang Wen, Tianwei Zhang. ASPLOS'23

[distributed training framework] [Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression](https://dl.acm.org/doi/pdf/10.1145/3575693.3575712). Jaeyong Song, Jinkyu Yim, Jaewon Jung, Hongsun Jang, Hyung-Jin Kim, Youngsok Kim, Jinho Lee. ASPLOS'23

[fine Tuning] [Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers](https://dl.acm.org/doi/pdf/10.1145/3575693.3575703). Yangyang Feng, Minhui Xie, Zijie Tian, Shuo Wang, Youyou Lu, Jiwu Shu. ASPLOS'23  

[GNN training] [Betty: Enabling Large-Scale GNN Training with Batch-Level Graph Partitioning](https://dl.acm.org/doi/pdf/10.1145/3575693.3575725). Shuangyan Yang, Minjia Zhang, Wenqian Dong, Dong Li. ASPLOS'23 

[overlap communication with computation] [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959). ASPLOS'23 

[tensor management] [DeepUM: Tensor Migration and Prefetching in Unified Memory](https://dl.acm.org/doi/pdf/10.1145/3575693.3575736). Jaehoon Jung, Jinpyo Kim, Jaejin Lee. ASPLOS'23 

[Tensor Fusion] [FLAT: An Optimized Dataflow forMitigating Attention Bottlenecks](https://dl.acm.org/doi/pdf/10.1145/3575693.3575747). Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, Amir Yazdanbakhsh, Tushar Krishna. ASPLOS'23



## NAACL-HLT

[language representation model] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf). Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. NAACL-HLT'19

##  ICML

[fine-tuning] [BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://arxiv.org/pdf/1902.02671.pdf). Asa Cooper Stickland, Iain Murray. ICML'19

[fine-tuning] [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf). Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly. ICML'19

[pipeline-parallel] [Memory-Efficient Pipeline-Parallel DNN Training](http://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf). Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, Matei Zaharia. ICML'21

[MoE training and inference solution] [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf). Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He.  ICML'22

[scale language models with MoE] [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c/du22c.pdf). Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, Claire Cui. ICML'22

[MoE training and inference] [DeepSpeed-MoE Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf). Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi. ICML'22

[robustness to attack] [Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations](https://proceedings.mlr.press/v162/alam22a/alam22a.pdf). Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt. ICML'22

[transfer learning] [Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer](https://proceedings.mlr.press/v162/alegre22a/alegre22a.pdf). Lucas N. Alegre, Ana L. C. Bazzan, Bruno C. da Silva. ICML'22

[explainable ai] [XAI for Transformers: Better Explanations through Conservative Propagation](https://proceedings.mlr.press/v162/ali22a/ali22a.pdf). Ameen Ali, Thomas Schnake, Oliver Eberle, Gregoire Montavon, Klaus Robert Muller, Lior Wolf. ICML'22

[LLM serving system] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192.pdf). Yaniv Leviathan, Matan Kalman, Yossi Matias. ICML'23

[offload in inference] [FlexGen High-Throughput Generative Inference of Large Language Models with a Single GPU](https://openreview.net/pdf?id=RRntzKrBTp). Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher ReIon Stoica, Ce Zhang. ICML'23

[KV cache eviction policy] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://openreview.net/pdf?id=ctPizehA9D). Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen. ICML'23

[pipeline parallelism] [BPIPE: Memory-Balanced Pipeline Parallelism for Training Large Language Models](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf). Taebum Kim, Hyoungjoo Kim,Gyeong-In Yu, Byung-Gon Chun. ICML'23

## EUROSYS

[graph sampling on GPUs] [Accelerating Graph Sampling for Graph Machine Learning using GPUs](https://dl.acm.org/doi/pdf/10.1145/3447786.3456244). Abhinav Jangda, Sandeep Polisetty, Arjun Guha, Marco Serafini. EUROSYS'21

[distributed, fair share scheduler] [Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3342195.3387555). Shubham Chaudhary, Ramachandran Ramjee, Muthian Sivathanu, Nipun Kwatra, Nipun Kwatra. EUROSYS'21

[distributed framework for GNN training] [FlexGraph: A Flexible and Efficient Distributed Framework for GNN Training](https://ipads.se.sjtu.edu.cn/_media/publications/flexgraph-eurosys21.pdf). Lei Wang, Qiang Yin, Chao Tian, Jianbang Yang, Rong Chen, Wenyuan Yu, Zihang Yao, Jingren Zhou. EUROSYS'21

[giant model training in cluster] [Varuna: scalable, low-cost training of massive deep learning models](https://dl.acm.org/doi/pdf/10.1145/3492321.3519584). Sanjith Athlur, Nitika Saran, Muthian Sivathanu, Ramachandran Ramjee, Nipun Kwatra. EUROSYS'22

[GPU resource usage] [GNNLab: A Factored System for Sample-based GNN Training over GPUs](https://dl.acm.org/doi/pdf/10.1145/3492321.3519557). Jianbang Yang, Dahai Tang, Xiaoniu Song, Lei Wang, Qiang Yin, Rong Chen, Wenyuan Yu, Jingren Zhou. EUROSYS'22

[GPU-resident cache] [Fleche: An Efficient GPU Embedding Cache for Personalized Recommendations](https://dl.acm.org/doi/pdf/10.1145/3492321.3519554). Minhui Xie, Youyou Lu, Jiazhen Lin, Qing Wang, Jian Gao, Kai Ren, Jiwu Shu. EUROSYS'22

[training DNN in GPU] [Out-Of-Order BackProp: An Effective Scheduling Technique for Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3492321.3519563). Hyungjun Oh, Junyeol Lee, Hyeongju Kim, Jiwon Seo. EUROSYS'22

[inference system] [Tabi: An Efficient Multi-Level Inference System for Large Language Models](https://yidingwang.xyz/public/files/tabi_eurosys23.pdf). Yiding Wang, Kai Chen, Haisheng Tan, Kun Guo. EUROSYS'23

[serving with direct-host-access] [Fast and Efficient Model Serving Using Multi-GPUs  with Direct-Host-Access](https://dl.acm.org/doi/pdf/10.1145/3552326.3567508). Jinwoo Jeong, Seungsu Baek, Jeongseob Ahn. EUROSYS'23

[gradient compression] [Hi-Speed DNN Training with Espresso Unleashing the Full Potential of Gradient Compression with Near-Optimal Usage Strategies](https://dl.acm.org/doi/pdf/10.1145/3552326.3567505). Zhuang Wang, Haibin Lin, Yibo Zhu, T. S. Eugene Ng. EUROSYS'23

## KDD

[large model training] [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://dl.acm.org/doi/pdf/10.1145/3394486.3406703). Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, Yuxiong He. KDD'20

## OSDI

[distributed DNN training] [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://dl.acm.org/doi/pdf/10.5555/3488766.3488792). Yimin Jiang, Tsinghua University and ByteDance; Yibo Zhu, ByteDance; Chang Lan, Google; Bairen Yi, ByteDance; Yong Cui, Tsinghua University; Chuanxiong Guo, ByteDance. osdi'20

[dynamic scaling on GPU clusters] [AntMan: Dynamic Scaling on GPU Clusters for Deep Learning](https://dl.acm.org/doi/pdf/10.5555/3488766.3488796). Wencong Xiao, Shiru Ren, Yong Li, Yang Zhang, Pengyang Hou, Zhi Li, Yihui Feng, Wei Lin, and Yangqing Jia, Alibaba Group. osdi'20

[CPU scheduler] [Caladan: Mitigating Interference at Microsecond Timescales](https://dl.acm.org/doi/pdf/10.5555/3488766.3488782). Joshua Fried and Zhenyuan Ruan, MIT CSAIL; Amy Ousterhout, UC Berkeley; Adam Belay, MIT CSAIL. osdi'20

[heterogeneity aware scheduler] [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](https://dl.acm.org/doi/pdf/10.5555/3488766.3488793). Deepak Narayanan and Keshav Santhanam, Stanford University and Microsoft Research; Fiodar Kazhamiaka, Stanford University; Amar Phanishayee, Microsoft Research; Matei Zaharia, Stanford University. osdi'20

[framework to share a GPU cluster safely] [HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees](https://www.usenix.org/system/files/osdi20-zhao_hanyu.pdf). Hanyu Zhao, Zhenhua Han, Zhi Yang, Quanlu Zhang, Fan Yang, Lidong Zhou, Mao Yang, Francis C.M. Lau, Yuqi Wang, Yifan Xiong, Bin Wang. osdi'20

[adaptive training] [KungFu: Making Training in Distributed Machine Learning Adaptive](https://dl.acm.org/doi/pdf/10.5555/3488766.3488819).  Luo Mai, Guo Li, Marcel Wagenländer, Konstantinos Fertakis, Andrei-Octavian Brabete, and Peter Pietzuch, Imperial College London. osdi'20

[rack-scale computer scheduler] [RackSched: A Microsecond-Scale Scheduler for Rack-Scale Computers](https://www.usenix.org/system/files/osdi20-zhu.pdf). Hang Zhu, Johns Hopkins University; Kostis Kaffes, Stanford University; Zixu Chen, Johns Hopkins University; Zhenming Liu, College of William and Mary; Christos Kozyrakis, Stanford University; Ion Stoica, UC Berkeley; Xin Jin, Johns Hopkins University. osdi'20

[distributed model serving system] [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up](https://dl.acm.org/doi/pdf/10.5555/3488766.3488791). Arpan Gujarati, Reza Karimi, Safya Alzayat, Wei Hao, Antoine Kaufmann,  Ymir Vigfusson. osdi'20

[distributed system for training GNNs] [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads](https://cs.stanford.edu/~zhihao/papers/Dorylus_OSDI21.pdf). John Thorpe, Yifan Qiao, Jonathan Eyolfson, and Shen Teng, UCLA; Guanzhou Hu, UCLA and University of Wisconsin, Madison; Zhihao Jia, CMU; Jinliang Wei, Google Brain; Keval Vora, Simon Fraser; Ravi Netravali, Princeton University; Miryung Kim and Guoqing Harry Xu, UCLA. osdi'21

[GNN acceleration] [GNNAdvisor: An Adaptive and Efficient Runtime System for GNN Acceleration on GPUs](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf). Yuke Wang, Boyuan Feng, Gushu Li, Shuangchen Li, Lei Deng, Yuan Xie, and Yufei Ding, University of California, Santa Barbara. osdi'21

[embeddings of large-scale graphs] [Marius: Learning Massive Graph Embeddings on a Single Machine](https://www.usenix.org/system/files/osdi21-mohoney.pdf). Jason Mohoney and Roger Waleffe, University of Wisconsin-Madison; Henry Xu, University of Maryland, College Park; Theodoros Rekatsinas and Shivaram Venkataraman, University of Wisconsin-Madison. osdi'21

[schedulingin deep learning clusters] [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning](https://www.usenix.org/system/files/osdi21-qiao.pdf). Aurick Qiao, Petuum, Inc. and Carnegie Mellon University; Sang Keun Choe and Suhas Jayaram Subramanya, Carnegie Mellon University; Willie Neiswanger, Petuum, Inc. and Carnegie Mellon University; Qirong Ho, Petuum, Inc.; Hao Zhang, Petuum, Inc. and UC Berkeley; Gregory R. Ganger, Carnegie Mellon University; Eric P. Xing, MBZUAI, Petuum, Inc., and Carnegie Mellon University. osdi'21

[distributed serving system] [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf). Gyeong-In Yu and Joo Seong Jeong, Seoul National University; Geon-Woo Kim, FriendliAI and Seoul National University; Soojeong Kim, FriendliAI; Byung-Gon Chun, FriendliAI and Seoul National University. osdi'22

[recommender system] [Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency Model Update](https://www.usenix.org/system/files/osdi22-sima.pdf). Chijun Sima, Tencent; Yao Fu and Man-Kit Sit, The University of Edinburgh; Liyi Guo, Xuri Gong, Feng Lin, Junyu Wu, Yongsheng Li, and Haidong Rong, Tencent; Pierre-Louis Aublin, IIJ research laboratory; Luo Mai, The University of Edinburgh. osdi'22

[resource sensitive scheduler for shared GPU clusters] [Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters](https://www.usenix.org/system/files/osdi22-mohan.pdf). Jayashree Mohan, Amar Phanishayee, and Janardhan Kulkarni, Microsoft Research; Vijay Chidambaram, The University of Texas at Austin and VMware Research. osdi'22

[distributed DNN training] [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/system/files/osdi22-unger.pdf). Colin Unger, Stanford University; Zhihao Jia, Carnegie Mellon University and Meta; Wei Wu, Los Alamos National Laboratory and NVIDIA; Sina Lin, Microsoft; Mandeep Baines, Carlos Efrain Quintero Narvaez, Vinay Ramakrishnaiah, Nirmal Prajapati, Pat McCormick, Jamaludin Mohd-Yusof, Xi Luo, Dheevatsa Mudigere, Jongsoo Park, Misha Smelyanskiy, Alex Aiken. osdi'22

[DNN inference] [Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences](https://www.usenix.org/system/files/osdi22-han.pdf). Mingcong Han, Institute of Parallel and Distributed Systems, SEIEE, Shanghai Jiao Tong University; Shanghai AI Laboratory; Hanze Zhang, Institute of Parallel and Distributed Systems, SEIEE, Shanghai Jiao Tong University; MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University, China; Rong Chen, Institute of Parallel and Distributed Systems, SEIEE, Shanghai Jiao Tong University; Shanghai AI Laboratory; Haibo Chen, Institute of Parallel and Distributed Systems, SEIEE, Shanghai Jiao Tong University; Engineering Research Center for Domain-specific Operating Systems, Ministry of Education, China. osdi'22

[device-cloud collaborative machine learning] [Walle: An End-to-End, General-Purpose, and Large-Scale Production System for Device-Cloud Collaborative Machine Learning](https://www.usenix.org/system/files/osdi22-lv.pdf). Chengfei Lv, Zhejiang University and Alibaba Group; Chaoyue Niu, Shanghai Jiao Tong University and Alibaba Group; Renjie Gu, Xiaotang Jiang, Zhaode Wang, Bin Liu, Ziqi Wu, Qiulin Yao, Congyu Huang, Panos Huang, Tao Huang, Hui Shu, Jinde Song, Bin Zou, Peng Lan, and Guohuan Xu, Alibaba Group; Fei Wu, Zhejiang University; Shaojie Tang, University of Texas at Dallas; Fan Wu and Guihai Chen, Shanghai Jiao Tong University. osdi'22

[automating parallelism] [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf). Lianmin Zheng, Zhuohan Li, and Hao Zhang, UC Berkeley; Yonghao Zhuang, Shanghai Jiao Tong University; Zhifeng Chen and Yanping Huang, Google; Yida Wang, Amazon Web Services; Yuanzhong Xu, Google; Danyang Zhuo, Duke University; Eric P. Xing, MBZUAI and Carnegie Mellon University; Joseph E. Gonzalez and Ion Stoica, UC Berkeley. osdi'22

[model parallelism] [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf). Zhuohan Li and Lianmin Zheng, UC Berkeley; Yinmin Zhong, Peking University; Vincent Liu, University of Pennsylvania; Ying Sheng, Stanford University; Xin Jin, Peking University; Yanping Huang and Zhifeng Chen, Google; Hao Zhang, UC San Diego; Joseph E. Gonzalez and Ion Stoica, UC Berkeley. osdi'23

[deep learning compiler] [Welder Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/system/files/osdi23-shi.pdf). Yining Shi, Zhi Yang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Ziming Miao, Yuxiao Guo, Fan Yang, Lidong Zhou. osdi'23

## SOSP

[pipeline parallelism] [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/pdf/10.1145/3341301.3359646). Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, Matei Zaharia. SOSP'19

[gradient compression] [Gradient Compression Supercharged High-Performance Data Parallel DNN Training](https://dl.acm.org/doi/10.1145/3477132.3483553). Youhui Bai, Cheng Li, Quan Zhou, Jun Yi, Ping Gong, Feng Yan, Ruichuan Chen, Yinlong Xu. SOSP'21

[sharing of KV cache] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf). Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. SOSP'23

## ICPP

[tensor parallelism] [Tesseract: Parallelize the Tensor Parallelism Efficiently](https://arxiv.org/pdf/2105.14500.pdf). Boxiang Wang, Qifan Xu, Zhengda Bian, Yang You. ICPP'22

[multiple inference tasks sharing single GPU] [SPLIT: QoS-Aware DNN Inference on Shared GPU via Evenly-Sized Model Splitting](https://dl.acm.org/doi/pdf/10.1145/3605573.3605627). Diaohan Luo, Tian Yu, Yuewen Wu, Heng Wu, Tao Wang, Wenbo Zhang. ICPP'23

[efficient all-reduce] [Wrht: Efficient All-reduce for Distributed DNN Training in Optical Interconnect Systems](https://dl.acm.org/doi/pdf/10.1145/3605573.3605624). Fei Dai, Yawen Chen, Zhiyi Huang, Haibo Zhang. ICPP'23

[cpu offload] [CoTrain: Efficient Scheduling for Large-Model Training upon GPU and CPU in Parallel](https://dl.acm.org/doi/pdf/10.1145/3605573.3605647). Zhenxing Li, Qiang Cao, Yajie Chen, Wenrui Yan. ICPP'23

[efficient communication in ddl] [OSP: Boosting Distributed Model Training with 2-stage Synchronization](https://dl.acm.org/doi/pdf/10.1145/3605573.3605650). Zixuan Chen, Lei Shi, Xuandong Liu, Jiahui Li, Sen Liu, Yang Xu. ICPP'23

[automatic parallelization] [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://dl.acm.org/doi/pdf/10.1145/3605573.3605613). Shenggui Li, Hongxin Liu, Zhengda Bian, Jiarui Fang, Haichen Huang, Yuliang Liu, Boxiang Wang, Yang You. ICPP'23

[training upon GPU and CPU] [CoTrain: Efficient Scheduling for Large-Model Training upon GPU and CPU in Parallel](https://dl.acm.org/doi/pdf/10.1145/3605573.3605647). Zhenxing Li, Qiang Cao, Yajie Chen, Wenrui Yan. ICPP'23

## EMNLP

[fine-tuning] [AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/pdf/2007.07779.pdf). Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, Iryna Gurevych. EMNLP'20

[fine-tuning] [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691). Brian Lester, Rami Al-Rfou, Noah Constant. EMNLP'21

## ACL

[fine-tuning] [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190). Xiang Lisa Li, Percy Liang. ALC'21

[fine-tuning] [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/pdf/2012.15723.pdf). Ziyun Xu, Chengyu Wang, Minghui Qiu, Fuli Luo, Runxin Xu, Songfang Huang, Jun Huang. ACL'21

[fine-tuning] [Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks](https://arxiv.org/pdf/2106.04489.pdf). Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, James Henderson. ACL'21

[fine-tuning] [MSP: Multi-Stage Prompting for Making Pre-trained Language Models Better Translators](https://arxiv.org/pdf/2110.06609.pdf). Zhixing Tan, Xiangwen Zhang, Shuo Wang, Yang Liu. ACL'22

[fine-tuning] [Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/pdf/2108.02035.pdf). Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li, Wei Wu, Maosong Sun. ACL'22

[routing strategy for MoE] [StableMoE: Stable Routing Strategy for Mixture of Experts](https://aclanthology.org/2022.acl-long.489.pdf). Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, Furu Wei. ACL'22 

[large language model] [What Language Model to Train if You Have One Million GPU Hours?](https://aclanthology.org/2022.findings-emnlp.54.pdf). Teven Le Scao, Thomas Wang, Daniel Hesslow, Stas Bekman, M Saiful Bari, Stella Biderman, Hady Elsahar, Niklas Muennighoff, Jason Phang, Ofir Press, Colin Raffel, Victor Sanh, Sheng Shen, Lintang Sutawika, Jaesung Tae, Zheng Xin Yong, Julien Launay, Iz Beltagy. ACL'22

## ICLR

[large mini-batches] [LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76 MINUTES](https://openreview.net/pdf?id=Syx4wnEtvH). Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh. ICLR'20

[scaling giant models] [Gshard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://openreview.net/pdf?id=qrwe7XHTmYb). Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen. ICLR'21

[transformer for image recognition] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy). Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. ICLR'21

[expert-based model] [Taming Sparsely Activated Transformer with Stochastic Experts](https://openreview.net/pdf?id=B72HXs80q4). Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, Jianfeng Gao. ICLR'22

[large language model] [GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL](https://openreview.net/pdf?id=-Aw0rrrPUF). Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, Jie Tang. ICLR'23

[transformer block] [Brainformers Trading Simplicity for Efficiency](https://openreview.net/pdf?id=w5q6tHO1dl1). Yanqi Zhou, Nan Du, Yanping Huang, Daiyi Peng, Chang Lan, Da Huang, Siamak Shakeri, David So, Andrew Dai, Yifeng Lu, Zhifeng Chen, Quoc Le, Claire Cui, James Laundon, Jeff Dean. ICLR'23 

[rlhf] [SAFE RLHF: SAFE REINFORCEMENT LEARNING FROM HUMAN FEEDBACK](https://openreview.net/pdf?id=TyFrPOKYXw). Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang. ICLR'24

## VLDB

[data parallelism] [PyTorch distributed: experiences on accelerating data parallel training](https://arxiv.org/pdf/2006.15704.pdf). Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, Soumith Chintala. VLDB'20

[automatic parallelism] [Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism](https://arxiv.org/pdf/2211.13878.pdf). Xupeng Miao, Yujie Wang, Youhe Jiang, Chunan Shi, Xiaonan Nie, Hailin Zhang, Bin Cui. VLDB'22

## NSDI

[workload analysis and scheduling] [MLaaS: in the Wild Workload Analysis and Scheduling in Large-Scale Heterogeneous](https://www.usenix.org/system/files/nsdi22-paper-weng.pdf). Qizhen Weng, Hong Kong University of Science and Technology and Alibaba Group; Wencong Xiao, Alibaba Group; Yinghao Yu, Alibaba Group and Hong Kong University of Science and Technology; Wei Wang, Hong Kong University of Science and Technology; Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin, and Yu Ding, Alibaba Group. nsdi'22

## AAAI

[model framework] [Go Wider Instead of Deeper](https://cdn.aaai.org/ojs/20858/20858-13-24871-1-2-20220628.pdf). Fuzhao Xue, Ziji Shi, Futao Wei, Yuxuan Lou, Yong Liu, Yang You. AAAI'22

## ICCV

[vision transformer] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf). Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. ICCV'21

## FAST

[offloading data to SSD] [FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks](https://www.usenix.org/system/files/fast21-bae.pdf). Jonghyun Bae, Seoul National University; Jongsung Lee, Seoul National University and Samsung Electronics; Yunho Jin and Sam Son, Seoul National University; Shine Kim, Seoul National University and Samsung Electronics; Hakbeom Jang, Samsung Electronics; Tae Jun Ham and Jae W. Lee, Seoul National University. FAST'21

## ISCA

[GPU memory expansion] [Buddy Compression: Enabling Larger Memory for Deep Learning and HPC Workloads on GPUs](https://research.nvidia.com/sites/default/files/pubs/2020-06_Buddy-Compression%3A-Enabling/chouske.isca2020.pdf). Esha Choukse, Michael B. Sullivan, Mike O’Connor, Mattan Erez, Jeff Pool, David Nellans, Stephen W. Keckler. ISCA'20

## HPCA

[tensor management] [Sentinel: Efficient Tensor Migration and Allocation on Heterogeneous Memory Systems for Deep Learning](http://www.pasalabs.org/papers/2021/hpca21_sentinel.pdf). Jie Ren,  Jiaolin Luo, Kai Wu, Minjia Zhang,  Hyeran Jeon, Dong Li. HPCA'21

[memory-saving]  [MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism](https://ieeexplore.ieee.org/document/10071077). Quan Zhou, Haiquan Wang, Xiaoyan Yu, Cheng Li, Youhui Bai, Feng Yan, Yinlong. HPCA'23

[alleviate PCIe channel contention] [Tensor Movement Orchestration in Multi-GPU Training Systems](https://ieeexplore.ieee.org/document/10071043). Shao-Fu Lin, Yi-Jung Chen, Hsiang-Yun Cheng, Chia-Lin Yang. HPCA'23

[operators fusion] [Chimera An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071018). Size Zheng, Siyuan Chen, Peidi Song, Renze Chen,  Xiuhong Li, Shengen Yan, Dahua Lin, Jingwen Leng, Yun Liang. HPCA'23

[reducing computation complexity in attention] [CTA Hardware-Software Co-design for Compressed Token Attention Mechanism](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10070997). Haoran Wang, Haobo Xu, Ying Wang, Yinhe Han. HPCA'23

[unified virtual memory] [Trans-FW Short Circuiting Page Table Walk in Multi-GPU Systems via Remote Forwarding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071054) Bingyao Li, Jieming Yin, Anup Holey, Youtao Zhang, Jun Yang, Xulong Tang. HPCA'23

[boosting the inference efficiency of ViTs] [ViTALiTy Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071081). Jyotikrishna Dass, Shang Wu, Huihong Shi, Chaojian Li, Zhifan Ye, Zhongfeng Wang, Yingyan Lin. HPCA'23

## INFOCOM

[hide the communication] [PipeMoE: Accelerating Mixture-of-Experts through Adaptive Pipelining](https://ieeexplore.ieee.org/document/10228874). Shaohuai Shi, Xinglin Pan, Xiaowen Chu, Bo Li. INFOCOM'23

## IPDPS

[accelerate MoE training] [MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism](https://liamding.cc/slides/MPipeMoE_IPDPS23.pdf). Zheng Zhang, Donglin Yang, Yaqi Xia, Liang Ding, Dacheng Tao, Xiaobo Zhou, Dazhao Cheng. IPDPS'23

## MLSys

[memory reuse] [SAFE OPTIMIZED STATIC MEMORY ALLOCATION FOR PARALLEL DEEP LEARNING](https://proceedings.mlsys.org/paper_files/paper/2023/file/0c8abcf158ed12d0dd94480681186fda-Paper-mlsys2023.pdf). Ioannis Lamprou, Zhen Zhang, Javier de Juan, Hang Yang, Yongqiang Lai, Etienne Filhol, Cedric Bastoul. MLSys'23

[moe kernel] [MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS](https://proceedings.mlsys.org/paper_files/paper/2023/file/f9f4f0db4894f77240a95bde9df818e0-Paper-mlsys2023.pdf). Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. MLSys'23

## SIGCOMM

[moe all2all] [Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/pdf/10.1145/3603269.3604869). Juncai Liu, Jessie Hui Wang, Yimin Jiang. SIGCOMM'23

## Journals

## TPDS

[parallel training] [PatrickStar: Parallel Training of Pre-Trained Models via Chunk-Based Dynamic Memory Management](https://arxiv.org/pdf/2108.05818.pdf). Jiarui Fang, Zilin Zhu, Shenggui Li, Hui Su, Yang Yu, Jie Zhou, Yang You. TPDS'23

[3D parallel] [Merak: An Efficient Distributed DNN Training Framework With Automated 3D Parallelism for Giant Foundation Models](https://arxiv.org/pdf/2206.04959). Zhiquan Lai, Shengwei Li, Xudong Tang, Keshi Ge, Weijie Liu, Yabo Duan, Linbo Qiao, Dongsheng Li. TPDS'23

## ACMComputingSurveys

[fine-tuning] [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://dl.acm.org/doi/pdf/10.1145/3560815). Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig. ACM Computing Surveys'23

## JournalofMachineLearningResearch

[training with lower precision] [Switch Transformers Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://dl.acm.org/doi/pdf/10.5555/3586589.3586709). William Fedus, Barret Zoph, Noam Shazeer. Journal of Machine Learning Research'22

## TMLR

[rlhf] [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment](https://arxiv.org/pdf/2304.06767.pdf). Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, Tong Zhang. TMLR'23

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)


This list is released into the public domain.