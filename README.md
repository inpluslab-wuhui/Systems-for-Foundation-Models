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

- [The IEEE / CVF Computer Vision and Pattern Recognition Conference(CVPR)](#CVPR)

   


## Table of Listed Journals
- [IEEE Transactions on Parallel and Distributed Systems (TPDS)](#TPDS)
- [ACMComputingSurveys](#ACMComputingSurveys)
- [JournalofMachineLearningResearch](#JournalofMachineLearningResearch)
- [Transactions on Machine Learning Research(TMLR)](#TMLR)
- [IEEE Transactions on Mobile Computing(TMC)](#TMC)

## Conferences

## PPoPP

[dynamic GPU memory scheduling] [Superneurons: dynamic GPU memory management for training deep neural networks](https://dl.acm.org/doi/pdf/10.1145/3200691.3178491). Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuaiwen Leon Song, Zenglin Xu, Tim Kraska. PPoPP'18

[training on supercomputer] [BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/ppopp22-bagualu/ppopp22-bagualu.pdf). Zixuan Ma1 , Jiaao He1 , Jiezhong Qiu1,4, Huanqi Cao1 , Yuanwei Wang1 , Zhenbo Sun1 , Liyan Zheng1 , Haojie Wang1 , Shizhi Tang1 , Tianyu Zheng3 , Junyang Lin2 , Guanyu Feng1 , Zeqiang Huang3 , Jie Gao3 , Aohan Zeng1,4, Jianwei Zhang2 , Runxin Zhong1 , Tianhui Shi1 , Sha Liu3 , Weimin Zheng1 , Jie Tang1,4, Hongxia Yang2 , Xin Liu3 , Jidong Zhai1 , Wenguang Chen1. PPoPP'22

[distributed MoE model training] [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/ppopp22-fastermoe/ppopp22-fastermoe.pdf). Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, Qin Li. PPoPP'22

[pipeline parallelism] [Elastic Averaging for Efficient Pipelined DNN Training](https://doi.org/10.1145/3572848.3577484). Zihao Chen, Chen Xu, Weining Qian, Aoying Zhou. PPoPP'23

[sparse attention] [Dynamic N M Fine-grained Structured Sparse Attention Mechanism](https://dl.acm.org/doi/pdf/10.1145/3572848.3577500). Zhaodong Chen, Zheng Qu, Yuying Quan, Liu Liu, Yufei Ding, Yuan Xie. PPoPP'23

[failure recovery] [POSTER Swift Expedited Failure Recovery for Large-scale DNN Training](https://arxiv.org/pdf/2302.06173.pdf). Yuchen Zhong, Guangming Sheng, Juncheng Liu, Jinhui Yuan, and Chuan Wu. PPoPP'23

[computation and communication overlap] [Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference](https://dl.acm.org/doi/pdf/10.1145/3627535.3638466). Jiangsu Du, PictureJinhui Wei, PictureJiazhi Jiang, PictureShenggan Cheng, PictureDan Huang, PictureZhiguang Chen, PictureYutong Lu. PPoPP'24

[moe all2all] [Harnessing Inter-GPU Shared Memory for Seamless MoE Communication-Computation Fusion](https://dl.acm.org/doi/pdf/10.1145/3710848.3710868). Hulin Wang, Yaqi Xia, Donglin Yang, Xiaobo Zhou, Dazhao Cheng. PPoPP'25

[activation checkpointing in pipeline parallelism] [Mario: Near Zero-cost Activation Checkpointing in Pipeline Parallelism](https://dl.acm.org/doi/pdf/10.1145/3710848.3710878). Weijian Liu, Mingzhen Li, Guangming Tan, Weile Jia. PPoPP'25

[weight-passing pipeline] [WeiPipe: Weight Pipeline Parallelism for Communication-Effective Long-Context Large Model Training](https://dl.acm.org/doi/pdf/10.1145/3710848.3710869). Junfeng Lin, Ziming Liu, Yang You, Jun Wang, Weihao Zhang, Rong Zhao. PPoPP'25

[weight quantization] [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://dl.acm.org/doi/pdf/10.1145/3710848.3710871). Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh. PPoPP'25

## NIPS

[network architecture] [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. NIPS'17

[parallel decoding] [Blockwise Parallel Decoding for Deep Autoregressive Models](https://proceedings.neurips.cc/paper_files/paper/2018/file/c4127b9194fe8562c64dc0f5bf2c93bc-Paper.pdf). Mitchell Stern, Noam Shazeer, Jakob Uszkoreit. NIPS'18

[pipeline parallelism] [GPipe: efficient training of giant neural networks using pipeline parallelism](https://dl.acm.org/doi/pdf/10.5555/3454287.3454297). Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen. NIPS'19

[pre-training] [Language Models are Few-Shot Learners](https://dl.acm.org/doi/pdf/10.5555/3495724.3495883). Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. NIPS'20

[fine-tuning] [COMPACTER:Efficient Low-Rank Hypercomplex Adapter Layers](https://arxiv.org/pdf/2106.04647.pdf). Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder. NIPS'21

[reinforcement learning] [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345). 	Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch. NIPS'21

[moe routing] [Hash Layers For Large Sparse Models](https://openreview.net/pdf?id=lMgDDWb1ULW). Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason E Weston. NIPS'21

[moe model] [Scaling Vision with Sparse Mixture of Experts](https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf). Carlos Riquelme, Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, Neil Houlsby. NIPS'22

[length generalization] [Exploring Length Generalization in Large Language Models](https://arxiv.org/pdf/2207.04901). Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay V. Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, Behnam Neyshabur. NIPS'22

[model compression] [XTC: Extreme Compression for Pre-trained Transformers Made Simple and Efficient](https://arxiv.org/pdf/2206.01859). Xiaoxia Wu, Zhewei Yao, Minjia Zhang, Conglong Li, Yuxiong He. NIPS'22

[zero-shot] [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/pdf/2202.04538). Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han. NIPS'22

[memory footprint reduction] [Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction](https://arxiv.org/pdf/2210.10246). Muralidhar Andoorveedu, Zhanda Zhu, Bojian Zheng, Gennady Pekhimenko. NIPS'22

[model compression] [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/pdf/2206.01861). 	Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, Yuxiong He. NIPS'22

[moe routing] [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf). Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, James Laudon. NIPS'22

[moe model] [Towards Understanding the Mixture-of-Experts Layer in Deep Learning](https://openreview.net/pdf?id=MaYzugDmQV). Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, Yuanzhi Li. NIPS'22

[moe model] [Uni-Perceiver-MoE: Learning Sparse Generalist Models with Conditional MoEs](https://openreview.net/pdf?id=agJEk7FhvKL). Jinguo Zhu, Xizhou Zhu, Wenhai Wang, Xiaohua Wang, Hongsheng Li, Xiaogang Wang, Jifeng Dai. NIPS'22

[rlhf] [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://arxiv.org/pdf/2306.01693.pdf). Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, Hannaneh Hajishirzi. NIPS'23

[rlhf] [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/pdf/2304.05302.pdf). Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang. NIPS'23

[parallel sampling] [Parallel Sampling of Diffusion Models](https://arxiv.org/pdf/2305.16317). Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari. NIPS'23

[diffusion on mobile] [SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds](https://arxiv.org/pdf/2306.00980). Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, Jian Ren. NIPS'23

[dynamic sparse attention] [MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention](https://openreview.net/pdf?id=fPBACAbqSN). Huiqiang Jiang, YUCHENG LI, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. NIPS'24

[operator] [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://openreview.net/pdf?id=tVConYid20). Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. NIPS'24

[speculative decoding] [Sequoia: Scalable and Robust Speculative Decoding](https://openreview.net/pdf?id=rk2L9YGDi2). Zhuoming Chen, Avner May, Ruslan Svirschevski, Yu-Hsun Huang, Max Ryabinin, Zhihao Jia, Beidi Chen. NIPS'24

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

[elastic training system] [EasyScale: Accuracy-consistent Elastic Training for Deep Learning](https://arxiv.org/pdf/2208.14228). Mingzhen Li, Wencong Xiao, Biao Sun, Hanyu Zhao, Hailong Yang, Shiru Ren, Zhongzhi Luan, Xianyan Jia, Yi Liu, Yong Li, Wei Lin, Depei Qian. SC'23

[gpu configuration] [LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services](https://ieeexplore.ieee.org/document/10793215 ). Malgorzata Lazuka, Andreea Anghel, Thomas Parnell. SC'24

[pipeline acceleration] [PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation](https://ieeexplore.ieee.org/document/10793190 ). Branden Butler, Sixing Yu, Arya Mazaheri, Ali Jannesari. SC'24

[large-scale training] [Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers](https://ieeexplore.ieee.org/document/10793182 ). Siddharth Singh, Prajwal Singhania, Aditya Ranjan. SC'24

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

[DNN training in GPU] [HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism](https://www.usenix.org/system/files/atc20-paper1132-slides-park.pdf). Jay H. Park, Gyeongchan Yun, Chang M. Yi, Nguyen T. Nguyen, and Seungmin Lee, UNIST; Jaesik Choi, KAIST; Sam H. Noh and Young-ri Choi, UNIST. ATC'20

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

[GPU fragmentation] [Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent](https://www.usenix.org/system/files/atc23-weng.pdf). Qizhen Weng and Lingyun Yang, Hong Kong University of Science and Technology; Yinghao Yu, Alibaba Group and Hong Kong University of Science and Technology; Wei Wang, Hong Kong University of Science and Technology; Xiaochuan Tang, Guodong Yang, and Liping Zhang, Alibaba Group. ATC'23

[multiple DNN on device] [Decentralized Application-Level Adaptive Scheduling for Multi-Instance DNNs on Open Mobile Devices](https://www.usenix.org/system/files/atc23-sung.pdf). HH Sung, JA Chen, W Niu, J Guan, B Ren, X Shen. ATC'23

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

[tensor fusion] [FLAT: An Optimized Dataflow forMitigating Attention Bottlenecks](https://dl.acm.org/doi/pdf/10.1145/3575693.3575747). Sheng-Chun Kao, Suvinay Subramanian, Gaurav Agrawal, Amir Yazdanbakhsh, Tushar Krishna. ASPLOS'23

[pipelining on edge] [STI: Turbocharge NLP Inference at the Edge via Elastic Pipelining](https://dl.acm.org/doi/pdf/10.1145/3575693.3575698). Liwei Guo, Wonkyo Choe, Felix Xiaozhu Lin. ASPLOS'23

[communication overlap] [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](https://arxiv.org/pdf/2401.16677). Suchita Pati, Shaizeen Aga, Mahzabeen Islam, Nuwan Jayasena, Matthew D. Sinclair. ASPLOS'24

[speculative inference] [SpecInfer: Accelerating Generative Large Language Model Serving with Traee-based Speculative Inference and Verification](https://arxiv.org/pdf/2305.09781). Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia. ASPLOS'24

[inference system] [ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference](https://arxiv.org/pdf/2404.07947). Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-seong Chang, Jiwon Seo. ASPLOS'24

[inference system] [Proteus: A High-Throughput Inference-Serving System with Accuracy Scaling](https://guanh01.github.io/files/2024proteus.pdf). Sohaib Ahmad, Hui Guan, Brian D. Friedman, Nokia Bell Labs, Nokia Bell Labs, Ramesh K. Sitaraman, Thomas Woo. ASPLOS'24

[preemptible inference system] [SpotServe: Serving Generative Large Language Models on Preemptible Instances](https://arxiv.org/pdf/2311.15566). Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, Zhihao Jia. ASPLOS'24

[pipeline parallelism] [AdaPipe: Optimizing Pipeline Parallelism with Adaptive Recomputation and Partitioning](https://dl.acm.org/doi/pdf/10.1145/3620666.3651359). Zhenbo Sun, PictureHuanqi Cao, PictureYuanwei Wang, PictureGuanyu Feng, PictureShengqi Chen, PictureHaojie Wang, PictureWenguang Chen. ASPLOS'24

[memory optimization] [MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN](https://dl.acm.org/doi/pdf/10.1145/3620666.3651330). Renze Chen, PictureZijian Ding, PictureSize Zheng, PictureChengrui Zhang, PictureJingwen Leng, PictureXuanzhe Liu, PictureYun Liang. ASPLOS'24

[tensor partitioning] [PrimePar: Efficient Spatial-temporal Tensor Partitioning for Large Transformer Model Training](https://dl.acm.org/doi/pdf/10.1145/3620666.3651357). Haoran Wang, PictureLei Wang, PictureHaobo Xu, PictureYing Wang, PictureYuming Li, PictureYinhe Han. ASPLOS'24

[layout transformation elimination] [SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile](https://arxiv.org/pdf/2404.13528). Wei Niu, Md Musfiqur Rahman Sanim, Zhihao Shu, Jiexiong Guan, Xipeng Shen, Miao Yin, Gagan Agrawal, Bin Ren. ASPLOS'24

[collective communication library] [TCCL: Discovering Better Communication Paths for PCIe GPU Clusters](https://dl.acm.org/doi/pdf/10.1145/3620666.3651362). Heehoon Kim, PictureJunyeol Ryu, PictureJaejin Lee. ASPLOS'24

[gnn training system] [MaxK-GNN: Extremely Fast GPU Kernel Design for Accelerating Graph Neural Networks Training](https://arxiv.org/pdf/2312.08656). Hongwu Peng, Xi Xie, Kaustubh Shivdikar, MD Amit Hasan, Jiahui Zhao, Shaoyi Huang, Omer Khan, David Kaeli, Caiwen Ding. ASPLOS'24

[private information retrieval] [GPU-based Private Information Retrieval for On-Device Machine Learning Inference](https://arxiv.org/pdf/2301.10904). Maximilian Lam, Jeff Johnson, Wenjie Xiong, Kiwan Maeng, Udit Gupta, Yang Li, Liangzhen Lai, Ilias Leontiadis, Minsoo Rhu, Hsien-Hsin S. Lee, Vijay Janapa Reddi, Gu-Yeon Wei, David Brooks, G. Edward Suh. ASPLOS'24

[model pruning] [Fractal: Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning](https://dl.acm.org/doi/pdf/10.1145/3620666.3651351). Yue Guan, Changming Yu, Yangjie Zhou, Jingwen Leng, Chao Li, and Minyi Guo. ASPLOS'24

[SpMM acceleration] [DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication with Tensor Cores](https://dl.acm.org/doi/pdf/10.1145/3620666.3651378). Ruibo Fan, Wei Wang, and Xiaowen Chu. ASPLOS'24

[elastic training] [Heet: Accelerating Elastic Training in Heterogeneous Deep Learning Clusters](https://dl.acm.org/doi/pdf/10.1145/3620665.3640375). Zizhao Mo, Huanle Xu, and Chengzhong Xu. APLOS'24

[training compiler] [EVT: Accelerating Deep Learning Training with
Epilogue Visitor Tree](https://dl.acm.org/doi/pdf/10.1145/3620666.3651369). Zhaodong Chen, Andrew Kerr, Richard Cai, Jack Kosaian, Haicheng Wu, Yufei Ding, and Yuan Xie. ASPLOS'24

[communication partitioning] [Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning](https://dl.acm.org/doi/pdf/10.1145/3620666.3651379). Chang Chen, Xiuhong Li, Qianchao Zhu, Jiangfei Duan, Peng Sun, Xingcheng Zhang, and Chao Yang. ASPLOS'24

[8-bit inference and fine-tuning] [8-bit Transformer Inference and Fine-tuning for Edge Accelerators](https://dl.acm.org/doi/pdf/10.1145/3620666.3651368). Jeffrey Yu, Kartik Prabhu, Yonatan Urman, Robert M. Radway, Eric Han, and Priyanka Raina. ASPLOS'24

[pim] [PIM-DL: Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization](https://dl.acm.org/doi/pdf/10.1145/3620665.3640376). Cong Li, Zhe Zhou, Yang Wang, Fan Yang, Ting Cao, Mao Yang, Yun Liang, Guangyu Sun. ASPLOS'24

[parallel approach in AI compiler] [Concerto: Automatic Communication Optimization and Scheduling for Large-Scale Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3669940.3707223). Cheng Shenggan, Lin Shengjie, Diao Lansong, Wu Hao, Wang Siyu, Si Chang, Liu Ziming, Zhao Xuanlei, Du Jiangsu, Lin Wei, You Yang. ASPLOS'25

[Serverless DL Serving] [Dilu: Enabling GPU Resourcing-on-Demand for Serverless DL Serving via Introspective Elasticity](https://dl.acm.org/doi/pdf/10.1145/3669940.3707251). Lv Cunchi, Shi Xiao, Lei Zhengyu, Huang Jinyue, Tan Wenting, Zheng Xiaohui, Zhao Xiaofang. ASPLOS'25

[npu hybrid inference] [Fast On-device LLM Inference with NPUs](https://dl.acm.org/doi/pdf/10.1145/3669940.3707239). Xu Daliang, Zhang Hao, Yang Liming, Liu Ruiqi, Huang Gang, Xu Mengwei, Liu Xuanzhe. ASPLOS'25

[performance prediction] [Forecasting GPU Performance for Deep Learning Training and Inference](https://dl.acm.org/doi/pdf/10.1145/3669940.3707265). Lee Seonho, Phanishayee Amar, Mahajan Divya. ASPLOS'25

[embedding model training] [Efficient and Economic Embedding Model Training with Commodity GPUs](https://dl.acm.org/doi/pdf/10.1145/3669940.3707245). Xie Minhui, Zeng Shaoxun, Guo Hao, Gao Shiwei, Lu Youyou. ASPLOS'25

[moe training] [FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/pdf/10.1145/3669940.3707272). Pan Xinglin, Lin Wenxiang, Zhang Lin, Shi Shaohuai, Tang Zhenheng, Wang Rui, Li Bo, Chu Xiaowen. ASPLOS'25

[DNN Training, Graph Pipeline Parallelism] [Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism](https://dl.acm.org/doi/pdf/10.1145/3669940.3707220). Jeon Byungsoo, Wu Mengdi, Cao Shiyi, Kim Sunghyun, Park Sunghyun, Aggarwal Neeraj, Unger Colin, Arfeen Daiyaan, Liao Peiyuan, Miao Xupeng, Alizadeh Mohammad, Ganger Gregory R., Chen Tianqi, Jia Zhihao. ASPLOS'25

[DNN serving, Heterogeneous GPUs and Network] [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://dl.acm.org/doi/pdf/10.1145/3669940.3707215). Mei Yixuan, Zhuang Yonghao, Miao Xupeng, Yang Juncheng, Jia Zhihao, Vinayak Rashmi. ASPLOS'25

[Serverless DL Serving] [Medusa: Accelerating Serverless LLM Inference with Materialization](https://dl.acm.org/doi/pdf/10.1145/3669940.3707285). Zeng Shaoxun, Xie Minhui, Gao Shiwei, Chen Youmin, Lu Youyou. ASPLOS'25

[MoE Inference, on-device] [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://dl.acm.org/doi/pdf/10.1145/3669940.3707267). Cao Shiyi, Liu Shu, Griggs Tyler, Schafhalter Peter, Liu Xiaoxuan, Sheng Ying, Gonzalez Joseph E., Zaharia Matei, Stoica Ion. ASPLOS'25

[DNN Compression] [MVQ: Towards Efficient DNN Compression and Acceleration with Masked Vector Quantization](https://dl.acm.org/doi/pdf/10.1145/3669940.3707268). Li Shuaiting, Wang Chengxuan, Deng Juncan, Wang Zeyu, Ye Zewen, Wang Zongsheng, Shen Haibin, Huang Kejie. ASPLOS'25

[parallel checkpoint saving, DNN training] [PCcheck: Persistent Concurrent Checkpointing for ML](https://dl.acm.org/doi/pdf/10.1145/3669940.3707255). Strati Foteini, Friedman Michal, Klimovic Ana. ASPLOS'25

[Confidential computing LLM] [PipeLLM: Fast and Confidential Large Language Model Services with Speculative Pipelined Encryption](https://dl.acm.org/doi/pdf/10.1145/3669940.3707224). Tan Yifan, Tan Cheng, Mi Zeyu, Chen Haibo. ASPLOS'25

[kv-cache Management] [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://dl.acm.org/doi/pdf/10.1145/3669940.3707256). Prabhu Ramya, Nayak Ajay, Mohan Jayashree, Ramjee Ramachandran, Panwar Ashish. ASPLOS'25

[Energy Efficiency] [Using Analytical Performance/Power Model and Fine-Grained DVFS to Enhance AI Accelerator Energy Efficiency](https://dl.acm.org/doi/pdf/10.1145/3669940.3707231). Wang Zibo, Zhang Yijia, Wei Fuchun, Wang Bingqiang, Liu Yanlin, Hu Zhiheng, Zhang Jingyi, Xu Xiaoxin, He Jian, Wang Xiaoliang, Dou Wanchun, Chen Guihai, Tian Chen. ASPLOS'25

[multi-level KV cache, LLM Serving, Multi-turn Dialogues] [Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management](https://dl.acm.org/doi/pdf/10.1145/3676641.3716245). Jeong Jinwoo, Ahn Jeongseob. ASPLOS'25

[LLM serving, preemptive scheduling] [Aqua: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains](https://dl.acm.org/doi/pdf/10.1145/3676641.3715983). Vijaya Kumar Abhishek, Antichi Gianni, Singh Rachee. ASPLOS'25

[Collaboration-of-Experts, llm serving] [CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory](https://dl.acm.org/doi/pdf/10.1145/3676641.3715986). Suo Jiashun, Liao Xiaojian, Xiao Limin, Ruan Li, Wang Jinquan, Su Xiao, Huo Zhisheng. ASPLOS'25

[LLM Serving, Quantization] [COMET: Towards Practical W4A4KV4 LLMs Serving](https://dl.acm.org/doi/pdf/10.1145/3676641.3716252). Liu Lian, Cheng Long, Ren Haimeng, Xu Zhaohui, Pan Yudong, Wang Mengdi, Li Xiaowei, Han Yinhe, Wang Ying. ASPLOS'25

[Sparse Attention, long-sequence] [DynaX: Sparse Attention Acceleration with Dynamic X:M Fine-Grained Structured Pruning](https://dl.acm.org/doi/pdf/10.1145/3676641.3715991). Xiong Xiao, Chen Zhaorui, Liang Yue, Tian Minghao, Shang Jiaxing, Zhong Jiang, Liu Dajiang. ASPLOS'25

[LLM Training, sequence parallelism] [FlexSP: Accelerating Large Language Model Training via Flexible Sequence Parallelism](https://dl.acm.org/doi/pdf/10.1145/3676641.3715998). Wang Yujie, Wang Shiju, Zhu Shenhan, Fu Fangcheng, Liu Xinyi, Xiao Xuefeng, Li Huixia, Li Jiashi, Wu Faming, Cui Bin. ASPLOS'25

[MOE Inference, Multi-Batch Pipeline] [Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline](https://dl.acm.org/doi/pdf/10.1145/3676641.3716261). Fang Zhiyuan, Huang Yuegui, Hong Zicong, Lyu Yufeng, Chen Wuhui, Yu Yue, Yu Fan, Zheng Zibin. ASPLOS'25

[MoE training] [MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training](https://dl.acm.org/doi/pdf/10.1145/3676641.3716006). Cai Weilin, Qin Le, Huang Jiayi. ASPLOS'25

[LLM serving, PIM] [PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System](https://dl.acm.org/doi/pdf/10.1145/3676641.3716009). He Yintao, Mao Haiyu, Giannoula Christina, Sadrosadati Mohammad, Gómez-Luna Juan, Li Huawei, Li Xiaowei, Wang Ying, Mutlu Onur. ASPLOS'25

[DNN serving, deep learning recommendation model] [Load and MLP-Aware Thread Orchestration for Recommendation Systems Inference on CPUs](https://dl.acm.org/doi/pdf/10.1145/3676641.3716003). Jain Rishabh, Chou Teyuh, Kayiran Onur, Kalamatianos John, Loh Gabriel H., Kandemir Mahmut T., Das Chita R.. ASPLOS'25

[LLM serving, Operations optimization, AI compiler] [PICACHU: Plug-In CGRA Handling Upcoming Nonlinear Operations in LLMs](https://dl.acm.org/doi/pdf/10.1145/3676641.3716013). Qin Jiajun, Xia Tianhua, Tan Cheng, Zhang Jeff, Zhang Sai Qian. ASPLOS'25

[LLM inference, GPU kernel, hybrid batching] [POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference](https://dl.acm.org/doi/pdf/10.1145/3676641.3715996). Kamath Aditya K., Prabhu Ramya, Mohan Jayashree, Peter Simon, Ramjee Ramachandran, Panwar Ashish. ASPLOS'25

[LLM training, cloud-native system, data center] [Vela: A Virtualized LLM Training System with GPU Direct RoCE](https://dl.acm.org/doi/pdf/10.1145/3676641.3716280). Mohan Apoorve, Walkup Robert, Karacali Bengi, Chen Ming-hung, Kayi Abdullah, Schour Liran, Salaria Shweta, Wen Sophia, Chung I-hsin, Alim Abdul, Evangelinos Constantinos, Luo Lixiang, Dombrowa Marc, Schares Laurent, Sydney Ali, Maniotis Pavlos, Koteshwara Sandhya, Tang Brent, Belog Joel, Odaira Rei, Tarasov Vasily, Gampel Eran, Thorstensen Drew, Gershon Talia, Seelam Seetharami. ASPLOS'25

[LLM serving, end-to-end optimization] [Towards End-to-End Optimization of LLM-based Applications with Ayo](https://dl.acm.org/doi/pdf/10.1145/3676641.3716278). Tan Xin, Jiang Yimin, Yang Yitao, Xu Hong. ASPLOS'25

[LLM Inference, data center, energy efficient] [TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms](https://dl.acm.org/doi/pdf/10.1145/3676641.3716025). Stojkovic Jovan, Zhang Chaojie, Goiri Íñigo, Choukse Esha, Qiu Haoran, Fonseca Rodrigo, Torrellas Josep, Bianchini Ricardo. ASPLOS'25

[LLM inference, PIM with CXL] [PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference](https://dl.acm.org/doi/pdf/10.1145/3676641.3716267). Gu Yufeng, Khadem Alireza, Umesh Sumanth, Liang Ning, Servot Xavier, Mutlu Onur, Iyer Ravi, Das Reetuparna. ASPLOS'25

[LLM serving, Scheduler] [Past-Future Scheduler for LLM Serving under SLA Guarantees](https://dl.acm.org/doi/pdf/10.1145/3676641.3716011). Gong Ruihao, Bai Shihao, Wu Siyu, Fan Yunqian, Wang Zaijun, Li Xiuhong, Yang Hailong, Liu Xianglong. ASPLOS'25

[DNN training, multi-modal, multi-task] [Spindle: Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling](https://dl.acm.org/doi/pdf/10.1145/3676641.3715992). Wang Yujie, Zhu Shenhan, Fu Fangcheng, Miao Xupeng, Zhang Jie, Zhu Juan, Hong Fan, Li Yong, Cui Bin. ASPLOS'25

## NAACL-HLT

[language representation model] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf). Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. NAACL-HLT'19

##  ICML

[fine-tuning] [BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning](https://arxiv.org/pdf/1902.02671.pdf). Asa Cooper Stickland, Iain Murray. ICML'19

[fine-tuning] [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf). Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly. ICML'19

[pipeline-parallel] [Memory-Efficient Pipeline-Parallel DNN Training](http://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf). Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, Matei Zaharia. ICML'21

[scale language models with MoE] [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c/du22c.pdf). Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, Claire Cui. ICML'22

[MoE training and inference] [DeepSpeed-MoE Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf). Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi. ICML'22

[robustness to attack] [Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations](https://proceedings.mlr.press/v162/alam22a/alam22a.pdf). Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt. ICML'22

[transfer learning] [Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer](https://proceedings.mlr.press/v162/alegre22a/alegre22a.pdf). Lucas N. Alegre, Ana L. C. Bazzan, Bruno C. da Silva. ICML'22

[explainable ai] [XAI for Transformers: Better Explanations through Conservative Propagation](https://proceedings.mlr.press/v162/ali22a/ali22a.pdf). Ameen Ali, Thomas Schnake, Oliver Eberle, Gregoire Montavon, Klaus Robert Muller, Lior Wolf. ICML'22

[LLM serving system] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192.pdf). Yaniv Leviathan, Matan Kalman, Yossi Matias. ICML'23

[offload in inference] [FlexGen High-Throughput Generative Inference of Large Language Models with a Single GPU](https://openreview.net/pdf?id=RRntzKrBTp). Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher ReIon Stoica, Ce Zhang. ICML'23

[KV cache eviction policy] [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://openreview.net/pdf?id=ctPizehA9D). Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen. ICML'23

[pipeline parallelism] [BPIPE: Memory-Balanced Pipeline Parallelism for Training Large Language Models](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf). Taebum Kim, Hyoungjoo Kim,Gyeong-In Yu, Byung-Gon Chun. ICML'23

[privacy-preserving] [Are Diffusion Models Vulnerable to Membership Inference Attacks?](https://arxiv.org/pdf/2302.01316). Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, Kaidi Xu. ICML'23

[accelerate speculative decoding] [GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding](https://arxiv.org/pdf/2402.02082). Cunxiao Du, Jing Jiang, Xu Yuanchen, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, Yang You. ICML'24

[overlap all2all] [Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training](https://proceedings.mlr.press/v235/kim24w.html). Yechan Kim, Hwijoon Lim, and Dongsu Han. ICML'24

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

[memory-efficient training] [Accordion: Memory-Efficient DNN Training Using Adaptive Local Learning](https://arxiv.org/pdf/2402.14139). Dhananjay Saikumar，Blesson Varghese. EUROSYS'24

[multi-task training] [DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines](https://arxiv.org/pdf/2311.10418). Chenyu Jiang, Zhen Jia, Shuai Zheng, Yida Wang, Chuan Wu. EUROSYS'24

[auto parallelism] [Aceso: Efficient Parallel DNN Training through Iterative Bottleneck Alleviation](https://dl.acm.org/doi/pdf/10.1145/3627703.3629554). Guodong Liu, PictureYoushan Miao, PictureZhiqi Lin, PictureXiaoxiang Shi, PictureSaeed Maleki, PictureFan Yang, PictureYungang Bao, PictureSa Wang. EUROSYS'24

[moe training system] [ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling](https://dl.acm.org/doi/pdf/10.1145/3627703.3650083). Shaohuai Shi, Xinglin Pan, Qiang Wang, Chengjian Liu, Xiaozhe Ren, Zhongzhe Hu, Yu Yang, Bo Li, Xiaowen Chu. EUROSYS'24

[considering inter-arrival patterns] [Model Selection for Latency-Critical Inference Serving](https://dl.acm.org/doi/pdf/10.1145/3627703.3629565). Daniel Mendoza, Francisco Romero, and Caroline Trippel. EUROSYS'24

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

[scheduling DNN computational graphs] [Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators](https://www.usenix.org/system/files/osdi23-zhao.pdf). Jie Zhao, Siyuan Feng, Xiaoqiang Dan, Fei Liu, Chengke Wang, Sheng Yuan, Wenyuan Lv, and Qikai Xie. osdi'23

[inference latency] [ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](https://arxiv.org/pdf/2401.14351). Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai. osdi'24

[inference latency] [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/pdf/2401.09670.pdf). Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. osdi'24

[inference latency] [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/pdf/2403.02310). Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. osdi'24

[fair inference] [Fairness in Serving Large Language Models](https://arxiv.org/pdf/2401.00588). Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica. osdi'24

[schedule training workload on datacenter] [MAST: Global Scheduling of ML Training across Geo-Distributed Datacenters at Hyperscale](https://www.usenix.org/system/files/osdi24-choudhury.pdf). Arnab Choudhury, Meta Platforms; Yang Wang, Meta Platforms and The Ohio State University; Tuomas Pelkonen, Meta Platforms; Kutta Srinivasan, LinkedIn; Abha Jain, Shenghao Lin, Delia David, Siavash Soleimanifard, Michael Chen, Abhishek Yadav, Ritesh Tijoriwala, Denis Samoylov, and Chunqiang Tang, Meta Platforms. osdi'24

[runtime rescheduling across multiple model instances] [Llumnix: Dynamic Scheduling for Large Language Model Serving](https://arxiv.org/pdf/2406.03243). Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, Wei Lin. osdi'24

[considering correlation among multiple llm requests] [Parrot: Efficient Serving of LLM-based  Applications with Semantic Variable](https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf). Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, Chen Chen, Lili Qiu. osdi'24

[prefetch kv cache] [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/pdf/2406.19707). Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim. osdi'24

## SOSP

[pipeline parallelism] [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/pdf/10.1145/3341301.3359646). Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, Matei Zaharia. SOSP'19

[gradient compression] [Gradient Compression Supercharged High-Performance Data Parallel DNN Training](https://dl.acm.org/doi/10.1145/3477132.3483553). Youhui Bai, Cheng Li, Quan Zhou, Jun Yi, Ping Gong, Feng Yan, Ruichuan Chen, Yinlong Xu. SOSP'21

[sharing of KV cache] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf). Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. SOSP'23

[checkpoint] [GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints](https://dl.acm.org/doi/pdf/10.1145/3600006.3613145). Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, T. S. Eugene Ng, Yida Wang. SOSP'23

[dynamic parallelism strategy] [Enabling Parallelism Hot Switching for Efficient Training of Large Language Models](https://dl.acm.org/doi/pdf/10.1145/3694715.3695969). Hao Ge, Fangcheng Fu, Haoyang Li, Xuanyu Wang, Sheng Lin, Yujie Wang, Xiaonan Nie, Hailin Zhang, Xupeng Miao, Bin Cui. SOSP'24

## ICPP

[tensor parallelism] [Tesseract: Parallelize the Tensor Parallelism Efficiently](https://arxiv.org/pdf/2105.14500.pdf). Boxiang Wang, Qifan Xu, Zhengda Bian, Yang You. ICPP'22

[multiple inference tasks sharing single GPU] [SPLIT: QoS-Aware DNN Inference on Shared GPU via Evenly-Sized Model Splitting](https://dl.acm.org/doi/pdf/10.1145/3605573.3605627). Diaohan Luo, Tian Yu, Yuewen Wu, Heng Wu, Tao Wang, Wenbo Zhang. ICPP'23

[efficient all-reduce] [Wrht: Efficient All-reduce for Distributed DNN Training in Optical Interconnect Systems](https://dl.acm.org/doi/pdf/10.1145/3605573.3605624). Fei Dai, Yawen Chen, Zhiyi Huang, Haibo Zhang. ICPP'23

[cpu offload] [CoTrain: Efficient Scheduling for Large-Model Training upon GPU and CPU in Parallel](https://dl.acm.org/doi/pdf/10.1145/3605573.3605647). Zhenxing Li, Qiang Cao, Yajie Chen, Wenrui Yan. ICPP'23

[efficient communication in ddl] [OSP: Boosting Distributed Model Training with 2-stage Synchronization](https://dl.acm.org/doi/pdf/10.1145/3605573.3605650). Zixuan Chen, Lei Shi, Xuandong Liu, Jiahui Li, Sen Liu, Yang Xu. ICPP'23

[automatic parallelization] [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://dl.acm.org/doi/pdf/10.1145/3605573.3605613). Shenggui Li, Hongxin Liu, Zhengda Bian, Jiarui Fang, Haichen Huang, Yuliang Liu, Boxiang Wang, Yang You. ICPP'23

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

[kv cache eviction] [NACL: A General and Effective KV Cache Eviction Framework for LLM at Inference Time](https://aclanthology.org/2024.acl-long.428.pdf).Yilong Chen, Guoxia Wang, Junyuan Shang, Shiyao Cui, Zhenyu Zhang, Tingwen Liu, Shuohuan Wang, Yu Sun, Dianhai Yu, Hua Wu. ACL'24

[kv cache compression] [PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference](https://arxiv.org/pdf/2405.12532). Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, Hai Zhao. ACL'24

[trainable parallel context encoding] [Long-Context Language Modeling with Parallel Context Encoding](https://arxiv.org/pdf/2402.16617). Howard Yen, Tianyu Gao, Danqi Chen. ACL'24

## ICLR

[moe model] [Learning Factored Representations in a Deep Mixture of Experts](https://arxiv.org/abs/1312.4314). David Eigen, Marc'Aurelio Ranzato, Ilya Sutskever. ICLR'14

[moe model] [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://openreview.net/pdf?id=B1ckMDqlg). Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. ICLR'17

[large mini-batches] [LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76 MINUTES](https://openreview.net/pdf?id=Syx4wnEtvH). Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh. ICLR'20

[scaling giant models] [Gshard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://openreview.net/pdf?id=qrwe7XHTmYb). Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen. ICLR'21

[transformer for image recognition] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy). Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. ICLR'21

[expert-based model] [Taming Sparsely Activated Transformer with Stochastic Experts](https://openreview.net/pdf?id=B72HXs80q4). Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, Jianfeng Gao. ICLR'22

[large language model] [GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL](https://openreview.net/pdf?id=-Aw0rrrPUF). Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, Jie Tang. ICLR'23

[transformer block] [Brainformers Trading Simplicity for Efficiency](https://openreview.net/pdf?id=w5q6tHO1dl1). Yanqi Zhou, Nan Du, Yanping Huang, Daiyi Peng, Chang Lan, Da Huang, Siamak Shakeri, David So, Andrew Dai, Yifeng Lu, Zhifeng Chen, Quoc Le, Claire Cui, James Laundon, Jeff Dean. ICLR'23 

[rlhf] [SAFE RLHF: SAFE REINFORCEMENT LEARNING FROM HUMAN FEEDBACK](https://openreview.net/pdf?id=TyFrPOKYXw). Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang. ICLR'24

[llm for control] [DILU: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models](https://arxiv.org/pdf/2309.16292). Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yu Qiao. ICLR'24

[infinite sequence length inference] [EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS](https://arxiv.org/pdf/2309.17453). Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. ICLR'24

[agent] [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/pdf/2406.04692). Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, James Zou. ICLR'25

[agent] [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/pdf/2410.10762). Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, Chenglin Wu. ICLR'25

[agent] [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/pdf/2410.07095). Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, Aleksander Mądry. ICLR'25

## VLDB

[data parallelism] [PyTorch distributed: experiences on accelerating data parallel training](https://arxiv.org/pdf/2006.15704.pdf). Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, Soumith Chintala. VLDB'20

[automatic parallelism] [Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism](https://arxiv.org/pdf/2211.13878.pdf). Xupeng Miao, Yujie Wang, Youhe Jiang, Chunan Shi, Xiaonan Nie, Hailin Zhang, Bin Cui. VLDB'22

## NSDI

[workload analysis and scheduling] [MLaaS: in the Wild Workload Analysis and Scheduling in Large-Scale Heterogeneous](https://www.usenix.org/system/files/nsdi22-paper-weng.pdf). Qizhen Weng, Hong Kong University of Science and Technology and Alibaba Group; Wencong Xiao, Alibaba Group; Yinghao Yu, Alibaba Group and Hong Kong University of Science and Technology; Wei Wang, Hong Kong University of Science and Technology; Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin, and Yu Ding, Alibaba Group. nsdi'22

[checkpoint] [Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models](https://www.usenix.org/system/files/nsdi22-paper-eisenman.pdf). Assaf Eisenman, Kiran Kumar Matam, Steven Ingram, Dheevatsa Mudigere, Raghuraman Krishnamoorthi, Krishnakumar Nair, and Misha Smelyanskiy, Facebook; Murali Annavaram, Facebook and USC. nsdi'22

[checkpoint] [Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs](https://www.usenix.org/system/files/nsdi23-thorpe.pdf). John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, and Yifan Qiao, UCLA; Zhihao Jia, CMU; Minjia Zhang, Microsoft Research; Ravi Netravali, Princeton University; Guoqing Harry Xu, UCLA. nsdi'23

[cache among request] [Approximate Caching for Efficiently Serving Diffusion Models](https://arxiv.org/pdf/2312.04429). Shubham Agarwal, Subrata Mitra, Sarthak Chakraborty, Srikrishna Karanam, Koyel Mukherjee, Shiv Saini. nsdi'24

## AAAI

[model framework] [Go Wider Instead of Deeper](https://cdn.aaai.org/ojs/20858/20858-13-24871-1-2-20220628.pdf). Fuzhao Xue, Ziji Shi, Futao Wei, Yuxuan Lou, Yong Liu, Yang You. AAAI'22

[computation reuse in diffusion] [Accelerating Text-to-Image Editing via Cache-Enabled Sparse Diffusion Inference](https://arxiv.org/pdf/2305.17423). Zihao Yu, Haoyang Li, Fangcheng Fu, Xupeng Miao, Bin Cui. AAAI'24

## ICCV

[vision transformer] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf). Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. ICCV'21

## FAST

[offloading data to SSD] [FlashNeuron: SSD-Enabled Large-Batch Training of Very Deep Neural Networks](https://www.usenix.org/system/files/fast21-bae.pdf). Jonghyun Bae, Seoul National University; Jongsung Lee, Seoul National University and Samsung Electronics; Yunho Jin and Sam Son, Seoul National University; Shine Kim, Seoul National University and Samsung Electronics; Hakbeom Jang, Samsung Electronics; Tae Jun Ham and Jae W. Lee, Seoul National University. FAST'21

[checkpoint] [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://www.usenix.org/system/files/fast21-mohan.pdf). Jayashree Mohan, UT Austin; Amar Phanishayee, Microsoft Research; Vijay Chidambaram, UT Austin and VMware research. FAST'21 

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

[flash caching compression] [CDCache: Space-Efficient Flash Caching via Compression-before-Deduplication])(https://ieeexplore.ieee.org/abstract/document/10621089). Hengying Xiao, Jingwei Li, Yanjing Ren, Ruijin Wang, and Xiaosong Zhang. INFOCOM'24

[Edge-Cloud system request scheduling] [Cur-CoEdge: Curiosity-Driven Collaborative Request Scheduling in Edge-Cloud Systems](https://ieeexplore.ieee.org/abstract/document/10621190). Yunfeng Zhao, Chao Qiu, Xiaoyun Shi, Xiaofei Wang, Dusit Niyato, Victor C. M. Leung. INFOCOM'24

[Collaborative Edge Computing] [Exploiting Storage for Computing: Computation Reuse in Collaborative Edge Computing](https://ieeexplore.ieee.org/abstract/document/10621100). Xingqiu He, Chaoqun You, Tony Q. S. Quek. INFOCOM'24

[edge collaborative inference] [Galaxy: A Resource-Efficient Collaborative Edge AI System for In-situ Transformer Inference](https://ieeexplore.ieee.org/abstract/document/10621342). Shengyuan Ye, Jiangsu Du, Liekang Zeng, Wenzhong Ou, Xiaowen Chu, Yutong Lu, Xu Chen. INFOCOM'24

[CPU-only Mutil-DNN inference] [Minimizing Latency for Multi-DNN Inference on Resource-Limited CPU-Only Edge Devices](https://ieeexplore.ieee.org/abstract/document/10621120). Tao Wang, Tuo Shi, Xiulong Liu, Jianping Wang, Bin Liu, Yingshu Li, Yechao She. INFOCOM'24

[lightweight token management] [OTAS: An Elastic Transformer Serving System via Token Adaptation](https://ieeexplore.ieee.org/abstract/document/10621087). Jinyu Chen, Wenchao Xu, Zicong Hong, Song Guo, Haozhao Wang, Jie Zhang, and Deze Zeng. INFOCOM'24

[MoE training] [Parm: Efficient Training of Large Sparsely-Activated Models with Dedicated Schedules](https://ieeexplore.ieee.org/abstract/document/10621327). Xinglin Pan, Wenxiang Lin, Shaohuai Shi, Xiaowen Chu, Weinong Sun, Bo Li. INFOCOM'24

[End-Cloud Collaborative Inference] [Accelerating End-Cloud Collaborative Inference via Near Bubble-free Pipeline Optimization](https://arxiv.org/pdf/2501.12388). Luyao Gao, Jianchun Liu, Hongli Xu, Sun Xu, Qianpiao Ma, Liusheng Huang. INFOCOM'25

[MoE Inference in Serverless Computing] [Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing](https://arxiv.org/pdf/2501.05313). Mengfan Liu, Wei Wang, and Chuan Wu. INFOCOM'25

[Cost-Efficient Large Model Training] [Espresso: Cost-Efficient Large Model Training by Exploiting GPU Heterogeneity in the Cloud](https://fangmingliu.github.io/files/infocom25-train.pdf). Qiannan Zhou, Fei Xu†∗, Lingxuan Weng, Ruixing Li, Xudong Wu, Li Chen, Zhi Zhou, Fangming Liu. INFOCOM'25

[KV Cache] [Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management](https://arxiv.org/pdf/2501.06709). Qianli Liu, Zicong Hong1, Peng Li, Fahao Chen and Song Guo. INFOCOM'25

[speculative decoding] [SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models](https://arxiv.org/pdf/2503.15921). Fahao Chen, Peng Li, Tom H. Luan, Zhou Su, and Jing Deng. INFOCOM'25

[Offload Training] [MemFerry: A Fast and Memory Efficient Offload Training Framework with Hybrid GPU Computation](http://iir.ruc.edu.cn/~litong/papers/MemFerry_A%20Fast%20and%20Memory%20Efficient%20Offload%20Training%20Framework%20with%20Hybrid%20GPU%20Computation.pdf). Zhiyi Yao, Zuning Liang, Yuedong Xu, Jin Zhao, Jessie Hui Wang, Tong Li. INFOCOM'25

[Resource-Efficient Collaborative Inference] [Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices](https://arxiv.org/pdf/2504.08242). Shengyuan Ye, Bei Ouyang, Liekang Zeng, Tianyi Qian, Xiaowen Chu, Jian Tang, Xu Chen. INFOCOM'25

## MLSys

[memory reuse] [SAFE OPTIMIZED STATIC MEMORY ALLOCATION FOR PARALLEL DEEP LEARNING](https://proceedings.mlsys.org/paper_files/paper/2023/file/0c8abcf158ed12d0dd94480681186fda-Paper-mlsys2023.pdf). Ioannis Lamprou, Zhen Zhang, Javier de Juan, Hang Yang, Yongqiang Lai, Etienne Filhol, Cedric Bastoul. MLSys'23

[moe kernel] [MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS](https://proceedings.mlsys.org/paper_files/paper/2023/file/f9f4f0db4894f77240a95bde9df818e0-Paper-mlsys2023.pdf). Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. MLSys'23

[gpu-cpu memory swap] [μ-TWO: 3× Faster Multi-Model Training with Orchestration and Memory Optimization](https://proceedings.mlsys.org/paper_files/paper/2023/file/a72071d84c001596e97a2c7e1e880559-Paper-mlsys2023.pdf). Sanket Purandare, Abdul Wasay, Stratos Idreos, NameError. MLSys'23

[communication system] [On Optimizing the Communication of Model Parallelism](https://arxiv.org/pdf/2211.05322). Yonghao Zhuang, Hexu Zhao, Lianmin Zheng, Zhuohan Li, Eric P. Xing, Qirong Ho, Joseph E. Gonzalez, Ion Stoica, Hao Zhang. MLSys'23

[parameter partitioning] [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102). Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean. MLSys'23

[operator fusion] [Transcending runtime-memory tradeoffs in checkpointing by being fusion aware](https://proceedings.mlsys.org/paper_files/paper/2023/file/8a27bb69950c0b46cdb36d10e5514cc8-Paper-mlsys2023.pdf). Horace He, Shangdi Yu. MLSys'23

[pipeline parallelism] [Breadth-First Pipeline Parallelism](https://arxiv.org/pdf/2211.05953). Joel Lamy-Poirier. MLSys'23

[adaptive parallelism and pipelining] [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/pdf/2206.03382). Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, Yongqiang Xiong. MLSys'23

[parallel computing using CPUs and GPUs] [HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices](https://arxiv.org/pdf/2403.01164). Xuanlei Zhao, Bin Jia, Haotian Zhou, Ziming Liu, Shenggan Cheng, Yang You. MLSys'24

[accelerate all2all] [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/pdf/2502.19811). Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, Quan Chen, Xin Liu. MLSys'25

[sparse gradient for faster training] [Radius: Range-based Gradient Sparsity for Large Foundation Model Pre-training](https://mlsys.org/virtual/2025/poster/3256). Mingkai Zheng, Zhao Zhang. MLSys'25

[a simulation engine that improves the efficiency of LLM agent simulations] [AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution](https://arxiv.org/pdf/2411.03519). Zhiqiang Xie, Hao Kang, Ying Sheng, Tushar Krishna, Kayvon Fatahalian, Christos Kozyrakis. MLSys'25

[sparse metrix operations for training] [SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations](https://arxiv.org/pdf/2502.16949). Md Saidul Hoque Anik, Ariful Azad. MLSys'25

[heterogeneous attention compute on edge] [MAS-ATTENTION: MEMORY-AWARE STREAM PROCESSING FOR ATTENTION ACCELERATION ON RESOURCE-CONSTRAINED EDGE DEVICES](https://arxiv.org/pdf/2411.17720). Mohammadali Shakerdargah, Shan Lu, Chao Gao, Di Niu. MLSys'25

[new variants of attention compute] [SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention](https://arxiv.org/pdf/2406.15486). Qianchao Zhu, Jiangfei Duan, Chang Chen, Siran Liu,  Xiuhong Li, Guanyu Feng, Xin Lv, Xiao Chuanfu, Dahua Lin, Chao Yang. MLSys'25

[context parallelism for long-context LLM inference] [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/pdf/2411.01783). Amy Yang, Jingyi Yang, Aya Ibrahim, Xinfeng Xie, Bangsheng Tang, Grigory Sizov, Jongsoo Park, Jianyu Huang. MLSys'25

[fills pipeline bubbles with execution of other pending jobs] [PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training](https://arxiv.org/pdf/2410.07192). Daiyaan Arfeen · Zhen Zhang · Xinwei Fu · Gregory R. Ganger · Yida Wang. MLSys'25

[speed up LLM-application by reuse kv] [Optimizing LLM Queries in Relational Data Analytics Workloads](https://arxiv.org/pdf/2403.05821). Shu Liu · Asim Biswal · Audrey Cheng · Amog Kamsetty · Luis Gaspar Schroeder · Liana Patel · Shiyi Cao · Xiangxi Mo · Ion Stoica · Joseph. MLSys'25

[compress activation in LoRA training] [HyC-LoRA: Memory Efficient LoRA Fine-tuning with Hybrid Activation Compression](https://mlsys.org/virtual/2025/poster/3254). Yujin Wang · Shunan Dong · Zongle Huang · Yichen You · Liu He · Huazhong Yang · Yongpan Liu · Hongyang Jia. MLSys'25

[evaluate kv compression tech] [Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving](https://arxiv.org/pdf/2503.24000). Wei Gao · Xinyu Zhou · Peng Sun · Tianwei Zhang · Yonggang Wen. MLSys'25

[new sparse attention for faster prefill] [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](https://arxiv.org/pdf/2502.14866). Shang Yang · Junxian Guo · Haotian Tang · Qinghao Hu · Guangxuan Xiao · Jiaming Tang · Yujun Lin · Zhijian Liu · Yao Lu · Song Han. MLSys'25

[attention kernel for tree-structued kv reuse] [FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference](https://mlsys.org/virtual/2025/poster/3278). Zaifeng Pan · Yitong Ding · Yue Guan · Zheng Wang · Zhongkai Yu · Xulong Tang · Yida Wang · Yufei Ding. MLSys'25

[offload kv storage+compute to cpu] [FlexInfer: Flexible LLM Inference with CPU Computations](https://arxiv.org/html/2503.03777). Seonjin Na · Geonhwa Jeong · Byung Hoon Ahn · Aaron Jezghani · Jeffrey Young · Christopher Hughes · Tushar Krishna · Hyesoon Kim. MLSys'25

[using different parallelism in PD] [Seesaw: High-throughput LLM Inference via Model Re-sharding](https://arxiv.org/pdf/2503.06433). Qidong Su · Wei Zhao · Xin Li · Muralidhar Andoorveedu · Chenhao Jiang · Zhanda Zhu · Kevin Song · Christina Giannoula · Gennady Pekhimenko. MLSys'25

[consider vocabulary layers(embd+lmhead) in PP] [Balancing Pipeline Parallelism with Vocabulary Parallelism](https://arxiv.org/pdf/2411.05288). Man Tsung Yeung · Penghui Qi · Min Lin · Xinyi Wan. MLSys'25

[state schedule for SLO optimization] [SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling](https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2Fce55a1f2-ff0d-45f6-8985-f4f251b2a0d4.pdf). Ke Hong · Xiuhong Li · Lufang Chen · Qiuli Mao · Guohao Dai · Xuefei Ning · Shengen Yan · Yun Liang · Yu Wang. MLSys'25

[serving in heterogeneous cloud environments] [ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments](https://arxiv.org/pdf/2502.09334). YOUHE JIANG · Fangcheng Fu · Xiaozhe Yao · Taiyi Wang · Bin CUI · Ana Klimovic · Eiko Yoneki. MLSys'25

[comp. comm. overlap in MoE] [COMET: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/pdf/2502.19811). Shulai Zhang · Ningxin Zheng · Haibin Lin · Ziheng Jiang · Wenlei Bao · Chengquan Jiang · Qi Hou · Weihao Cui · Size Zheng · Li-Wen Chang · Quan Chen · Xin Liu. MLSys'25

[offload kv storage+compute to cpu] [NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference](https://arxiv.org/pdf/2411.01142). Xuanlin Jiang · Yang Zhou · Shiyi Cao · Ion Stoica · Minlan Yu. MLSys'25

[long context training method] [Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer](https://arxiv.org/pdf/2408.16978). Jinghan Yao · Sam Jacobs · Masahiro Tanaka · Olatunji Ruwase · Hari Subramoni · Dhabaleswar Panda. MLSys'25

[new variants of attention compute] [LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers](https://arxiv.org/pdf/2405.10480). Rya Sanovar · Srikant Bharadwaj · Renée St. Amant · Victor Ruehle · Saravan Rajmohan. MLSys'25

[new type of training] [APOLLO: SGD-like Memory, AdamW-level Performance](https://arxiv.org/pdf/2412.05270). Hanqing Zhu · Zhenyu Zhang · Wenyan Cong · Xi Liu · Sem Park · Vikas Chandra · Bo Long · David Pan · Atlas Wang · Jinwon Lee. MLSys'25

[optimize off-chip memory access communication on Edge] [MEADOW: Memory-efficient Dataflow and Data Packing for Low Power Edge LLMs](https://arxiv.org/pdf/2503.11663). Hanqing Zhu · Zhenyu Zhang · Wenyan Cong · Xi Liu · Sem Park · Vikas Chandra · Bo Long · David Pan · Atlas Wang · Jinwon Lee. MLSys'25

[improves hardware utilization of pp] [Scaling Deep Learning Training with MPMD Pipeline Parallelism](https://arxiv.org/pdf/2412.14374). Anxhelo Xhebraj · Sean Lee · Hanfeng Chen · Vinod Grover. MLSys'25


[on device training with fwd] [Efficient On-Device Machine Learning with a Biologically-Plausible Forward-Only Algorithm](https://mlsys.org/virtual/2025/poster/3264). Baichuan Huang · Amir Aminifar. MLSys'25

[sparse inference] [Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking](https://arxiv.org/pdf/2412.01380). Marco Federici · Davide Belli · Mart van Baalen · Amir Jalalirad · Andrii Skliar · Bence Major · Markus Nagel · Paul Whatmough. MLSys'25

[quantization + system design] [QServe:W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/pdf/2405.04532). Yujun Lin · Haotian Tang · Shang Yang · Zhekai Zhang · Guangxuan Xiao · Chuang Gan · Song Han. MLSys'25

[a compiler auto implement attention variants kernel] [FlexAttention: A Programming Model for Generating Fused Attention Variants.](https://arxiv.org/pdf/2412.05496). Juechu Dong · BOYUAN FENG · Driss Guessous · Yanbo Liang · Horace He. MLSys'25

[efficient prefix kv storage] [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/pdf/2411.19379). Rui Pan · Zhuang Wang · Zhen Jia · Can Karakus · Luca Zancato · Tri Dao · Yida Wang · Ravi Netravali. MLSys'25

[efficient attention kernel] [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/pdf/2501.01005). Zihao Ye · Lequn Chen · Ruihang Lai · Wuwei Lin · Yineng Zhang · Stephanie Wang · Tianqi Chen · Baris Kasikci · Vinod Grover · Arvind Krishnamurthy · Luis Ceze. MLSys'25

[quantized MoEs with a mixture of low-rank compensators] [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://arxiv.org/pdf/2504.02658). Beichen Huang · Yueming Yuan · ZELEI SHAO · Minjia Zhang. MLSys'25

[compilation and generation of overlapped compute-communication kernels] [TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives](https://arxiv.org/pdf/2503.20313). Beichen Huang · Yueming Yuan · ZELEI SHAO · Minjia Zhang. MLSys'25

[speed up sparse DNNs] [Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators](https://mlsys.org/virtual/2025/poster/3262). Geonhwa Jeong · Po-An Tsai · Abhimanyu Rajeshkumar Bambhaniya · Stephen Keckler · Tushar Krishna. MLSys'25

[new variants of attention compute] [TurboAttention: Efficient attention approximation for high throughputs llm](https://arxiv.org/pdf/2412.08585). Hao Kang · Srikant Bharadwaj · James Hensman · Tushar Krishna · Victor Ruehle · Saravan Rajmohan. MLSys'25

[agent grammar] [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/pdf/2411.15100). Yixin Dong · Charlie Ruan · Yaxing Cai · Ziyi Xu · Yilong Zhao · Ruihang Lai · Tianqi Chen. MLSys'25

## SIGCOMM

[moe all2all] [Janus: A Unified Distributed Training Framework for Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/pdf/10.1145/3603269.3604869). Juncai Liu, Jessie Hui Wang, Yimin Jiang. SIGCOMM'23

## CVPR

[diffusion on mobile] [Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations](https://arxiv.org/pdf/2304.11267). Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, Matthias Grundmann. CVPR'23

[computation reuse in diffusion] [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/pdf/2312.00858). Xinyin Ma, Gongfan Fang, Xinchao Wang. CVPR'24

[cloud-device collaboration for mllm] [Cloud-Device Collaborative Learning for Multimodal Large Language Models](https://arxiv.org/pdf/2312.16279v1). Guanqun Wang, Jiaming Liu, Chenxuan Li, Junpeng Ma, Yuan Zhang, Xinyu Wei, Kevin Zhang, Maurice Chong, Ray Zhang, Yijiang Liu, Shanghang Zhang. CVPR'24

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

## TMC

[speculative decoding on edge] [EdgeLLM: Fast On-device LLM Inference with Speculative Decoding](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10812936). Daliang Xu; Wangsong Yin; Hao Zhang; Xin Jin; Ying Zhang; Shiyun Wei. TMC'24

## arxiv

[inference on edge device] [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/pdf/1908.09791). Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han. 19

[privacy-preserving] [DataMix: Efficient Privacy-Preserving Edge-Cloud Inference](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560562.pdf). Zhijian Liu, Zhanghao Wu, Chuang Gan, Ligeng Zhu, and Song Han. 20

[checkpoint] [On Efficient Constructions of Checkpoints](https://arxiv.org/pdf/2009.13003.pdf). Yu Chen, Zhenming Liu, Bin Ren, Xin Jin. 20

[checkpoint] [A Study of Checkpointing in Large Scale Training of Deep Neural Networks](https://arxiv.org/pdf/2012.00825.pdf). Elvis Rojas, Albert Njoroge Kahira, Esteban Meneses, Leonardo Bautista Gomez, Rosa M Badia. 20

[checkpoint] [ECRM: Efficient Fault Tolerance for Recommendation Model Training via Erasure Coding](https://arxiv.org/pdf/2104.01981.pdf). Kaige Liu, Jack Kosaian, K. V. Rashmi. 21

[moe model] [BASE Layers: Simplifying Training of Large, Sparse Models](https://arxiv.org/pdf/2103.16716.pdf). Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, Luke Zettlemoyer. 21

[moe model] [Learning Large-scale Universal User Representation with Sparse Mixture of Experts](https://arxiv.org/pdf/2207.04648.pdf). Caigao Jiang, Siqiao Xue, James Zhang, Lingyue Liu, Zhibo Zhu, Hongyan Hao. 22

[moe training and inference] [SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System](https://arxiv.org/pdf/2205.10034.pdf). Liang Shen, Zhihua Wu, WeiBao Gong, Hongxiang Hao, Yangfan Bai, HuaChao Wu, Xinxuan Wu, Jiang Bian, Haoyi Xiong, Dianhai Yu, Yanjun Ma. 22

[diffusion transformer] [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748). William Peebles, Saining Xie. 22

[vision-language model pretrain] [BLIP:Bootstrapping Language-Image Pretraining](https://arxiv.org/pdf/2201.12086). Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi. 22

[long context] [FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf). Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. 22

[moe] [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/pdf/2202.09368). Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, James Laudon. 22

[privacy-preserving] [Model Protection: Real-Time Privacy-Preserving
Inference Service for Model Privacy at the Edge](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9609559). Jiahui Hou , Huiqi Liu , Yunxin Liu, Yu Wang, Peng-Jun Wan, and Xiang-Yang Li. 22

[autoregressive diffusion] [Diffusion Probabilistic Modeling for Video Generation](https://arxiv.org/pdf/2203.09481). Ruihan Yang, Prakhar Srivastava, Stephan Mandt. 22

[kv cache management] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf). Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. 23

[automatic KV cache reuse] [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/pdf/2312.07104.pdf). Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng. 23

[moe] [Mixture of Experts with Uncertainty Voting for Imbalanced Deep Regression Problems](https://arxiv.org/pdf/2305.15178). Yuchang Jiang, Vivien Sainte Fare Garnot, Konrad Schindler, Jan Dirk Wegner. 23

[batching LoRA inference] [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/pdf/2311.03285.pdf). Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica. 23

[reducing GPU memory] [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/pdf/2312.12456.pdf). Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen. 23

[reducing communication volume] [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/pdf/2306.10209.pdf). Guanhua Wang, Heyang Qin, Sam Ade Jacobs, Connor Holmes, Samyam Rajbhandari, Olatunji Ruwase, Feng Yan, Lei Yang, Yuxiong He. 23

[efficient generative llm inference] [Splitwise: Efficient generative LLM inference using phase splitting](https://arxiv.org/pdf/2311.18677.pdf). Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, Ricardo Bianchini. 23

[inference in flash] [LLM in a flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/pdf/2312.11514). Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar. 23

[efficient generative llm inference] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/pdf/2308.16369.pdf). Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee. 23

[efficient generative llm inference] [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference](https://arxiv.org/pdf/2307.02628.pdf). Luciano Del Corro, Allie Del Giorno, Sahaj Agarwal, Bin Yu, Ahmed Awadallah, Subhabrata Mukherjee. 23

[efficient generative llm inference] [Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/pdf/2401.11181.pdf). Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan. 23

[model pruning] [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/pdf/2309.10285.pdf). Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, Shuaiwen Leon Song. 23

[kv cache] [Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs](https://arxiv.org/pdf/2310.01801). Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, Jianfeng Gao. 23

[model pruning] [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/pdf/2305.11627.pdf). Xinyin Ma, Gongfan Fang, Xinchao Wang. 23

[model pruning] [ZipLM: Inference-Aware Structured Pruning of Language Models](https://arxiv.org/pdf/2302.04089.pdf). Eldar Kurtic, Elias Frantar, Dan Alistarh. 23

[model pruning] [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/pdf/2402.05406.pdf). Lucio Dery, Steven Kolawole, Jean-François Kagy, Virginia Smith, Graham Neubig, Ameet Talwalkar. 23

[contextual sparsity] [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/pdf/2310.17157.pdf). Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, Beidi Chen. 23

[checkpoint] [SWIFT: Expedited Failure Recovery for Large-scale DNN Training](https://arxiv.org/pdf/2302.06173.pdf). Yuchen Zhong, Guangming Sheng, Juncheng Liu, Jinhui Yuan, Chuan Wu. 23

[checkpoint] [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates](https://arxiv.org/pdf/2309.08125.pdf). Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, Mosharaf Chowdhury. 23

[expert placement] [FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement](https://arxiv.org/pdf/2304.03946.pdf). Xiaonan Nie, Xupeng Miao, Zilong Wang, Zichao Yang, Jilong Xue, Lingxiao Ma, Gang Cao, Bin Cui. 23

[moe model] [LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment](https://arxiv.org/pdf/2312.09979.pdf). Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang. 23

[moe model] [Mixture of Experts with Uncertainty Voting for Imbalanced Deep Regression Problems](https://arxiv.org/pdf/2305.15178.pdf). Yuchang Jiang, Vivien Sainte Fare Garnot, Konrad Schindler, Jan Dirk Wegner. 23

[computation reuse in diffusion] [Cache Me if You Can: Accelerating Diffusion Models through Block Caching](https://arxiv.org/pdf/2312.03209). Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, Christian Rupprecht, Daniel Cremers, Peter Vajda, Jialiang Wang. 23

[moe fine-tuning] [MOELoRA: An MOE-based Parameter Efficient Fine-Tuning Method for Multi-task Medical Applications](https://arxiv.org/pdf/2310.18339.pdf). Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, Yefeng Zheng. 23

[moe inference] [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference](https://arxiv.org/pdf/2303.06182.pdf). Haiyang Huang, Newsha Ardalani, Anna Sun, Liu Ke, Hsien-Hsin S. Lee, Anjali Sridhar, Shruti Bhosale, Carole-Jean Wu, Benjamin Lee. 23

[multi-modality] [ImageBind-LLM: Multi-modality Instruction Tuning](https://arxiv.org/pdf/2309.03905.pdf). Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, Xudong Lu, Shuai Ren, Yafei Wen, Xiaoxin Chen, Xiangyu Yue, Hongsheng Li, Yu Qiao. 23

[moe] [Merge, Then Compress: Demystify Efficient SMoE with Hints from Its Routing Policy](https://arxiv.org/pdf/2310.01334). Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, Tianlong Chen. 23

[moe on edge] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/pdf/2308.14352). Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu. 23

[diffusion transformer] [Photorealistic Video Generation with Diffusion Models](https://arxiv.org/pdf/2312.06662). Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama. 23

[long context] [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/pdf/2305.19370). Hao Liu, Pieter Abbeel. 23

[inference serving system] [SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads](https://arxiv.org/pdf/2312.16733). Alind Khare, Dhruv Garg, Sukrit Kalra, Snigdha Grandhi, Ion Stoica, Alexey Tumanov. 23

[kv cache] [Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time](https://arxiv.org/pdf/2305.17118). Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava. 23

[inference serving system] [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/pdf/2305.05920). Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, Xin Jin. 23                                              

[sparsity] [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/pdf/2306.11222). Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang, Pengcheng He, Weizhu Chen, Tuo Zhao. 23     

[consistent style in image generation] [Style Aligned Image Generation via Shared Attention](https://arxiv.org/pdf/2312.02133). Amir Hertz, Andrey Voynov, Shlomi Fruchter, Daniel Cohen-Or. 23

[agent simulation framework] [SurrealDriver: Designing Generative Driver Agent Simulation Framework in Urban Contexts based on Large Language Model](https://arxiv.org/pdf/2309.13193). Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, Jiangtao Gong. 23

[llm for control] [LANGUAGEMPC: LARGE LANGUAGE MODELS AS DECISION MAKERS FOR AUTONOMOUS DRIVING](https://arxiv.org/pdf/2310.03026). Hao Sha, Yao Mu, Yuxuan Jiang, Li Chen, Chenfeng Xu, Ping Luo, Shengbo Eben Li, Masayoshi Tomizuka, Wei Zhan, Mingyu Ding. 23

[reuse kv cache] [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/pdf/2311.04934). In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, Lin Zhong. 23

[pre-gate] [Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference](https://arxiv.org/pdf/2308.12066). Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang. 23

[moe on device] [EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models](https://arxiv.org/pdf/2308.14352). Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu. 23

[inference using cpu and gpu] [FASTDECODE: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines](https://arxiv.org/pdf/2403.11421). Jiaao He, Jidong Zhai. 24

[llm within npu] [Fast On-device LLM Inference with NPUs](https://arxiv.org/pdf/2407.05858). Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu. 24

[efficient llm switching] [ELMS: Elasticized Large Language Models OnMobile Devices](https://arxiv.org/pdf/2409.09071). Wangsong Yin, Rongjie Yi, Daliang Xu, Gang Huang, Mengwei Xu, Xuanzhe Liu. 24

[sparse attention] [CHAI: Clustered Head Attention for Efficient LLM Inference](https://arxiv.org/pdf/2403.08058). Saurabh Agarwal, Bilge Acun, Basil Homer, Mostafa Elhoushi, Yejin Lee, Shivaram Venkataraman, Dimitris Papailiopoulos, Carole-Jean. 24  

[kv cache] [DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving](https://arxiv.org/pdf/2403.01876). Foteini Strati, Sara Mcallister, Amar Phanishayee, Jakub Tarnawski, Ana Klimovic. 24       

[moe] [OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models](https://arxiv.org/pdf/2402.01739). Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, Yang You. 24                                                                                                                                                                                                                                                                                                                                                                                                                 

[speculative decoding] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774.pdf). Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. 24

[speculative decoding] [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/pdf/2305.09781.pdf). Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia. 24

[small language model] [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/pdf/2401.02385). Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. 24

[long context, energy-efficient] [Squid: Long Context as a New Modality for Energy-Efficient On-Device Language Models](https://arxiv.org/pdf/2408.15518). Wei Chen, Zhiyuan Li, Shuo Xin, Yihao Wang. 24

[moe model] [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf). Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.  24 

[llm on mobile device] [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/pdf/2402.14905). Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra. 24

[small language model] [MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT](https://arxiv.org/pdf/2402.16840). Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M. Anwer, Michael Felsberg, Tim Baldwin, Eric P. Xing, Fahad Shahbaz Khan. 24

[distributed kv cache] [Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache](https://arxiv.org/pdf/2401.02669.pdf). Bin Lin, Tao Peng, Chen Zhang, Minmin Sun, Lanbo Li, Hanyu Zhao, Wencong Xiao, Qi Xu, Xiafei Qiu, Shen Li, Zhigang Ji, Yong Li, Wei Lin. 24

[fast moe inference with cpu] [FIDDLER: CPU-GPU ORCHESTRATION FOR FAST INFERENCE OF MIXTURE-OF-EXPERTS MODELS](https://arxiv.org/pdf/2402.07033). Keisuke Kamahori, Yile Gu, Kan Zhu, Baris Kasikci.24

[computation reuse in diffusion] [DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models](https://arxiv.org/pdf/2402.19481). Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li, Song Han. 24

[computation reuse in diffusion] [Fast Inference Through The Reuse Of Attention Maps In Diffusion Models](https://arxiv.org/pdf/2402.19481). Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li, Song Han. 24

[computation reuse in diffusion] [Training-Free Consistent Text-to-Image Generation](https://arxiv.org/pdf/2402.03286). Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, Yuval Atzmon. 24

[privacy-preserving] [Privacy-Preserving Diffusion Model Using Homomorphic Encryption](https://arxiv.org/pdf/2403.05794). Yaojian Chen, Qiben Yan. 24

[computation reuse in diffusion] [Exploring Collaborative Distributed Diffusion-Based AI-Generated Content (AIGC) in Wireless Networks](https://arxiv.org/pdf/2304.03446). Hongyang Du, Ruichen Zhang, Dusit Niyato, Jiawen Kang, Zehui Xiong, Dong In Kim, Xuemin (Sherman)Shen, H. Vincent Poor. 24

[fast moe inference with cpu] [MOE-INFINITY: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/pdf/2401.14361). Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, Mahesh Marina. 24

[kv cache] [FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines](https://arxiv.org/pdf/2403.11421). Jiaao He, Jidong Zhai. 24

[kv cache] [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/pdf/2402.09398). Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, Beidi Chen. 24

[lora serving system] [CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference](https://arxiv.org/pdf/2401.11240). Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng, Xusheng Chen, Yizhou Shan, Binhang Yuan, Wei Wang. 24

[llm on mobile device] [LLM as a System Service on Mobile Devices](https://arxiv.org/pdf/2403.11805). Wangsong Yin, Mengwei Xu, Yuanchun Li, Xuanzhe Liu. 24

[inference serving system] [MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving](https://arxiv.org/pdf/2404.02015). Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, Hao Zhang. 24

[kv cache] [Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference](https://arxiv.org/pdf/2403.09054). Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J. Nair, Ilya Soloveychik, Purushotham Kamath. 24

[elastic sequence parallelism] [LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism](https://arxiv.org/pdf/2404.09526). Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, Xin Jin. 24

[video diffusion] [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/pdf/2401.03048). Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, Yu Qiao. 24

[image diffusion] [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206). Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach. 24

[kv cache compression] [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/pdf/2402.09398). Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, Beidi Chen. 24

[stealing private information] [Teach LLMs to Phish: Stealing Private Information from Language Models](https://arxiv.org/pdf/2403.00871). Ashwinee Panda, Christopher A. Choquette-Choo, Zhengming Zhang, Yaoqing Yang, Prateek Mittal. 24

[accelerator for edge devices] [FlexNN: A Dataflow-aware Flexible Deep Learning Accelerator for Energy-Efficient Edge Devices](https://arxiv.org/pdf/2403.09026). Arnab Raha, Deepak A. Mathaikutty, Soumendu K. Ghosh, Shamik Kundu. 24

[llm for control] [DriveLLM: Charting the Path Toward Full Autonomous Driving With Large Language Models](https://arxiv.org/pdf/2312.09245). Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, Hao Tian, Lewei Lu, Xizhou Zhu, Xiaogang Wang, Yu Qiao, Jifeng Dai. 24

[multiple LLM serving using spatial-temporal multiplexing] [MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving](https://arxiv.org/pdf/2404.02015). Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, Hao Zhang. 24

[heterogeneous computation in the level of neuron] [PowerInfer-2: Fast Large Language Model Inference on a Smartphone](https://arxiv.org/pdf/2406.06282). Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, Haibo Chen. 24

[INT4 quantization on diverse edge devices] [An Empirical Analysis and Resource Footprint Study of Deploying Large Language Models on Edge Devices](https://dl.acm.org/doi/pdf/10.1145/3603287.3651205). Nobel Dhar, Bobin Deng, Dan Lo, Xiaofeng Wu, Liang Zhao, and Kun Suo. 24

[1-bit LLM] [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764). Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei. 24

[kv cache eviction policy] [CORM: Cache Optimization with Recent Message for Large Language Model Inference](https://arxiv.org/pdf/2404.15949). Jincheng Dai, Zhuowei Huang, Haiyun Jiang, Chen Chen, Deng Cai, Wei Bi, Shuming Shi. 24

[kv cache merging approach] [Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks](https://arxiv.org/pdf/2407.08454). Zheng Wang, Boxiao Jin, Zhongzhi Yu, Minjia Zhang. 24

[reuse of kv caches across conversations] [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](https://arxiv.org/pdf/2403.19708). Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo. 24

[prefetch kv cache] [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/pdf/2406.19707). Wonbeom Lee, Jungi Lee, Junghwan Seo, Jaewoong Sim. 24

[publicly model as firmware] [Mobile Foundation Model as Firmware](https://arxiv.org/pdf/2308.14363). Jinliang Yuan, Chen Yang, Dongqi Cai, Shihe Wang, Xin Yuan, Zeling Zhang, Xiang Li, Dingge Zhang, Hanzi Mei, Xianqing Jia, Shangguang Wang, Mengwei Xu. 24

[kv cache offload] [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/pdf/2407.00079). Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu. 24

[distributed kv cache] [MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool](https://arxiv.org/pdf/2406.17565). Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, Yizhou Shan. 24

[chunk attention on heterogeneous resources] [Empowering 1000 tokens/second on-device LLM prefilling with mllm-NPU](https://arxiv.org/pdf/2407.05858v1). Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu. 24

[gpu virtual memory management] [vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving](https://arxiv.org/pdf/2407.15309). Jiale Xu, Rui Zhang, Cong Guo, Weiming Hu, Zihan Liu, Feiyang Wu, Yu Feng, Shixuan Sun, Changxu Shao, Yuhong Guo, Junping Zhao, Ke Zhang, Minyi Guo, Jingwen Leng. 24

[improve slm quality] [Hybrid SLM and LLM for Edge-Cloud Collaborative Inference](https://dl.acm.org/doi/pdf/10.1145/3662006.3662067). Zixu Hao, Huiqiang Jiang, Shiqi Jiang, Ju Ren, Ting Cao. 24

[pipeline] [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/pdf/2408.12757). Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Yufei Gao, Qinyu Xu, Tian Tang, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci. 24

[kv cache compression] [Layer-Condensed KV Cache for Efficient Inference of Large Language Models](https://arxiv.org/pdf/2405.10637). Haoyi Wu, Kewei Tu. 24

[kv offload] [LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management](https://arxiv.org/pdf/2410.00428). Yi Xiong, Hao Wu, Changxu Shao, Ziqing Wang, Rui Zhang, Yuhong Guo, Junping Zhao, Ke Zhang, Zhenxuan Pan. 24

[token tree] [LLMCad: Fast and Scalable On-device Large Language Model Inference](https://arxiv.org/pdf/2309.04255). Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu. 24

[sparse kv] [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/pdf/2406.10774). Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. 24

[graph pipeline parallelism for multimodal model] [GraphPipe: Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism](https://arxiv.org/pdf/2406.17145). Byungsoo Jeon, Mengdi Wu, Shiyi Cao, Sunghyun Kim, Sunghyun Park, Neeraj Aggarwal, Colin Unger, Daiyaan Arfeen, Peiyuan Liao, Xupeng Miao, Mohammad Alizadeh, Gregory R. Ganger, Tianqi Chen, Zhihao Jia. 24

[kv cache pruning] [DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads](https://arxiv.org/pdf/2410.10819). Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, Song Han. 24

[overlap all-to-all] [Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping](https://arxiv.org/pdf/2404.19429). Chenyu Jiang, Ye Tian, Zhen Jia, Shuai Zheng, Chuan Wu, Yida Wang. 24

[overlap all-to-all] [Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts](https://arxiv.org/pdf/2404.05019). Weilin Cai, Juyong Jiang, Le Qin, Junwei Cui, Sunghun Kim, Jiayi Huang. 24

[compute and load kv cache] [Compute Or Load KV Cache? Why Not Both?](https://arxiv.org/pdf/2410.03065). Shuowei Jin, Xueshen Liu, Qingzhao Zhang, Z. Morley Mao. 24

[quantization-aware training and knowledge distillation] [BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation](https://arxiv.org/pdf/2402.10631). Dayou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, Ningyi Xu. 24

[npu+pim] [NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing](https://arxiv.org/pdf/2403.00579). Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, Sanghyeon Lee, Hyungkyu Ham, Gwangsun Kim, Divya Mahajan, Jongse Park. 24

[chunk-level prefill for NPU] [Empowering 1000 tokens/second on-device LLM prefilling with mllm-NPU](https://arxiv.org/pdf/2407.05858). Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, Xuanzhe Liu. 24

[cloud-edge collaboration] [EdgeShard: Efficient LLM Inference via Collaborative Edge Computing](https://arxiv.org/pdf/2405.14371). Mingjin Zhang, Jiannong Cao, Xiaoming Shen, Zeyang Cui. 24

[request reordering and prefix sharing] [BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching](https://arxiv.org/pdf/2411.16102). Yilong Zhao, Shuo Yang, Kan Zhu, Lianmin Zheng, Baris Kasikci, Yang Zhou, Jiarong Xing, Ion Stoica. 24

[kv cache compression] [ASimple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://arxiv.org/pdf/2406.11430). Alessio Devoto, Yu Zhao, Simone Scardapane, Pasquale Minervini. 24

[kv cache reduction] [Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters](https://arxiv.org/pdf/2406.12335). Zhiyu Guo, Hidetaka Kamigaito, Taro Watanabe. 24

[llm on device] [AutoDroid: LLM-powered Task Automation in Android](https://arxiv.org/pdf/2308.15272). Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, Yunxin Liu. 24

[weight quantization] [AWQ: ACTIVATION-AWARE WEIGHT QUANTIZATION FOR ON-DEVICE LLM COMPRESSION AND ACCELERATION](https://arxiv.org/pdf/2306.00978). Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. 24

[llm within npu and flash] [Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM](https://arxiv.org/pdf/2409.15654). Zhongkai Yu, Shengwen Liang, Tianyun Ma, Yunke Cai, Ziyuan Nan, Di Huang, Xinkai Song, Yifan Hao, Jie Zhang, Tian Zhi, Yongwei Zhao, Zidong Du, Xing Hu, Qi Guo, Tianshi Chen. 24

[reduce attention overhead] [CHAI: Clustered Head Attention for Efficient LLM Inference](https://arxiv.org/pdf/2403.08058). Saurabh Agarwal, Bilge Acun, Basil Hosmer, Mostafa Elhoushi, Yejin Lee, Shivaram Venkataraman, Dimitris Papailiopoulos, Carole-Jean Wu. 24

[contextual sparsity] [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/pdf/2310.17157). Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, Beidi Chen. 24

[key-value cache compression] [Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference](https://arxiv.org/pdf/2403.09636). Piotr Nawrot, Adrian Łańcucki, Marcin Chochowski, David Tarjan, Edoardo M. Ponti. 24

[dynamic token pruning] [LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference](https://arxiv.org/pdf/2407.14057v1). Qichen Fu, Minsik Cho, Thomas Merth, Sachin Mehta, Mohammad Rastegari, Mahyar Najibi. 24

[sparse attention in training] [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/pdf/2502.11089). Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng. 24

[adaptive sparse weight] [CoreInfer: Accelerating Large Language Model Inference with Semantics-Inspired Adaptive Sparse Activation](https://arxiv.org/pdf/2410.18311v1). Qinsi Wang, Saeed Vahidian, Hancheng Ye, Jianyang Gu, Jianyi Zhang, Yiran Chen. 24

[adaptive sparse weight] [Ripple: Accelerating LLM Inference on Smartphones with Correlation-Aware Neuron Management](https://arxiv.org/pdf/2410.19274). Tuowei Wang, Ruwen Fan, Minxing Huang, Zixu Hao, Kun Li, Ting Cao, Youyou Lu, Yaoxue Zhang, Ju Ren. 24

[sparse kv cache] [Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://arxiv.org/pdf/2406.10774). Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. 24

[sparse kv cache] [ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference](https://arxiv.org/pdf/2410.21465). Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, Beidi Chen. 24

[sparse kv cache] [ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression](https://arxiv.org/pdf/2412.03213). Guangda Liu, Chengwei Li, Jieru Zhao, Chenqi Zhang, Minyi Guo. 24

[inference on heterogeneous accelerators] [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](https://arxiv.org/pdf/2501.14794). Le Chen, Dahu Feng, Erhu Feng, Rong Zhao, Yingrui Wang, Yubin Xia, Haibo Chen, Pinjie Xu. 24

[cache-aware routing] [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference](https://arxiv.org/pdf/2412.00099). Andrii Skliar, Ties van Rozendaal, Romain Lepert, Todor Boinovski, Mart van Baalen, Markus Nagel, Paul Whatmough, Babak Ehteshami Bejnordi. 24

[kv cache offload] [HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading](https://www.arxiv.org/pdf/2502.12574). Cheng Luo, Zefan Cai, Hanshi Sun, Jinqi Xiao, Bo Yuan, Wen Xiao, Junjie Hu, Jiawei Zhao, Beidi Chen, Anima Anandkumar. 24

[weight compression] [When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models](https://arxiv.org/pdf/2502.15443). Weilan Wang, Yu Mao, Dongdong Tang, Hongchao Du, Nan Guan, Chun Jason Xue. 25

[parallel context encoding] [APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](https://arxiv.org/pdf/2502.05431). Xinyu Yang, Tianqi Chen, Beidi Chen. 25

[sparce attention] [HashAttention: Semantic Sparsity for Faster Inference](https://arxiv.org/pdf/2412.14468v1). Aditya Desai, Shuo Yang, Alejandro Cuadron, Ana Klimovic, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. 25

[agent] [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/pdf/2502.13138). Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, Yuxiang Wu. 25

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)


This list is released into the public domain.