# Awesome Incremental Learning / Lifelong learning

## Feel free to contact me if you find any interesting paper is missing.

## ContinualAI wiki

#### [An Open Community of Researchers and Enthusiasts on Continual/Lifelong Learning for AI](https://www.continualai.org/)

## Workshops

#### [Continual learning workshop NeurIPS 2018](https://sites.google.com/view/continual2018/home?authuser=0)

#### [4th Lifelong Learning Workshop at ICML 2020](https://lifelongml.github.io/)

#### [Continual Learning in Computer Vision Workshop CVPR 2020](https://sites.google.com/view/clvision2020/overview)


## single-head v.s. multi-head; w exemplars or w/o exemplars

Being aware of different training and test settings.


## Challenges or Competitions
#### [Lifelong Robotic Vision Challenge IROS 2019](https://lifelong-robotic-vision.github.io)
#### [Continual Learning in Computer Vision Challenge CVPR 2020](https://sites.google.com/view/clvision2020/challenge?authuser=0)

## Survey
- <a name="todo"></a> Continual learning: A comparative study on how
to defy forgetting in classification tasks (**arXiv 2019**) [[paper](https://arxiv.org/pdf/1909.08383.pdf)]
- <a name="todo"></a> Continual Lifelong Learning with Neural Networks: A Review
 (**arXiv 2018**) [[paper](https://arxiv.org/abs/1802.07569)]
## Papers
### 2020
- <a name="todo"></a> Semantic Drift Compensation for Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00440.pdf)] [[code](https://github.com/yulu0724/SDC-IL)]

先维护每类在embedding space中的mean（类中心）：当前任务的样本在训练前后会产生drift，用这些drift来计算每类中心点的drift。
分类是用NME做，loss 是triplet loss. 补充材料里使用了angular loss.

- <a name="todo"></a> Few-Shot Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.10956.pdf)]
- <a name="todo"></a> Conditional Channel Gated Networks for Task-Aware Continual Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00070.pdf)]
- <a name="todo"></a> Continual Learning with Extended Kronecker-factored Approximate Curvature
 (**CVPR2020**) [[paper](https://arxiv.org/abs/2004.07507)]
- <a name="todo"></a> iTAML : An Incremental Task-Agnostic Meta-learning Approach (**CVPR2020**) [[paper](https://arxiv.org/pdf/2003.11652.pdf)] [[code](https://github.com/brjathu/iTAML)]
- <a name="todo"></a> Mnemonics Training: Multi-Class Incremental Learning without Forgetting (**CVPR2020**) [[paper](https://arxiv.org/pdf/2002.10211.pdf)] [[code](https://github.com/yaoyao-liu/mnemonics)]
- <a name="todo"></a> Accepted papers(**ICLR2020**) [[paper](https://docs.google.com/presentation/d/17s5Y8N9dypH-59tuwKaCp80NYBxTmtT6V-zOFlsH-SA/edit?usp=sharing)]
### 2019
- <a name="todo"></a> Compacting, Picking and Growing for Unforgetting Continual Learning (**NeurIPS2019**)[[paper](https://papers.nips.cc/paper/9518-compacting-picking-and-growing-for-unforgetting-continual-learning.pdf)][[code](https://github.com/ivclab/CPG)]
- <a name="todo"></a> Increasingly Packing Multiple Facial-Informatics Modules in A Unified Deep-Learning Model via Lifelong Learning (**ICMR2019**) [[paper](https://dl.acm.org/doi/10.1145/3323873.3325053)][[code](https://github.com/ivclab/PAE)]
- <a name="todo"></a> Towards Training Recurrent Neural Networks for Lifelong Learning (**Neural Computation 2019**) [[paper](https://arxiv.org/pdf/1811.07017.pdf)]
- <a name="todo"></a> IL2M: Class Incremental Learning With Dual Memory
 (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)]
- <a name="todo"></a> Incremental Learning Using Conditional Adversarial Networks
 (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Xiang_Incremental_Learning_Using_Conditional_Adversarial_Networks_ICCV_2019_paper.html)]
- <a name="todo"></a> Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability (**KDD2019**) [[paper](http://www.lamda.nju.edu.cn/yangy/KDD19.pdf)]
- <a name="todo"></a> Random Path Selection for Incremental Learning (**NeurIPS2019**) [[paper](https://arxiv.org/pdf/1906.01120.pdf)]
- <a name="todo"></a> Online Continual Learning with Maximal Interfered Retrieval (**NeurIPS2019**) [[paper](http://papers.neurips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting with Unlabeled Data in the Wild (**ICCV2019**) [[paper](https://arxiv.org/pdf/1903.12648.pdf)]
- <a name="todo"></a> Continual Learning by Asymmetric Loss Approximation
with Single-Side Overestimation (**ICCV2019**) [[paper](https://arxiv.org/pdf/1908.02984.pdf)]
- <a name="todo"></a> Lifelong GAN: Continual Learning for Conditional Image Generation (**ICCV2019**) [[paper](https://arxiv.org/pdf/1907.10107.pdf)]
- <a name="todo"></a> Continual learning of context-dependent processing in neural networks (**Nature Machine Intelligence 2019**) [[paper](https://rdcu.be/bOaa3)] [[code](https://github.com/beijixiong3510/OWM)] 
- <a name="todo"></a> Large Scale Incremental Learning (**CVPR2019**) [[paper](https://arxiv.org/abs/1905.13260)] [[code](https://github.com/wuyuebupt/LargeScaleIncrementalLearning)]
- <a name="todo"></a> Learning a Unified Classifier Incrementally via Rebalancing (**CVPR2019**) [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf)] [[code](https://github.com/hshustc/CVPR19_Incremental_Learning)]
- <a name="todo"></a> Learning Without Memorizing (**CVPR2019**) [[paper](https://arxiv.org/pdf/1811.08051.pdf)] 
- <a name="todo"></a> Learning to Remember: A Synaptic Plasticity Driven Framework for Continual Learning (**CVPR2019**) [[paper](https://arxiv.org/abs/1904.03137)] 
- <a name="todo"></a> Task-Free Continual Learning (**CVPR2019**) [[paper](https://arxiv.org/pdf/1812.03596.pdf)]
- <a name="todo"></a> Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting (**ICML2019**) [[paper](https://arxiv.org/abs/1904.00310)]
- <a name="todo"></a> Efficient Lifelong Learning with A-GEM (**ICLR2019**) [[paper](https://openreview.net/forum?id=Hkf2_sC5FX)] [[code](https://github.com/facebookresearch/agem)]
- <a name="todo"></a> Learning to Learn without Forgetting By Maximizing Transfer and Minimizing Interference (**ICLR2019**) [[paper](https://openreview.net/forum?id=B1gTShAct7)] [[code](https://github.com/mattriemer/mer)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting via Model Adaptation (**ICLR2019**) [[paper](https://openreview.net/forum?id=ryGvcoA5YX)] 
- <a name="todo"></a> A comprehensive, application-oriented study of catastrophic forgetting in DNNs (**ICLR2019**) [[paper](https://openreview.net/forum?id=BkloRs0qK7)] 
- <a name="todo"></a> Incremental Learning Techniques for Semantic Segmentation (**ICCVW2019**) [[paper](https://arxiv.org/abs/1907.13372)] [[code](https://github.com/LTTM/IL-SemSegm)]

### 2018
- <a name="todo"></a> Memory Replay GANs: learning to generate images from new categories without forgetting
 (**NIPS2018**) [[paper](https://arxiv.org/abs/1809.02058)] [[code](https://github.com/WuChenshen/MeRGAN
 
 两种策略。
 1. 每次有新任务时，先用原GAN生成数据，这些数据和新数据一起用来训练新的GAN.
 2. 直接训练新GAN，同时用原GAN和新GAN用同一噪声(condition是原类的类别)训练出的图片做pixel-wise的aligned.
 
 - <a name="todo"></a> Reinforced Continual Learning (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7369-reinforced-continual-learning.pdf)] [[code](https://github.com/xujinfan/Reinforced-Continual-Learning)]
 - <a name="todo"></a> Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7631-online-structured-laplace-approximations-for-overcoming-catastrophic-forgetting.pdf)]
- <a name="todo"></a> Rotate your Networks: Better Weight Consolidation and Less Catastrophic Forgetting (R-EWC) (**ICPR2018**) [[paper](https://arxiv.org/abs/1802.02950)] [[code](https://github.com/xialeiliu/RotateNetworks)]
- <a name="todo"></a> Exemplar-Supported Generative Reproduction for Class Incremental Learning  (**BMVC2018**) [[paper](http://bmvc2018.org/contents/papers/0325.pdf)] [[code](https://github.com/TonyPod/ESGR)]
- <a name="todo"></a> DeeSIL: Deep-Shallow Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/pdf/1808.06396.pdf)] 
- <a name="todo"></a> End-to-End Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/abs/1807.09536)][[code](https://github.com/fmcp/EndToEndIncrementalLearning)]
- <a name="todo"></a> Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence (**ECCV2018**)[[paper](http://arxiv-export-lb.library.cornell.edu/abs/1801.10112)] 
- <a name="todo"></a> Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights (**ECCV2018**) [[paper](https://arxiv.org/abs/1801.06519)] [[code](https://github.com/arunmallya/piggyback)]
 - <a name="todo"></a> Memory Aware Synapses: Learning what (not) to forget (**ECCV2018**) [[paper](https://arxiv.org/abs/1711.09601)] [[code](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses)]
  - <a name="todo"></a> Lifelong Learning via Progressive Distillation and Retrospection (**ECCV2018**) [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.pdf)] 
- <a name="todo"></a> PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning (**CVPR2018**) [[paper](https://arxiv.org/abs/1711.05769)] [[code](https://github.com/arunmallya/packnet)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting with Hard Attention to the Task (**ICML2018**) [[paper](http://proceedings.mlr.press/v80/serra18a.html)] [[code](https://github.com/joansj/hat)]
- <a name="todo"></a> Lifelong Learning with Dynamically Expandable Networks (**ICLR2018**) [[paper](https://openreview.net/forum?id=Sk7KsfW0-)] 
- <a name="todo"></a> FearNet: Brain-Inspired Model for Incremental Learning (**ICLR2018**) [[paper](https://openreview.net/forum?id=SJ1Xmf-Rb)] 

### 2017
- <a name="todo"></a> Incremental Learning of Object Detectors Without Catastrophic Forgetting
 (**ICCV2017**) [[paper](http://openaccess.thecvf.com/content_iccv_2017/html/Shmelkov_Incremental_Learning_of_ICCV_2017_paper.html)] 
- <a name="todo"></a> Overcoming catastrophic forgetting in neural networks (EWC) (**PNAS2017**) [[paper](https://arxiv.org/abs/1612.00796)] [[code](https://github.com/ariseff/overcoming-catastrophic)] [[code](https://github.com/stokesj/EWC)]
- <a name="todo"></a> Continual Learning Through Synaptic Intelligence (**ICML2017**) [[paper](http://proceedings.mlr.press/v70/zenke17a.html)] [[code](https://github.com/ganguli-lab/pathint)]
- <a name="todo"></a> Gradient Episodic Memory for Continual Learning (**NIPS2017**) [[paper](https://arxiv.org/abs/1706.08840)] [[code](https://github.com/facebookresearch/GradientEpisodicMemory)]
- <a name="todo"></a> iCaRL: Incremental Classifier and Representation Learning (**CVPR2017**) [[paper](https://arxiv.org/abs/1611.07725)] [[code](https://github.com/srebuffi/iCaRL)]

每类维护若干标本，使其均值接近类中心。总标本数量固定，每次为新类构建标本，旧类减少标本。

- <a name="todo"></a> Continual Learning with Deep Generative Replay (**NIPS2017**) [[paper](https://arxiv.org/abs/1705.08690)] [[code](https://github.com/kuc2477/pytorch-deep-generative-replay)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting by Incremental Moment Matching (**NIPS2017**) [[paper](https://arxiv.org/abs/1703.08475)] [[code](https://github.com/btjhjeon/IMM_tensorflow)]
- <a name="todo"></a> Expert Gate: Lifelong Learning with a Network of Experts (**CVPR2017**) [[paper](https://arxiv.org/abs/1611.06194)] 
- <a name="todo"></a> Encoder Based Lifelong Learning (**ICCV2017**) [[paper](https://arxiv.org/abs/1704.01920)] 

### 2016
- <a name="todo"></a> Learning without forgetting (**ECCV2016**) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)] [[code](https://github.com/lizhitwo/LearningWithoutForgetting)]



