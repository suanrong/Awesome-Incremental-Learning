# Awesome Incremental Learning / Lifelong learning
## Feel free to contact me if you find any interesting paper is missing.

## [Awesome Long-Tailed Recognition / Imbalanced Learning](https://github.com/xialeiliu/Awesome-LongTailed-Recognition)
#### Find it interesting that there are more shared techniques than I thought for incremental learning (exemplars-based). 

## ContinualAI wiki
#### [An Open Community of Researchers and Enthusiasts on Continual/Lifelong Learning for AI](https://www.continualai.org/)

## Workshops
#### [4th Lifelong Learning Workshop at ICML 2020](https://lifelongml.github.io/)
#### [Workshop on Continual Learning at ICML 2020](https://icml.cc/Conferences/2020/Schedule?showEvent=5743)
#### [Continual Learning in Computer Vision Workshop CVPR 2020](https://sites.google.com/view/clvision2020/overview)
#### [Continual learning workshop NeurIPS 2018](https://sites.google.com/view/continual2018/home?authuser=0)

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
- <a name="todo"></a> GDumb: A Simple Approach that Questions Our Progress in Continual Learning (**ECCV2020**) [[paper](http://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)]  [[code](https://github.com/drimpossible/GDumb)]

直接选exemplars, 只基于exemplars训练，然后测试。

- <a name="todo"></a> Adversarial Continual Learning (**ECCV2020**) [[paper](https://arxiv.org/abs/2003.09553)]  [[code](https://github.com/facebookresearch/Adversarial-Continual-Learning)]

属于task_id aware的模型, 用对抗学习的方法分离feature中task-invariant和task-specific中的部分.

- <a name="todo"></a> Incremental Meta-Learning via Indirect Discriminant Alignment (**ECCV2020**) [[paper](https://arxiv.org/abs/2002.04162)]

论文做的是incremental的meta learning。训练incremental的meta数据时，让新类样本的特征在过旧类分类器后，能得到和之前相似的输出（公式6）。

- <a name="todo"></a> Memory-Efficient Incremental Learning Through Feature Adaptation (**ECCV2020**) [[paper](https://arxiv.org/abs/2004.00713)]
- <a name="todo"></a> Small-Task Incremental Learning (**ECCV2020**) [[paper](https://arxiv.org/abs/2004.13513)]
- <a name="todo"></a> Reparameterizing Convolutions for Incremental Multi-Task Learning Without Task Interference (**ECCV2020**) [[paper](https://arxiv.org/abs/2007.12540)]
- <a name="todo"></a> Learning latent representions across multiple data domains using Lifelong VAEGAN (**ECCV2020**) [[paper](https://arxiv.org/abs/2007.10221)]
- <a name="todo"></a> Online Continual Learning under Extreme Memory Constraints	(**ECCV2020**) [[paper](https://arxiv.org/abs/2008.01510)]
- <a name="todo"></a> Class-Incremental Domain Adaptation (**ECCV2020**) [[paper](https://arxiv.org/abs/2008.01389)]
- <a name="todo"></a> More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning (**ECCV2020**) [[paper]()]
- <a name="todo"></a> Piggyback GAN: Efficient Lifelong Learning for Image Conditioned Generation (**ECCV2020**) [[paper]()]
- <a name="todo"></a> Greedy Sampler and Dumb Learner: A Surprisingly Effective Approach for Continual Learning	 (**ECCV2020**) [[paper]()]
- <a name="todo"></a> Imbalanced Continual Learning with Partitioning Reservoir Sampling	 (**ECCV2020**) [[paper]()]
- <a name="todo"></a> Topology-Preserving Class-Incremental Learning (**ECCV2020**) [[paper]()]
- <a name="todo"></a> OvA-INN: Continual Learning with Invertible Neural Networks (**IJCNN2020**) [[paper](https://arxiv.org/abs/2006.13772)]
- <a name="todo"></a> XtarNet: Learning to Extract Task-Adaptive Representation
for Incremental Few-Shot Learning (**ICLM2020**) [[paper](https://arxiv.org/pdf/2003.08561.pdf)]
- <a name="todo"></a> Optimal Continual Learning has Perfect Memory and is NP-HARD (**ICML2020**) [[paper](https://arxiv.org/pdf/2006.05188.pdf)]

每个任务的最优参数空间是一个集合，那么持续学习就是在找这些集合的交里的的某个元素。
在参数空间里判断集合是否有交是np-complete，所以最优持续学习是np-hard的。

最优的持续学习需要perfect memory：新任务的最优参数空间是未知的，为了与新任务的参数空间求交，要知道到现在为止，最优的参数空间的交的集合，而不是集合的某个元素（当前模型的参数）。

- <a name="todo"></a> Neural Topic Modeling with Continual Lifelong Learning (**ICML2020**) [[paper](https://arxiv.org/pdf/2006.10909.pdf)]
- <a name="todo"></a> Semantic Drift Compensation for Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00440.pdf)] [[code](https://github.com/yulu0724/SDC-IL)]

先维护每类在embedding space中的mean（类中心）：当前任务的样本在训练前后会产生drift，用这些drift来计算每类中心点的drift。
分类是用NME做，loss 是triplet loss. 补充材料里使用了angular loss.

- <a name="todo"></a> Few-Shot Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.10956.pdf)]

用neural gas模型对feature space 建模，两个loss，第一个为了防止遗忘：neural gas中的点不能变化太多，这个loss考虑到了维度里面的方差。
第二个loss为了避免过拟合新任务：代表新类的anchor远离原有的点，靠近新类的sample。
neural gas在base时使用固定数量的anchor，之后新任务每次增加若干anchor.

- <a name="todo"></a> Conditional Channel Gated Networks for Task-Aware Continual Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00070.pdf)]

对于每个task，通过门结构选择来激活或不激活conv的某些channel（有点像attention）, 激活过的在之后任务中被freeze。模型包含task-classifier来确定task ID，使用了令网络稀疏的loss（使激活的通道尽量少），

- <a name="todo"></a> Continual Learning with Extended Kronecker-factored Approximate Curvature
 (**CVPR2020**) [[paper](https://arxiv.org/abs/2004.07507)]
- <a name="todo"></a> iTAML : An Incremental Task-Agnostic Meta-learning Approach (**CVPR2020**) [[paper](https://arxiv.org/pdf/2003.11652.pdf)] [[code](https://github.com/brjathu/iTAML)]

和maml的思路很像。有exemplars。在训练过程中，让模型分别对每个任务i进行更新，得到若干临时模型（feature extractor + classifier_i）。临时模型与原模型的差值作为梯度，来更新原模型。推断时先判断是哪个任务，然后用相关的exemplar做fine-tune，再进行预测。

效果特别好，诧异。

- <a name="todo"></a> Mnemonics Training: Multi-Class Incremental Learning without Forgetting (**CVPR2020**) [[paper](https://arxiv.org/pdf/2002.10211.pdf)] [[code](https://github.com/yaoyao-liu/mnemonics)]

对于某个任务，如果在exemplars上训练得到的模型和在此任务的所有数据上训练得到的模型的结果一致，那这些exemplars就是好的exemplars。论文基于这样的目的，把exemplars当成参数，优化之，目标是在此任务的所有数据上表现好。像maml。

额外的：weight transfer， fine-tune on exemplars， old emeplars adjustment（用上述思想，存疑）

- <a name="todo"></a> Accepted papers(**ICLR2020**) [[paper](https://docs.google.com/presentation/d/17s5Y8N9dypH-59tuwKaCp80NYBxTmtT6V-zOFlsH-SA/edit?usp=sharing)]
### 2019
- <a name="todo"></a> Compacting, Picking and Growing for Unforgetting Continual Learning (**NeurIPS2019**)[[paper](https://papers.nips.cc/paper/9518-compacting-picking-and-growing-for-unforgetting-continual-learning.pdf)][[code](https://github.com/ivclab/CPG)]
- <a name="todo"></a> Increasingly Packing Multiple Facial-Informatics Modules in A Unified Deep-Learning Model via Lifelong Learning (**ICMR2019**) [[paper](https://dl.acm.org/doi/10.1145/3323873.3325053)][[code](https://github.com/ivclab/PAE)]
- <a name="todo"></a> Towards Training Recurrent Neural Networks for Lifelong Learning (**Neural Computation 2019**) [[paper](https://arxiv.org/pdf/1811.07017.pdf)]
- <a name="todo"></a> IL2M: Class Incremental Learning With Dual Memory
 (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)]
 
 额外保存old class的在当年任务的激活值的平均值，在新任务推断时，用这些值对激活值做修正。

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

在训练新任务时，除了分类loss和蒸馏loss，还增加了对attention map的蒸馏loss.

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
 1. 每次有新任务时，先用原c-GAN生成数据，这些数据和新数据一起用来训练新的c-GAN.
 2. 直接训练新c-GAN，同时用原GAN和新GAN用同一噪声(condition是原类的类别)训练出的图片做pixel-wise的aligned.
 
 - <a name="todo"></a> Reinforced Continual Learning (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7369-reinforced-continual-learning.pdf)] [[code](https://github.com/xujinfan/Reinforced-Continual-Learning)]
 - <a name="todo"></a> Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7631-online-structured-laplace-approximations-for-overcoming-catastrophic-forgetting.pdf)]
- <a name="todo"></a> Rotate your Networks: Better Weight Consolidation and Less Catastrophic Forgetting (R-EWC) (**ICPR2018**) [[paper](https://arxiv.org/abs/1802.02950)] [[code](https://github.com/xialeiliu/RotateNetworks)]
- <a name="todo"></a> Exemplar-Supported Generative Reproduction for Class Incremental Learning  (**BMVC2018**) [[paper](http://bmvc2018.org/contents/papers/0325.pdf)] [[code](https://github.com/TonyPod/ESGR)]
- <a name="todo"></a> DeeSIL: Deep-Shallow Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/pdf/1808.06396.pdf)] 
- <a name="todo"></a> End-to-End Incremental Learning (**ECCV2018**) [[paper](https://arxiv.org/abs/1807.09536)][[code](https://github.com/fmcp/EndToEndIncrementalLearning)]

和icarl很像，区别在于icarl训练时把旧类别的label替换为old model的输出，然后计算分类loss。这篇直接把分类loss和蒸馏loss分别计算了。同时还加了数据增强和gradient noise。同时，在训练完一个任务后，使用了balance的dataset进行fine-tune，fine-tune过程也使用了分类loss和蒸馏loss。根据它的ablation study, 没有数据增强和balance fine-tune的话，它的效果并不如icarl.

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

使用GAN生成数据参与训练，每次再用合并数据重新训练GAN.

- <a name="todo"></a> Overcoming Catastrophic Forgetting by Incremental Moment Matching (**NIPS2017**) [[paper](https://arxiv.org/abs/1703.08475)] [[code](https://github.com/btjhjeon/IMM_tensorflow)]
- <a name="todo"></a> Expert Gate: Lifelong Learning with a Network of Experts (**CVPR2017**) [[paper](https://arxiv.org/abs/1611.06194)] 
- <a name="todo"></a> Encoder Based Lifelong Learning (**ICCV2017**) [[paper](https://arxiv.org/abs/1704.01920)] 

### 2016
- <a name="todo"></a> Learning without forgetting (**ECCV2016**) [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)] [[code](https://github.com/lizhitwo/LearningWithoutForgetting)]



