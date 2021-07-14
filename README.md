# Awesome Incremental Learning / Lifelong learning
## Survey
- <a name="todo"></a> Online Continual Learning in Image Classification: An Empirical Survey (**arXiv 2020**) [[paper](https://arxiv.org/abs/2101.10423)] [[code](https://github.com/RaptorMai/online-continual-learning)]
- <a name="todo"></a> Continual Lifelong Learning in Natural Language Processing: A Survey (**COLING 2020**) [[paper](https://www.aclweb.org/anthology/2020.coling-main.574/)]
- <a name="todo"></a> Class-incremental learning: survey and performance evaluation (**arXiv 2020**) [[paper](https://arxiv.org/abs/2010.15277)] [[code](https://github.com/mmasana/FACIL)]
- <a name="todo"></a> A Comprehensive Study of Class Incremental Learning Algorithms for Visual Tasks (**Neural Networks**) [[paper](https://arxiv.org/abs/2011.01844)] [[code](https://github.com/EdenBelouadah/class-incremental-learning/tree/master/cil)]
- <a name="todo"></a> A continual learning survey: Defying forgetting in classification tasks (**TPAMI 2021**) [[paper]](https://ieeexplore.ieee.org/abstract/document/9349197) [[arxiv](https://arxiv.org/pdf/1909.08383.pdf)]
- <a name="todo"></a> Continual Lifelong Learning with Neural Networks: A Review
 (**Neural Networks**) [[paper](https://arxiv.org/abs/1802.07569)]
## Papers
### 2021
- <a name="todo"></a> Continual and Multi-Task Architecture Search (**ACL, 2021**) [[paper](https://arxiv.org/pdf/1906.05226.pdf)]

    block-based的正交化处理可能更合理

- <a name="todo"></a> Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks (**NAACL, 2021**) [[paper](https://www.aclweb.org/anthology/2021.naacl-main.378.pdf)]
- <a name="todo"></a> Continual Learning for Text Classification with Information Disentanglement Based Regularization (**NAACL, 2021**) [[paper](https://www.aclweb.org/anthology/2021.naacl-main.218.pdf)]

    论文认为需要分开处理task-shared的task-specific的feature。分离这两个特征的方法是增加了两个辅助任务（预测task-id，预测两句话是顺序还是逆序），这两个都是用了交叉熵。论文还用了replay和正则化(两个feature分别有自己的正则化)。

- <a name="todo"></a> Prototype Augmentation and Self-Supervision for Incremental Learning (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> ORDisCo: Effective and Efficient Usage of Incremental Unlabeled Data for Semi-supervised Continual Learning (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_ORDisCo_Effective_and_Efficient_Usage_of_Incremental_Unlabeled_Data_for_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Incremental Learning via Rate Reduction (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Incremental_Learning_via_Rate_Reduction_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> IIRC: Incremental Implicitly-Refined Classification (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Abdelsalam_IIRC_Incremental_Implicitly-Refined_Classification_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)]

    在学习当前任务时，如果能同时学一些meta-domain的任务，那么可能未来在对新domain作adaptation的时候会比较简单，遗忘也会较轻，这是因为meta-domain有可能离未来的domain很近。meta-domain是用random的image transform实现的。论文本质还是在解决domain-incremental的问题

- <a name="todo"></a> Image De-raining via Continual Learning (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Image_De-Raining_via_Continual_Learning_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Continual Learning via Bit-Level Information Preserving (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_Continual_Learning_via_Bit-Level_Information_Preserving_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Hyper-LifelongGAN: Scalable Lifelong Learning for Image Conditioned Generation (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhai_Hyper-LifelongGAN_Scalable_Lifelong_Learning_for_Image_Conditioned_Generation_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Lifelong Person Re-Identification via Adaptive Knowledge Accumulation (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Pu_Lifelong_Person_Re-Identification_via_Adaptive_Knowledge_Accumulation_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Distilling Causal Effect of Data in Class-Incremental Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.01737)] 
- <a name="todo"></a> Self-Promoted Prototype Refinement for Few-Shot Class-Incremental Learning (**CVPR, 2021**) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Self-Promoted_Prototype_Refinement_for_Few-Shot_Class-Incremental_Learning_CVPR_2021_paper.pdf)] 
- <a name="todo"></a> Layerwise Optimization by Gradient Decomposition for Continual Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2105.07561)] 
- <a name="todo"></a> Adaptive Aggregation Networks for Class-Incremental Learning (**CVPR, 2021**) [[paper](https://arxiv.org/pdf/2010.05063.pdf)] 
- <a name="todo"></a> Incremental Few-Shot Instance Segmentation (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2105.05312)] 
- <a name="todo"></a> Efficient Feature Transformations for Discriminative and Generative Continual Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.13558)] 
- <a name="todo"></a> On Learning the Geodesic Path for Incremental Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2104.08572)]
- <a name="todo"></a> Few-Shot Incremental Learning with Continually Evolved Classifiers (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2104.03047)] 
- <a name="todo"></a> Rectification-based Knowledge Retention for Continual Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.16597)] 
- <a name="todo"></a> DER: Dynamically Expandable Representation for Class Incremental Learning (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.16788)] 
- <a name="todo"></a> Rainbow Memory: Continual Learning with a Memory of Diverse Samples (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.17230)] 
- <a name="todo"></a> Training Networks in Null Space of Feature Covariance for Continual Learning
 (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.07113)] 
- <a name="todo"></a> Semantic-aware Knowledge Distillation for Few-Shot Class-Incremental Learning
 (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.04059)] 
 - <a name="todo"></a> PLOP: Learning without Forgetting for Continual Semantic Segmentation
 (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2011.11390)] 
 - <a name="todo"></a> Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations
 (**CVPR, 2021**) [[paper](https://arxiv.org/abs/2103.06342)] 
- <a name="todo"></a> Online Class-Incremental Continual Learning with Adversarial Shapley Value(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2009.00093)] [[code](https://github.com/RaptorMai/online-continual-learning)]
- <a name="todo"></a> Lifelong and Continual Learning Dialogue Systems: Learning during Conversation(**AAAI, 2021**)  [[paper](https://www.cs.uic.edu/~liub/publications/LINC_paper_AAAI_2021_camera_ready.pdf)] 
- <a name="todo"></a> Continual learning for named entity recognition(**AAAI, 2021**) [[paper](https://www.amazon.science/publications/continual-learning-for-named-entity-recognition)] 
- <a name="todo"></a> Using Hindsight to Anchor Past Knowledge in Continual Learning(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2002.08165)] 
- <a name="todo"></a> Curriculum-Meta Learning for Order-Robust Continual Relation Extraction(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2101.01926)] 
- <a name="todo"></a> Continual Learning by Using Information of Each Class Holistically(**AAAI, 2021**) [[paper](https://www.cs.uic.edu/~liub/publications/AAAI2021_PCL.pdf)] 
- <a name="todo"></a> Gradient Regularized Contrastive Learning for Continual Domain Adaptation(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2007.12942)] 
- <a name="todo"></a> Unsupervised Model Adaptation for Continual Semantic Segmentation(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2009.12518)] 
- <a name="todo"></a> A Continual Learning Framework for Uncertainty-Aware Interactive Image Segmentation(**AAAI, 2021**) [[paper](https://www.aaai.org/AAAI21Papers/AAAI-2989.ZhengE.pdf)] 
- <a name="todo"></a> Do Not Forget to Attend to Uncertainty While Mitigating Catastrophic Forgetting(**WACV, 2021**) [[paper](https://openaccess.thecvf.com/content/WACV2021/html/Kurmi_Do_Not_Forget_to_Attend_to_Uncertainty_While_Mitigating_Catastrophic_WACV_2021_paper.html)] 
- <a name="todo"></a> Online Class-Incremental Continual Learning with Adversarial Shapley Value(**AAAI, 2021**) [[paper](https://arxiv.org/abs/2009.00093)] [[code](https://github.com/RaptorMai/online-continual-learning)]

    online的setting下，每次训练新的batch时，会从memory里选取一小部分进行合并，然后一起训练。选取的规则通过KNN的Shapley Value来确定，有一定道理，

### 2020
- <a name="todo"></a> Continual Learning for Natural Language Generation in Task-oriented Dialog Systems(**EMNLP, 2020**) [[paper](https://arxiv.org/abs/2010.00910)] 
- <a name="todo"></a> Distill and Replay for Continual Language Learning(**COLING, 2020**) [[paper](https://www.aclweb.org/anthology/2020.coling-main.318.pdf)] 
- <a name="todo"></a> Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks (**NeurIPS2020**) [[paper](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf)] [[code](https://github.com/ZixuanKe/CAT)]
- <a name="todo"></a> Calibrating CNNs for Lifelong Learning (**NeurIPS2020**) [[paper](http://people.ee.duke.edu/~lcarin/Final_Calibration_Incremental_Learning_NeurIPS_2020.pdf)] 

    训练完第一个任务后，固定CNN，对于每个新任务，CNN的每一层后面都接一个调整层，调整层的输出再作为CNN下一层的输入。

- <a name="todo"></a> Meta-Consolidation for Continual Learning (**NeurIPS2020**) [[paper](https://arxiv.org/abs/2010.00352?context=cs)] 

    挺有意思的工作。论文假设对于每个任务，模型的最优参数空间是个分布。对于当前任务，训练多个模型，用这些模型的参数，学习一个对当前任务的模型参数的分布（VAE）。在学完新任务后，把之前的VAE生成的模型参数和新得到的模型参数一起训练，得到新的VAE。最后测试前会采样多个模型进行finetune 和 ensemble。

- <a name="todo"></a> Understanding the Role of Training Regimes in Continual Learning (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2006.06958.pdf)] 

    论文提出来了一个遗忘的上界（和权重变化，loss关于权重的二次导的最大特征值有关），同时提出dropout regularization, large initial learning rate with exponential decay schedule at the end of each task, and small batch size有助于模型找到一个更平坦的最优值点。这可以有效地避免遗忘。 相当于是把泛化性与鲁棒性和遗忘率联系在了一起。

- <a name="todo"></a> Continual Learning with Node-Importance based Adaptive Group Sparse Regularization (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2003.13726.pdf)] 

    论文把神经网络中的节点分成重要的和不重要的，每次训练新任务后会更新这个划分。同时，不重要的节点会被初始化。论文的loss加入了稀疏的约束，同时还使用PGD的优化方法。

- <a name="todo"></a> Online Fast Adaptation and Knowledge Accumulation (OSAKA):a New Approach to Continual Learning (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2003.05856.pdf)] 

    论文提出了continual-MAML？

- <a name="todo"></a> Coresets via Bilevel Optimization for Continual Learning and Streaming (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2006.03875.pdf)] 
- <a name="todo"></a> RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2007.06271.pdf)] 
- <a name="todo"></a> Continual Deep Learning by Functional Regularisation of Memorable Past (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2004.14070.pdf)] 

    DNN2GP，把past任务的模型参数model成prior，不太懂。

- <a name="todo"></a> Dark Experience for General Continual Learning: a Strong, Simple Baseline (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2004.07211.pdf)] 

    论文针对没有task-boundary的情况，用reservoir采样替换herding采样，用logit的欧式距离替代概率的KL距离。

- <a name="todo"></a> GAN Memory with No Forgetting (**NeurIPS2020**) [[paper](https://arxiv.org/pdf/2006.07543.pdf)] 

    论文用Style-transfer的技术来解决continual GAN的问题，没有重训整个GAN，而是训练了一小部分task特定的参数。

- <a name="todo"></a> Adversarial Continual Learning (**ECCV2020**) [[paper](https://arxiv.org/abs/2003.09553)]  [[code](https://github.com/facebookresearch/Adversarial-Continual-Learning)]

    属于task_id aware的模型, 用对抗学习的方法分离feature中task-invariant和task-specific中的部分.

- <a name="todo"></a> REMIND Your Neural Network to Prevent Catastrophic Forgetting (**ECCV2020**) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530460.pdf)]  [[code](https://github.com/tyler-hayes/REMIND)]

    论文把nn分成两部分，前一部分frozen，后一部分可训练。exemplar不再是原始图片，而是前一部分的输出。另外还使用了tensor quantization来得到提高memory的储存效率。

- <a name="todo"></a> Incremental Meta-Learning via Indirect Discriminant Alignment (**ECCV2020**) [[paper](https://arxiv.org/abs/2002.04162)]

    论文做的是incremental的meta learning。训练incremental的meta数据时，让新类样本的特征在过旧类分类器后，能得到和之前相似的输出（公式6）。

- <a name="todo"></a> Memory-Efficient Incremental Learning Through Feature Adaptation (**ECCV2020**) [[paper](https://arxiv.org/abs/2004.00713)]

    论文使用了一个feature adaptation网络来解决任务切换导致的feature space不同的问题。

- <a name="todo"></a> PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning (**ECCV2020**) [[paper](https://arxiv.org/abs/2004.13513)] [[code](https://github.com/arthurdouillard/incremental_learning.pytorch)]

    分类时，对于每一类，用多个w和样本特征的距离来计算分类logit。 loss 是在CNN中间层加pooling后的Distillation约束 ， embedding层的Distillation约束。

- <a name="todo"></a> Reparameterizing Convolutions for Incremental Multi-Task Learning Without Task Interference (**ECCV2020**) [[paper](https://arxiv.org/abs/2007.12540)]
- <a name="todo"></a> Learning latent representions across multiple data domains using Lifelong VAEGAN (**ECCV2020**) [[paper](https://arxiv.org/abs/2007.10221)]
- <a name="todo"></a> Online Continual Learning under Extreme Memory Constraints	(**ECCV2020**) [[paper](https://arxiv.org/abs/2008.01510)]
- <a name="todo"></a> Class-Incremental Domain Adaptation (**ECCV2020**) [[paper](https://arxiv.org/abs/2008.01389)]
- <a name="todo"></a> More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning (**ECCV2020**) [[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710698.pdf)]

    在训练完一个任务后，又另外对多个新的边缘分类层进行训练（加入无监督的一些样本，同时这些分类层的输出要尽可能不一样）。在之后的任务中，这些层的输出会成为regularization loss的一部分.

- <a name="todo"></a> Piggyback GAN: Efficient Lifelong Learning for Image Conditioned Generation (**ECCV2020**) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660392.pdf)]
- <a name="todo"></a> GDumb: A Simple Approach that Questions Our Progress in Continual Learning	 (**ECCV2020**) [[paper](http://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)]

    直接选exemplars, 只基于exemplars训练，然后测试。

- <a name="todo"></a> Imbalanced Continual Learning with Partitioning Reservoir Sampling	 (**ECCV2020**) [[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580409.pdf)]

    解决multi-label问题，没仔细看。

- <a name="todo"></a> Topology-Preserving Class-Incremental Learning (**ECCV2020**) [[paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640256.pdf)]

    跟neural gas那篇比较像。没仔细看。
- <a name="todo"></a> GraphSAIL: Graph Structure Aware Incremental Learning for Recommender Systems (**CIKM2020**) [[paper](https://arxiv.org/abs/2008.13517)]
- <a name="todo"></a> OvA-INN: Continual Learning with Invertible Neural Networks (**IJCNN2020**) [[paper](https://arxiv.org/abs/2006.13772)]
- <a name="todo"></a> XtarNet: Learning to Extract Task-Adaptive Representation
for Incremental Few-Shot Learning (**ICLM2020**) [[paper](https://arxiv.org/pdf/2003.08561.pdf)]
- <a name="todo"></a> Optimal Continual Learning has Perfect Memory and is NP-HARD (**ICML2020**) [[paper](https://arxiv.org/pdf/2006.05188.pdf)]

    每个任务的最优参数空间是一个集合，那么持续学习就是在找这些集合的交里的的某个元素。在参数空间里判断集合是否有交是np-complete，所以最优持续学习是np-hard的。最优的持续学习需要perfect memory：新任务的最优参数空间是未知的，为了与新任务的参数空间求交，要知道到现在为止，最优的参数空间的交的集合，而不是集合的某个元素（当前模型的参数）。

- <a name="todo"></a> Neural Topic Modeling with Continual Lifelong Learning (**ICML2020**) [[paper](https://arxiv.org/pdf/2006.10909.pdf)]
- <a name="todo"></a> Continual Learning with Knowledge Transfer for Sentiment Classification (**ECML-PKDD2020**) [[paper](https://www.cs.uic.edu/~liub/publications/ECML-PKDD-2020.pdf)] [[code](https://github.com/ZixuanKe/LifelongSentClass)]
- <a name="todo"></a> Semantic Drift Compensation for Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00440.pdf)] [[code](https://github.com/yulu0724/SDC-IL)]

    先维护每类在embedding space中的mean（类中心）：当前任务的样本在训练前后会产生drift，用这些drift来计算每类中心点的drift。
分类是用NME做，loss 是triplet loss. 补充材料里使用了angular loss.

- <a name="todo"></a> Few-Shot Class-Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.10956.pdf)]

    用neural gas模型对feature space 建模，两个loss，第一个为了防止遗忘：neural gas中的点不能变化太多，这个loss考虑到了维度里面的方差。第二个loss为了避免过拟合新任务：代表新类的anchor远离原有的点，靠近新类的sample。neural gas在base时使用固定数量的anchor，之后新任务每次增加若干anchor.

- <a name="todo"></a> Modeling the Background for Incremental Learning in Semantic Segmentation (**CVPR2020**) [[paper](https://arxiv.org/pdf/2002.00718.pdf)]
- <a name="todo"></a> Incremental Few-Shot Object Detection (**CVPR2020**) [[paper](https://arxiv.org/pdf/2003.04668.pdf)]
- <a name="todo"></a> Incremental Learning In Online Scenario (**CVPR2020**) [[paper](https://arxiv.org/pdf/2003.13191.pdf)]
- <a name="todo"></a> Maintaining Discrimination and Fairness in Class Incremental Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/1911.07053.pdf)]
- <a name="todo"></a> Conditional Channel Gated Networks for Task-Aware Continual Learning (**CVPR2020**) [[paper](https://arxiv.org/pdf/2004.00070.pdf)]

    对于每个task，通过门结构选择来激活或不激活conv的某些channel（有点像attention）, 激活过的在之后任务中被freeze。模型包含task-classifier来确定task ID，使用了令网络稀疏的loss（使激活的通道尽量少），

- <a name="todo"></a> Continual Learning with Extended Kronecker-factored Approximate Curvature
 (**CVPR2020**) [[paper](https://arxiv.org/abs/2004.07507)]
- <a name="todo"></a> iTAML : An Incremental Task-Agnostic Meta-learning Approach (**CVPR2020**) [[paper](https://arxiv.org/pdf/2003.11652.pdf)] [[code](https://github.com/brjathu/iTAML)]

    和maml的思路很像。有exemplars。在训练过程中，让模型分别对每个任务i进行更新，得到若干临时模型（feature extractor + classifier_i）。临时模型与原模型的差值作为梯度，来更新原模型。推断时先判断是哪个任务，然后用相关的exemplar做fine-tune，再进行预测。效果特别好，诧异。

- <a name="todo"></a> Mnemonics Training: Multi-Class Incremental Learning without Forgetting (**CVPR2020**) [[paper](https://arxiv.org/pdf/2002.10211.pdf)] [[code](https://github.com/yaoyao-liu/mnemonics)]

    对于某个任务，如果在exemplars上训练得到的模型和在此任务的所有数据上训练得到的模型的结果一致，那这些exemplars就是好的exemplars。论文基于这样的目的，把exemplars当成参数，优化之，目标是在此任务的所有数据上表现好。像maml。额外的：weight transfer， fine-tune on exemplars， old emeplars adjustment（用上述思想，存疑）

- <a name="todo"></a> ScaIL: Classifier Weights Scaling for Class Incremental Learning (**WACV2020**) [[paper](https://arxiv.org/abs/2001.05755)]
- <a name="todo"></a> Accepted papers(**ICLR2020**) [[paper](https://docs.google.com/presentation/d/17s5Y8N9dypH-59tuwKaCp80NYBxTmtT6V-zOFlsH-SA/edit?usp=sharing)]
- <a name="todo"></a> Brain-inspired replay for continual learning with artificial neural networks (**Natrue Communications 2020**) [[paper](https://www.nature.com/articles/s41467-020-17866-2)] [[code](https://github.com/GMvandeVen/brain-inspired-replay)]
### 2019
- <a name="todo"></a> Compacting, Picking and Growing for Unforgetting Continual Learning (**NeurIPS2019**)[[paper](https://papers.nips.cc/paper/9518-compacting-picking-and-growing-for-unforgetting-continual-learning.pdf)][[code](https://github.com/ivclab/CPG)]

    论文先用一个完整的网络训练task 1， 然后迭代地裁剪网络(裁剪一下，训一下，使准确率不低于一值)，最后task1最终网络为A,被裁剪的为B。之后学习新任务，会在A上学习一个mask，同时利用所有的B。收敛后再迭代地裁剪B。

- <a name="todo"></a> Increasingly Packing Multiple Facial-Informatics Modules in A Unified Deep-Learning Model via Lifelong Learning (**ICMR2019**) [[paper](https://dl.acm.org/doi/10.1145/3323873.3325053)][[code](https://github.com/ivclab/PAE)]
- <a name="todo"></a> Towards Training Recurrent Neural Networks for Lifelong Learning (**Neural Computation 2019**) [[paper](https://arxiv.org/pdf/1811.07017.pdf)]
- <a name="todo"></a> Complementary Learning for Overcoming Catastrophic Forgetting Using Experience Replay  (**IJCAI2019**) [[paper]](https://www.ijcai.org/Proceedings/2019/0463.pdf)
- <a name="todo"></a> IL2M: Class Incremental Learning With Dual Memory
 (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)]
 
     额外保存old class的在当年任务的激活值的平均值，在新任务推断时，用这些值对激活值做修正。

- <a name="todo"></a> Incremental Learning Using Conditional Adversarial Networks
 (**ICCV2019**) [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Xiang_Incremental_Learning_Using_Conditional_Adversarial_Networks_ICCV_2019_paper.html)]
- <a name="todo"></a> Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability (**KDD2019**) [[paper](http://www.lamda.nju.edu.cn/yangy/KDD19.pdf)]
- <a name="todo"></a> Random Path Selection for Incremental Learning (**NeurIPS2019**) [[paper](https://arxiv.org/pdf/1906.01120.pdf)]
- <a name="todo"></a> Online Continual Learning with Maximal Interfered Retrieval (**NeurIPS2019**) [[paper](http://papers.neurips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval)]

     在更新当前任务的batch后，选择memory里那些loss上升最大的样本，来参与更新。

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

    模型在训练新任务时，对每一层可以选择reuse，adaptation(加一层conv 1X1)，new三种操作，用architecture weight去搜索新任务的结构。

- <a name="todo"></a> Efficient Lifelong Learning with A-GEM (**ICLR2019**) [[paper](https://openreview.net/forum?id=Hkf2_sC5FX)] [[code](https://github.com/facebookresearch/agem)]
- <a name="todo"></a> Learning to Learn without Forgetting By Maximizing Transfer and Minimizing Interference (**ICLR2019**) [[paper](https://openreview.net/forum?id=B1gTShAct7)] [[code](https://github.com/mattriemer/mer)]
- <a name="todo"></a> Overcoming Catastrophic Forgetting via Model Adaptation (**ICLR2019**) [[paper](https://openreview.net/forum?id=ryGvcoA5YX)] 
- <a name="todo"></a> A comprehensive, application-oriented study of catastrophic forgetting in DNNs (**ICLR2019**) [[paper](https://openreview.net/forum?id=BkloRs0qK7)] 

### 2018
- <a name="todo"></a> Memory Replay GANs: learning to generate images from new categories without forgetting
 (**NIPS2018**) [[paper](https://arxiv.org/abs/1809.02058)] [[code](https://github.com/WuChenshen/MeRGAN
 
     两种策略。
     1. 每次有新任务时，先用原c-GAN生成数据，这些数据和新数据一起用来训练新的c-GAN.
     2. 直接训练新c-GAN，同时用原GAN和新GAN用同一噪声(condition是原类的类别)训练出的图片做pixel-wise的aligned.
  
 - <a name="todo"></a> Selective Experience Replay for Lifelong Learning (**AAAI2018**) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16054/16703)] 

    与icarl类似，提出了几种选择exemplar的策略。

 - <a name="todo"></a> Reinforced Continual Learning (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7369-reinforced-continual-learning.pdf)] [[code](https://github.com/xujinfan/Reinforced-Continual-Learning)]
 - <a name="todo"></a> Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting (**NIPS2018**) [[paper](http://papers.nips.cc/paper/7631-online-structured-laplace-approximations-for-overcoming-catastrophic-forgetting.pdf)]
- <a name="todo"></a> Rotate your Networks: Better Weight Consolidation and Less Catastrophic Forgetting (R-EWC) (**ICPR2018**) [[paper](https://arxiv.org/abs/1802.02950)] [[code](https://github.com/xialeiliu/RotateNetworks)]
- <a name="todo"></a> Exemplar-Supported Generative Reproduction for Class Incremental Learning  (**BMVC2018**) [[paper](http://bmvc2018.org/contents/papers/0325.pdf)] [[code](https://github.com/TonyPod/ESGR)]
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
#### [1st Lifelong Learning for Machine Translation Shared Task at WMT20 (EMNLP 2020)](http://www.statmt.org/wmt20/lifelong-learning-task.html)
#### [Continual Learning in Computer Vision Challenge CVPR 2020](https://sites.google.com/view/clvision2020/challenge?authuser=0)
#### [Lifelong Robotic Vision Challenge IROS 2019](https://lifelong-robotic-vision.github.io)

## Feel free to contact me if you find any interesting paper is missing.
## Workshop papers are currently out due to space.

