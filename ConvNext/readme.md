# ConvNeXt 
作者在这篇论文中指出随着Vision Transformer(*ViTs*)的提出，cv领域里的很多模型都是基于Transformer，而不再是以往的卷积神经网络(*ConvNet*)。但作者觉得ConvNet仍然有设计改进的空间，
所以在这篇论文中又提出了ConvNeXt网络，完全是基于ConvNet模式，来对标Transformer，并且取得了不错的效果。 以下是其中一段原文：
```
The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely 
from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, 
achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity 
and efficiency of standard ConvNets.
```
