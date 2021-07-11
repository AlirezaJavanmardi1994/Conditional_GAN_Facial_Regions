# Conditional_GAN_for_Facial_Regions
Utilizing CGAN for generating four key patches of the face
Since the process of generating samples using GAN was completely unsupervised and out of control, some researchers were investigating methods to tackle this issue. Mirza and Osindero proposed Conditional generative adversarial network or CGAN for short.
In CGAN, both the generator and discriminator are conditioned on a control variable c which can be a class label or other kinds of information. This extra variable will play an important role in generating label-preserved samples, especially when there are two or more classes of data. In the generator, the input noise z and conditional variable c have been combined to construct the input of the network. Respectively, the input of the discriminator is a combination of real or generated data and conditional information.
It is worth mentioning that, the output of the discriminator will show the real value if and only if the input sample and corresponding conditional variable match each other.

![alt text](https://www.uplooder.net/img/image/97/2b73fde10b5cbf752ade35450408be0d/335c540c7c5fc7113d44bbb82484ce0e.png)
![alt text](https://www.uplooder.net/img/image/41/bad00f60263cbe92c5ba8bd247ff612e/Untitled.png)
