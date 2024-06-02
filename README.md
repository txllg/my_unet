# my_unet
This study presents a deep learning model based on the U-Net architecture designed to address the challenge of accurate polyp segmentation in colonoscopy images. The link to my zenodo repository is below:[![DOI](https://zenodo.org/badge/775984203.svg)](https://zenodo.org/doi/10.5281/zenodo.11418329)
# Project Catalog Structure and Documentation Explanation
## **my_unet**<br/>
│   **unet.py**            : Helper scripts to support neural network training.<br/>
│   **predict.py**        : Script for generating predictions using trained models.<br/>
│   **train.py**           : Script for training neural network models.<br/>
│   **predict ans**        : Directory for storing predicted answers or results.<br/>

├── **nets**               : Directory containing implementations of various neural network architectures.<br/>
│   │   **attention.py**   : Implementation of attention mechanisms for neural networks,Support for the improved network                           proposed in this paper.<br/>
│   |   **mobilenetv2.py** : Implementation of MobileNetV2 architecture,To support the comparative experiments in this                             paper.<br/>
│   │   **resnet.py**      : Implementation of ResNet architecture,To support the comparative experiments in this paper.<br/>
│   │   **unet.py**        : _This is the core code of this paper_, which contains the source code of the segmentation model proposed in this paper as well as the code of the comparison and ablation experiments. **The source code of the proposed method is in a class called "Unet"**, and I have                           commented it above.<br/>
│   │   **vgg.py**         : Implementation of VGG architecture.vgg was used as the encoder of choice for the initial                             experiments, but was not used later.<br/>
│   │   **xception.py**    : Implementation of Xception architecture.To support the comparative experiments in this                                paper.<br/>

└── **utils**              : Directory containing utility scripts for data processing, training, and evaluation.<br/>
    │   **callbacks.py**   : Implementation of callback functions for model training.<br/>
    │   **dataloader.py**  : Implementation of data loader for general datasets.<br/>
    │   **utils.py**       : General utility functions.<br/>
    │   **utils_fit.py**   : Utility functions related to model fitting.<br/>
    │   **utils_metrics.py**: Utility functions for computing evaluation metrics.<br/>

# dataset
In this study, our primary experimental dataset is sourced from the CVC-ClinicDB (Bernal et al., 2015). The CVC-ClinicDB is a database comprising frames extracted from colonoscopy videos, serving as the official database for the training phase of the MICCAI 2015 Colonoscopy Polyp Detection Challenge.The link to the dataset is as follows：https://polyp.grand-challenge.org/CVCClinicDB/

# Reference
Alom MZ, Hasan M, Yakopcic C, Taha TM, Asari VK. 2018. Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.

Bernal J, Sánchez FJ, Fernández-Esparrach G, Gil D, Rodríguez C, Vilariño F. 2015. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized medical imaging and graphics 43:99–111.

Bernal J, Sánchez J, Vilarino F. 2012. Towards automatic polyp detection with a polyp appearance model. Pattern Recognition 45:3166–3182.

Bernal J, Tajkbaksh N, Sanchez FJ, Matuszewski BJ, Chen H, Yu L, Angermann Q, Romain O, Rustad B, Balasingham I. 2017. Comparative validation of polyp detection methods in video colonoscopy: results from the MICCAI 2015 endoscopic vision challenge. IEEE transactions on medical imaging 36:1231–1249.

Chen L-C, Papandreou G, Schroff F, Adam H. 2017. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.

Chen L-C, Zhu Y, Papandreou G, Schroff F, Adam H. 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation. In: Proceedings of the European conference on computer vision (ECCV). 801–818.

He K, Zhang X, Ren S, Sun J. 2016. Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

Hoff G, Foerster A, Vatn MH, Sauar J, Larsen S. 1986. Epidemiology of Polyps in the Rectum and Colon: Recovery and Evaluation of Unresected Polyps 2 Years after Detection. Scandinavian Journal of Gastroenterology 21:853–862. DOI: 10.3109/00365528609011130.

Hu J, Shen L, Sun G. 2018. Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 7132–7141.

Huang H, Lin L, Tong R, Hu H, Zhang Q, Iwamoto Y, Han X, Chen Y-W, Wu J. 2020. Unet 3+: A full-scale connected unet for medical image segmentation. In: ICASSP 2020-2020 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 1055–1059.

Jha D, Smedsrud PH, Riegler MA, Johansen D, De Lange T, Halvorsen P, Johansen HD. 2019. Resunet++: An advanced architecture for medical image segmentation. In: 2019 IEEE international symposium on multimedia (ISM). IEEE, 225–2255.

Ji G-P, Chou Y-C, Fan D-P, Chen G, Fu H, Jha D, Shao L. 2021. Progressively Normalized Self-Attention Network for Video Polyp Segmentation. In: De Bruijne M, Cattin PC, Cotin S, Padoy N, Speidel S, Zheng Y, Essert C eds. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. Lecture Notes in Computer Science. Cham: Springer International Publishing, 142–152. DOI: 10.1007/978-3-030-87193-2_14.

Kim SY, Cho JH, Kim EJ, Chung DH, Kim KK, Park YH, Kim YS. 2018. The efficacy of real-time colour Doppler flow imaging on endoscopic ultrasonography for differential diagnosis between neoplastic and non-neoplastic gallbladder polyps. European Radiology 28:1994–2002. DOI: 10.1007/s00330-017-5175-3.

Kim T, Lee H, Kim D. 2021. UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation. In: Proceedings of the 29th ACM International Conference on Multimedia. Virtual Event China: ACM, 2167–2175. DOI: 10.1145/3474085.3475375.

Long J, Shelhamer E, Darrell T. 2015. Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 3431–3440.

Mahmud T, Paul B, Fattah SA. 2021. PolypSegNet: A modified encoder-decoder architecture for automated polyp segmentation from colonoscopy images. Computers in Biology and Medicine 128:104119.

Murugesan B, Sarveswaran K, Shankaranarayana SM, Ram K, Joseph J, Sivaprakasam M. 2019. Psi-Net: Shape and boundary aware joint multi-task deep network for medical image segmentation. In: 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 7223–7226.

Ng D, Chen Y, Tian B, Fu Q, Chng ES. 2022. Convmixer: Feature interactive convolution with curriculum learning for small footprint and noisy far-field keyword spotting. In: ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 3603–3607.

Nogueira-Rodríguez A, Domínguez-Carbajales R, López-Fernández H, Iglesias Á, Cubiella J, Fdez-Riverola F, Reboiro-Jato M, Glez-Pena D. 2021. Deep neural networks approaches for detecting and classifying colorectal polyps. Neurocomputing 423:721–734.

Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32.

Qiu Z, Wang Z, Zhang M, Xu Z, Fan J, Xu L. 2022. BDG-Net: boundary distribution guided network for accurate polyp segmentation. In: Medical Imaging 2022: Image Processing. SPIE, 792–799.

Quan SY, Wei MT, Lee J, Mohi-Ud-Din R, Mostaghim R, Sachdev R, Siegel D, Friedlander Y, Friedland S. 2022. Clinical evaluation of a real-time artificial intelligence-based polyp detection system: a US multi-center pilot study. Scientific Reports 12:6598.

Ronneberger O, Fischer P, Brox T. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N, Hornegger J, Wells WM, Frangi AF eds. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Lecture Notes in Computer Science. Cham: Springer International Publishing, 234–241. DOI: 10.1007/978-3-319-24574-4_28.

Sanchez-Peralta LF, Bote-Curiel L, Picon A, Sanchez-Margallo FM, Pagador JB. 2020. Deep learning to find colorectal polyps in colonoscopy: A systematic literature review. Artificial intelligence in medicine 108:101923.

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser \Lukasz, Polosukhin I. 2017. Attention is all you need. Advances in neural information processing systems 30.

Wang Z, Li L, Anderson J, Harrington DP, Liang Z. 2004. Computer-aided detection and diagnosis of colon polyps with morphological and texture features. In: Medical Imaging 2004: Image Processing. SPIE, 972–979.

Wang S, Li L, Zhuang X. 2022. AttU-NET: Attention U-Net for Brain Tumor Segmentation. In: Crimi A, Bakas S eds. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. Lecture Notes in Computer Science. Cham: Springer International Publishing, 302–311. DOI: 10.1007/978-3-031-09002-8_27.

Wang Q, Wu B, Zhu P, Li P, Zuo W, Hu Q. 2020. ECA-Net: Efficient channel attention for deep convolutional neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11534–11542.

Woo S, Park J, Lee J-Y, Kweon IS. 2018. Cbam: Convolutional block attention module. In: Proceedings of the European conference on computer vision (ECCV). 3–19.

Xia S, Krishnan SM, Tjoa MP, Goh PM. 2003. A novel methodology for extracting colon’s lumen from colonoscopic images. Journal of Systemics, Cybernetics and Informatics 1:7–12.

Xie S, Girshick R, Dollár P, Tu Z, He K. 2017. Aggregated residual transformations for deep neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 1492–1500.

Yu L, Chen H, Dou Q, Qin J, Heng PA. 2016. Integrating online and offline three-dimensional deep learning for automated polyp detection in colonoscopy videos. IEEE journal of biomedical and health informatics 21:65–75.

Yu F, Koltun V. 2015. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122.

Zhang H, Wu C, Zhang Z, Zhu Y, Lin H, Zhang Z, Sun Y, He T, Mueller J, Manmatha R. 2022. Resnest: Split-attention networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2736–2746.

Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. 2019. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE transactions on medical imaging 39:1856–1867.
