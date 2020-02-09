# Please go to the [release page](https://github.com/AvsThiago/DeepBee-source/tree/release-0.1)


## DeepBee


### Citation
```
Thiago S. Alves, M. Alice Pinto, Paulo Ventura, Cátia J. Neves, David G. Biron, Arnaldo C. Junior, Pedro L. De Paula Filho, Pedro J. Rodrigues,
Automatic detection and classification of honey bee comb cells using deep learning,
Computers and Electronics in Agriculture,
Volume 170,
2020,
105244,
ISSN 0168-1699,
https://doi.org/10.1016/j.compag.2020.105244.
(http://www.sciencedirect.com/science/article/pii/S0168169919307690)
Abstract: In a scenario of worldwide honey bee decline, assessing colony strength is becoming increasingly important for sustainable beekeeping. Temporal counts of number of comb cells with brood and food reserves offers researchers data for multiple applications, such as modelling colony dynamics, and beekeepers information on colony strength, an indicator of colony health and honey yield. Counting cells manually in comb images is labour intensive, tedious, and prone to error. Herein, we developed a free software, named DeepBee©, capable of automatically detecting cells in comb images and classifying their contents into seven classes. By distinguishing cells occupied by eggs, larvae, capped brood, pollen, nectar, honey, and other, DeepBee© allows an unprecedented level of accuracy in cell classification. Using Circle Hough Transform and the semantic segmentation technique, we obtained a cell detection rate of 98.7%, which is 16.2% higher than the best result found in the literature. For classification of comb cells, we trained and evaluated thirteen different convolutional neural network (CNN) architectures, including: DenseNet (121, 169 and 201); InceptionResNetV2; InceptionV3; MobileNet; MobileNetV2; NasNet; NasNetMobile; ResNet50; VGG (16 and 19) and Xception. MobileNet revealed to be the best compromise between training cost, with ~9 s for processing all cells in a comb image, and accuracy, with an F1-Score of 94.3%. We show the technical details to build a complete pipeline for classifying and counting comb cells and we made the CNN models, source code, and datasets publicly available. With this effort, we hope to have expanded the frontier of apicultural precision analysis by providing a tool with high performance and source codes to foster improvement by third parties (https://github.com/AvsThiago/DeepBee-source).
Keywords: Cell classification; Apis mellifera L.; Semantic segmentation; Machine learning; Deep learning; DeepBee software

```
