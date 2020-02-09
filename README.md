## DeepBee

DeepBee is a project that aims to assist in the assessment of honey bee colonies using image processing and machine learning.

Video:
<div align="center">
  <a href="https://www.youtube.com/watch?v=W47sMDIS9zc"><img src="https://img.youtube.com/vi/W47sMDIS9zc/0.jpg" alt="Demonstration"></a>
</div>

### How to install?

#### Windows executable
You can download the windows executables from [this page](https://avsthiago.github.io/DeepBee/downloads/deepbee). You need to choose between the CPU or GPU version based on your computing resources.

#### From source
* Download [this repository](https://github.com/AvsThiago/DeepBee-source/archive/release-0.1.zip);
* install [Anaconda](https://docs.anaconda.com/anaconda/install/); 
* Create a conda environment based on the file [environment.yml](https://github.com/AvsThiago/DeepBee-source/blob/release-0.1/environment.yml);

```
conda env create -f environment.yml
``` 

* Before running the scripts you need to [activate](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) your conda environment. It's name is `deepbee`.


### How to use?

**Detecting and classifying the cells** 

* Download the [classification's model](https://drive.google.com/file/d/15P1tQ5658Hc6Q80PiygZOH-w45FG0nEj/view?usp=sharing
) before classifying the cells if you are running from the source code.  

* Move your comb images to the folder
    * ``/DeepBee-source/src/DeepBee/original_images/`` if you are running from the sources;
    * `/DeepBee/original_images/` if you are using the Windows version;
* Run the file `detection_and_classification.*`  
    * `/DeepBee-source/src/DeepBee/software/detection_and_classification.py` if you are running from the sources;
    * `DeepBee/software/detection_and_classification.exe` if you are running the Windows version;

**Visualising the predictions**

You can find the classification's output within the folder `/DeepBee/output/labeled_images` or:

* Run the file `visualization.*`
    * `/DeepBee-source/src/DeepBee/software/visualization.py` if you are running from the sources;
    * `/DeepBee/software/visualization.exe` if you are running the Windows version;

**Interacting with the visualization tool**

Once you have the visualization tool open you can interact with it as follows:

``N`` - Next image;

``P`` - Previous image;

``V`` - Toggle detections;

``keys 1 to 7`` - Defines the cell class to be added or changed;

``keys 1 to 7 + mouse click`` - Changes the cells class;

``A`` - Add cell;

``Mouse click on a cell`` - Also toggles a red dot in the center of the cell. Cells without the red dot can be used to retrain the model;

``D`` - Remove cell;

``Space`` - Enables moving mode;

``R`` - Enables region selection. Select the region using the mouse;

``BackSpace`` - Resets changes;

``S`` - Save changes;

``Esq`` - Quit;

**Exporting the predictions to a CSV file**

* Run the file `export_spreadsheet.*`
    * `/DeepBee-source/src/DeepBee/software/export_spreadsheet.py` if you are running from the sources;
    * `/DeepBee/software/export_spreadsheet.exe` if you are running the Windows version;


**Retraining the model**

* Once you fixed the predictions using the visualization tool you can retrain your model using the file `train.py` or `train.exe`.

* We advise you make the training using GPUs otherwise it easily take days to be trained. We also highly recommend to make a backup of the model `classification.h5` since the training can make the predictions worse.  


### Models

* Classification
* Segmentation

### Datasets

* Classification
    * Train: [images](https://github.com/AvsThiago/DeepBee-source/tree/release-0.1/src/data), [labels](https://github.com/AvsThiago/DeepBee-source/tree/release-0.1/src/data/resources);
    * Test: [images](https://github.com/AvsThiago/DeepBee-source/tree/release-0.1/src/data/resources), [labels](https://github.com/AvsThiago/DeepBee-source/tree/release-0.1/src/data/resources);
* Segmentation
    * [Train + Test](https://data.mendeley.com/datasets/db35fj73x5/1)

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