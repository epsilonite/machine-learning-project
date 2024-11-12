### machine-learning-project
# Mammography Mass Classifier

### Be, Ritu, Tatyana, Caitlin
<br>

![nn_art](https://github.com/epsilonite/machine-learning-project/blob/main/images/nn_art_2.jpg)


---

## Project Overview:

### Motivation: 

Breast cancer is one of the most common cancers worldwide and the leading cause of cancer-related deaths among women, with over 2.29 million new cases and 666,000 deaths in 2022 alone. While survival rates are high in developed countries — over 90% in places like the U.S. — rates drop significantly in low-income countries, where access to early detection and treatment is often limited.

The disparities in breast cancer outcomes highlight the need for more accessible diagnostic tools, especially in underserved areas. Machine learning has the potential to make a big impact here, offering scalable, efficient, and accurate solutions where healthcare infrastructure is lacking. By using image classification models, we can improve early detection and outcomes, ultimately contributing to global efforts to reduce breast cancer mortality. This approach could help bridge the gap between regions with advanced medical systems and those without, improving access to life-saving diagnoses.

### Question: 

How can we leverage existing machine learning research to improve breast cancer diagnostics?

### Description: 

This project aims to explore the potential of machine learning in improving breast cancer diagnosis, addressing both the need for efficiency and global access to quality care. We used several machine learning models, like ResNet50, EfficientNet, and DenseNet, to analyze mammography images from the CBIS-DDSM dataset. By comparing these models, we aimed to identify the best approach for accurate diagnosis while considering computational efficiency.

[Dashboard for Breast Cancer Statistics](https://public.tableau.com/app/profile/rituparna.nandi/viz/BreastCancer_17313655373100/BreastCancerFacts?publish=yes)

---

## Other Links

[Google Drive](https://drive.google.com/drive/folders/1JP11rqUjeKADC7EEGXHP2PvhYDgKx2Rf?usp=drive_link)

[Project Slides](https://docs.google.com/presentation/d/1bDKMw3RxJN8R6Rw_AKEVLY7Ms4kcZ9B1iXscR8kd3bc/edit#slide=id.p)

---

## Table of Contents

[Data](https://github.com/epsilonite/machine-learning-project#data)

[Processing](https://github.com/epsilonite/machine-learning-project#processing)

[Models](https://github.com/epsilonite/machine-learning-project#models)

[Results](https://github.com/epsilonite/machine-learning-project#results)

[App](https://github.com/epsilonite/machine-learning-project?tab=readme-ov-file#app)

[Resources](https://github.com/epsilonite/machine-learning-project?tab=readme-ov-file#resources)


---

## Data

We used data from the CBIS-DDSM in the Cancer Imaging Archive. The CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is an updated and standardized version of the Digital Database for Screening Mammography (DDSM). The original DDSM consists of 2,620 scanned mammography studies, including normal, benign, and malignant cases, all with verified pathology information.

[CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

Example of Image: 

<p align="center">
  <img src="https://github.com/epsilonite/machine-learning-project/blob/main/code/data/cbis-ddsm-png/Mass-Training_P_00001_LEFT_CC.png" alt="Alt text" width="400" height="auto">
</p>

---

## Processing

Data Processing Workflow:

1.) File Hierarchy and Naming: Updated the file hierarchy and renamed files.

2.) DICOM Import: Imported provided DICOM files as NumPy arrays.

3.) Bounding Box Retrieval: Retrieved the pixel boundary box of regions of interest (ROIs) using the mask provided in the dataset.

4.) Padding: Added a 25% padding around the bounding box to ensure additional context around the ROI.

5.) Mammogram Extraction: Extracted mammogram data within the padded bounding box region.

6.) Normalization: Converted the 16-bit grayscale images to float values scaled between 0 and 255.

7.) Export: Saved the processed data as both TFRecord files and .npy files.

![image1]()
![image2]()

Can update formatting with below to change image size and move to middle of screen etc. 

---

## Models

### CNN Overview and Project Approach

CNNs are specialized neural networks optimized for image data. They perform convolutional operations to extract features from input images, using:

* Convolutional Layers: Apply filters (kernels) to scan the input image, detecting features like edges and textures, and generating feature maps.
* Pooling Layers: Reduce the spatial dimensions of feature maps, commonly using max pooling, to decrease the number of parameters and computational load while preserving significant information.
* Flatten Layers: Convert multi-dimensional feature maps into a one-dimensional vector for input into fully connected layers.
* Fully Connected (Dense) Layers: Integrate extracted features and perform the final classification.
* Dropout Layers: Randomly deactivate neurons during training to prevent overfitting and enhance model generalization.
* Batch Normalization Layers: Standardize the inputs to each layer to maintain training stability and accelerate convergence.

Example of CNN:

![cnn](https://github.com/epsilonite/machine-learning-project/blob/main/images/cnn_model_structure.png)

In our project, we evaluated various CNN architectures to classify mammography images, aiming to find the most accurate model for detecting abnormalities.

### Models We Used

* EfficientNet: Balances accuracy and resource efficiency, performing well on radiology tasks with fewer parameters - Ideal for scenarios requiring computational efficiency, but may not capture as intricate details as ResNet in complex medical images.
* ResNet50: Best overall for radiology due to its ability to learn intricate details in medical images with deeper networks.
* MobileNetV2/V3: Designed for low computational power, suitable for situations where fast inference is needed but might not achieve the highest accuracy on complex medical images. 
* Inception: Can perform well on radiology images, especially with its ability to handle different image scales, but may not be as widely preferred as ResNet in medical imaging applications. 
* DenseNet: Though capable of learning complex features, might not be as widely used in radiology as ResNet due to its higher computational cost.

ResNet50 Model:

![resnet](https://github.com/epsilonite/machine-learning-project/blob/main/images/resnet.png)
---

## Results

The ResNet50 (Residual Neural Network) model was most accurate, reflecting an overall strong performance.

### Training and Validation Accuracy and Loss

* The model’s training accuracy reached 100%, showing it effectively learned the patterns in the data. The validation accuracy settled around 90%, indicating strong generalization with no major signs of overfitting. Initial fluctuations in validation accuracy were likely addressed by the learning rate adjustments, especially with early stopping and the reduction in learning rate helping to stabilize the performance.

* The training loss progressively decreased to near-zero, with a final value close to 0.0016, demonstrating the model’s effective optimization on the training data. However, the validation loss plateaued around 0.55, showing a good balance without unnecessary complexity in the model. This suggests that the model achieved a good fit on the validation data as well.
<br>
<p align="center">
  <img src="https://github.com/epsilonite/machine-learning-project/blob/main/images/resnet_accuracy.png" alt="Alt text" width="400" height="auto">
</p>

### Precision, Recall, and F1 Score

* For classes with larger representation (0 and 2), precision, recall, and F1 scores were high, around 95%. Class 1, with fewer instances, had slightly lower precision and recall at 89%. This lower performance for the minority class reflects its limited data, which may have restricted the model’s ability to generalize as effectively.
<br>
<p align="center">
  <img src="https://github.com/epsilonite/machine-learning-project/blob/main/images/resnet_precision.png" alt="Alt text" width="400" height="auto">
</p>

### Confusion Matrix

* The matrix reveals high accuracy across all classes, with classes 0 and 2 being classified correctly 95% of the time and class 1 at 89%. The high accuracy and recall demonstrate the model’s effective distinction among classes.
<br>
<p align="center">
  <img src="https://github.com/epsilonite/machine-learning-project/blob/main/images/resnet_matrix.png" alt="Alt text" width="400" height="auto">
</p>

### Overall Test Performance

* With a test accuracy of 94.7% and a test loss of 0.26, the model exhibits strong reliability on new data, maintaining a high level of accuracy across all classes.

---

## App

We built a web application using Flask, Java, HTML, and CSS, which allows users to upload images and have them classified using a pre-trained ResNet Deep Learning model, which was our most accurate model. 

### Features
* Image Upload: Intuitive web interface for uploading images.
* Real-Time Classification: Immediate results with class label and confidence score.
* Responsive Design: Clean and user-friendly interface.
* Backend Processing: Flask handles server logic and model interaction, with Java for specific backend tasks.

### Technology Stack
* Flask/Python: Core web framework for handling requests and routing.
* Java: Used for backend performance tasks.
* HTML & CSS: Build and style the web interface.
* ResNet Model: Pre-trained for accurate image classification.

### How It Works
* Upload: Users upload an image.
* Processing: Image is preprocessed and sent to the model.
* Classification: Model returns a class label and confidence score.

<br>
<p align="center">
  <img src="https://github.com/epsilonite/machine-learning-project/blob/main/images/app_2.png" alt="Alt text" width="400" height="auto">
</p>

---

## Resources
### CBIS-DDSM: Curated Breast Imaging Subset of Digital Database for Screening Mammography
**CBIS-DDSM dataset:**<br>
Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set]. *The Cancer Imaging Archive*. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

**CBIS-DDSM description**<br>
Lee, Rita S., Hongming Shan, Babak Saboury, Adam Sieh, Michael W. Liu, and Despina Kontos. (2017). A Curated Mammography Data Set for Use in Computer-Aided Detection and Diagnosis Research. *Scientific Data*, 4, Article 170177. https://doi.org/10.1038/sdata.2017.177

### Breast Cancer | CNN Mammography:
**Mass Segmentation + Classification: Ensemble Learning: ResNet50 + LSTM + RF + XGB88**<br>
Malebary, Sharaf J., Arshad Hashmi. (2021). Automated Breast Mass Classification System Using Deep Learning and Ensemble Learning in Digital Mammogram. *IEEE Access*, 9, 55312–55328. https://doi.org/10.1109/ACCESS.2021.3071297

**Mass Classification**<br>
Enas M.F. El Houby, Nisreen I.R. Yassin. (2021). Malignant and nonmalignant classification of breast lesions in mammograms using convolutional neural networks. *Biomedical Signal Processing and Control*, 70, Article 102954. https://doi.org/10.1016/j.bspc.2021.102954

**Catalog of CNNs**<br>
Wang, Lulu. (2024). Mammography with Deep Learning for Breast Cancer Detection. *Frontiers in Oncology*, 14, Article 1281922. https://doi.org/10.3389/fonc.2024.1281922

**Guidelines**<br>
Abdelhafiz, Dina, Clifford Yang, Reda Ammar, and Sheida Nabavi. (2019). Deep convolutional neural networks for mammography: Advances, challenges and applications. *BMC Bioinformatics*, 20(11), Article 281. https://doi.org/10.1186/s12859-019-2823-4

**Cancer Detection: Kaggle Competition**<br>
Khan, Muhammad Aasim, Muhammad Attique Khan, Muhammad Sharif, Tariq Umer, and Muhammad Younas Javed. (2023). Breast cancer detection in mammography images: A CNN-based approach with feature selection. *Information*, 14(7), Article 410. https://doi.org/10.3390/info14070410

**Cancer Detection: Cubic SVM: NasNet + MobileNetV2**<br>
Zahoor, Saliha, Umar Shoaib, and Ikram Ullah Lali. (2022). Breast Cancer Mammograms Classification Using Deep Neural Network and Entropy-Controlled Whale Optimization Algorithm. *Diagnostics*, 12(2), Article 557. https://doi.org/10.3390/diagnostics12020557

**Cancer Detection**<br>
Mudeng, Vicky, Jin-woo Jeong and Se-woon Choe. (2022). Simply fine-tuned deep learning-based classification for breast cancer with mammograms. *Computers, Materials & Continua*, 73(3), 49077. https://doi.org/10.32604/cmc.2022.031046

**BI-RADS Classification**
Geras, Krzysztof J., Stacey Wolfson, Gene Kim, Eric Kim, Linda Moy, and Kyunghyun Cho. (2017). High-Resolution Breast Cancer Screening with Multi-View Deep Convolutional Neural Networks. *arXiv preprint*, arXiv:1703.07047. https://doi.org/10.48550/arXiv.1703.07047.

### CNN Resources:
Weirich, Alfred. (2020, July 4). Finetuning TensorFlow/Keras Networks - Basics using MobileNetV2 as an Example. *Medium*. https://medium.com/@alfred.weirich/finetuning-tensorflow-keras-networks-basics-using-mobilenetv2-as-an-example-8274859dc232

Chollet, François, & others. (2023). *Transfer learning & fine-tuning*. Keras. https://keras.io/guides/transfer_learning/

Chollet, François, & others. (2023). *Keras Applications*. Keras. https://keras.io/api/applications/

### Other Resources:
ChatGPT. (2024). Assistance with programming, debugging and formating. OpenAI.
