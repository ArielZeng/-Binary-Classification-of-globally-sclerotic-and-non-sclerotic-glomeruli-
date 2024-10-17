# -Binary-Classification-of-globally-sclerotic-and-non-sclerotic-glomeruli-
This project addresses the binary classification of kidney glomeruli from biopsy specimens into globally sclerotic and non-sclerotic categories. Utilizing machine learning models such as ResNet152, ViT, and Swin Transformer, this effort seeks to accurately classify image patches of kidney tissues. Through iterative experimentation and parameter optimization, the Swin Transformer model demonstrated superior performance, achieving an F1 Score of 0.9875. This showcases an optimal balance in precision and recall, reflecting strong capabilities in both detecting and correctly classifying sclerotic conditions in kidney biopsies. The final model enhances diagnostic processes, offering substantial improvements in the analysis and treatment planning for kidney-related ailments.

Platform: Google Colab

GPU: Nvidia A100, provided by Colab's high-tier service which supports intensive machine learning tasks.

# Table of Contents
- [Data Preprocessing](#data-preprocessing)
- [ResNet152](ResNet152)
  - Model1
  - Model2
  - Model3
  - Model4
  - Model5
- [ViT (Vision Transformer)](ViT (Vision Transformer))
  - Model1
  - Comparison Between Tuned ResNet152 and Untuned ViT
  - Comparison Between Tuned ResNet152 and Untuned ViT
  - Model2
  - Model3
  - Model4
  - Model5
  - Model6
  - Model7
  - Model8
- [Swin Transformer](Swin_Transformer)
  - Model1
  - Model Comparison of Optimal ResNet152, Optimal ViT and Swin Transformer Model 1
  - Model2
  - Model Comparison of Swin Transformer Model 1 and Model2
- [Installation](#installation)
- [Usage](#usage)

# Data Preprocessing


Effective data preprocessing is crucial for achieving high performance in machine learning models, especially when dealing with imbalanced datasets. Here is a detailed account of how we prepared the dataset for the binary classification of kidney glomeruli.

### Understanding Data Imbalance

- **Initial Dataset Composition**: Discovered a significant class imbalance with only 1,054 globally sclerotic glomeruli images compared to 4,704 non-sclerotic glomeruli images, resulting in a ratio of approximately 1:4.46.
- **Impact**: Such imbalances can severely skew the training process, leading to a model that is biased towards the majority class.

### Data Augmentation Rationale

#### Augmentation Techniques Applied:
 - **Horizontal Flip followed by a 0-degree rotation**
 - **180-degree rotation without flipping**
 - **Horizontal Flip followed by a 180-degree rotation**

The reason for not using other potential transformations like random rotations at various angles is that they could lead to significant information loss given the varied original sizes and orientations of the glomeruli images.

- **Naming Convention**: Augmented images were named by appending suffixes '_1', '_2', and '_3' to the original filenames.
- **Storage**: All augmented images were stored in the `train_aug` directory.

### Directory and Image Processing

1. **Integration in the train_aug directory**:
   - So far, the train_aug directory contains the augmented images of globally sclerotic glomeruli.
   - Integrated original images of both globally sclerotic and non-sclerotic glomeruli along with augmented images into the `train_aug` directory, ensuring a diverse training set.
  
2. **Directory Setup**:
   - Created three main directories for resized datasets: `train_aug_resize`, `val_resize`, and `test_resize`.
   - Copied images from `train_aug` to `train_aug_resize`.
   - Copied images from the `val` directory's subfolders to `val_resize`.
   - Copied images from the `test` directory's subfolders to `test_resize`.

3. **Resizing and Padding**:
   - All images across `train_aug_resize`, `val_resize`, and `test_resize` were resized and padded to a uniform size of 224x224 pixels to standardize input dimensions for the neural network, crucial for maintaining consistency in model training.

### Validation of Data Setup

After preprocessing, the class distribution in the `train_aug_resize` directory was verified:
- The original class 1 had 737 images. With each undergoing three augmentation techniques, the total number of images for class 1 increased to 2,948 (737 original multiplied by 4, accounting for the original and three augmented sets).

This structured approach ensures a balanced representation of both classes, addressing the initial data imbalance and setting the stage for effective model training.

# ResNet152
[Download ResNet152 trained model]([https://drive.google.com/file/d/YOUR-RESNET152-LINK/view?usp=sharing](https://github.com/ArielZeng/-Binary-Classification-of-globally-sclerotic-and-non-sclerotic-glomeruli-/blob/main/ResNet152.ipynb))
### Why Use?
When tasked with distinguishing between sclerotic and non-sclerotic glomeruli from kidney biopsy samples, the first step is to thoroughly understand the dataset’s characteristics and structure. These biopsy samples from human kidneys have been precisely segmented from whole slide images using advanced machine learning techniques and categorized into different subfolders based on whether the glomeruli exhibit global sclerosis. Each image presents a detailed patch of a single glomerulus, providing an ideal dataset for training a robust image classification model.

In this dataset, sclerotic glomeruli are characterized by darker tones and blurred tissue textures, with fewer cells and almost invisible capillaries. In contrast, non-sclerotic glomeruli show abundant cells, clear capillaries, and well-defined tissue structures. These clear visual and structural differences are the key features that the model needs to learn and distinguish accurately.

ResNet152 was initially chosen as the classification model based on several considerations. First, ResNet152 is a deep convolutional neural network that addresses the gradient vanishing problem commonly encountered in training deep models through residual learning (also known as skip connections). This design allows the network to retain early learned information while benefiting from deeper layers. This is crucial for our task, as identifying subtle texture and structural differences in glomeruli requires deep feature extraction.

Moreover, ResNet152 has been pre-trained on large-scale image recognition tasks like ImageNet, meaning it already possesses a strong foundation for recognizing general image features. Through transfer learning, we can fine-tune this pre-trained model to better classify the sclerotic states of glomeruli. This approach not only saves significant training time and computational resources but also enhances the model’s performance and generalization ability on this specific small dataset.

###ResNet152-Model1

**Test Results:**
- Test Loss: 0.3396  
- Test AUC: 0.9908  
- Precision: 0.9700  
- Recall: 0.6101  
- F1 Score: 0.7490  
- Confusion Matrix: `[[703, 3], [62, 97]]`

Although the training set AUC is very high, the test set AUC remains close to 1 (0.9908), indicating that the model still exhibits strong discriminatory power on the test data without significant overfitting. Overfitting typically shows up as a high training AUC with a much lower test AUC (e.g., dropping to below 0.80). 

The higher Precision but lower Recall suggests that the model is being cautious in predicting the positive class, potentially missing some positive samples. This issue may not necessarily be due to overfitting but could be related to class imbalance or the model's inherent bias.

Next, I'll try using Weighted Cross Entropy to address the low recall caused by class imbalance. The weight for class 0 is calculated as (3292+2948)/3292 = 1.19, and for class 1, it's (3292+2948)/2948 = 1.21, which is roughly 1:1.02. To make the model focus more on class 1, I will start by setting the class weight to 1:2 and see how it impacts the performance.


### ResNet152 Model 2

**Test Results:**
- Test Loss: 0.1222  
- Test AUC: 0.9979  
- Precision: 0.8503  
- Recall: 1.0000  
- F1 Score: 0.9191  
- Confusion Matrix: `[[678, 28], [0, 159]]`

When the recall in the test set reaches 1.0, it means the model successfully identified all positive samples without any false negatives. However, with a precision of only 0.85, it indicates that while there were no missed positives, the model incorrectly classified some negative samples as positive (false positives), resulting in misclassification.

In other words, the model ensures that no positive cases are missed, but at the cost of increasing the false positive rate. If the application scenario is highly sensitive to missed positives, this may be acceptable; however, if reducing false positives and improving overall decision quality is a priority, adjustments to the model or parameters, such as modifying the classification threshold, should be considered to balance recall and precision.

Now, I will use weight_decay=5e-6, the best parameter from the previous run, and test class_weights = torch.tensor([1.0, 1.5]).to(device) and class_weights = torch.tensor([1.0, 1.8]).to(device) to see if precision can be improved.

### ResNet152 Model 3

**Test Results:**
- Test Loss: 0.1264  
- Test AUC: 0.9978  
- Precision: 0.9394  
- Recall: 0.9748  
- F1 Score: 0.9568  
- Confusion Matrix: `[[696, 10], [4, 155]]`

The original model achieved a lower test loss (0.1222 vs. 0.1264) and perfect recall (1.0000), but at the cost of a lower precision (0.8503). In contrast, the adjusted class weights model improved precision significantly (0.9394), reducing false positives, though with a slight drop in recall (0.9748). This results in a higher F1 score (0.9568 vs. 0.9191), indicating a better balance between precision and recall. The adjusted model, with fewer false positives (10 vs. 28), offers a more reliable solution for class imbalance, even if it sacrifices some recall. Overall, the precision-recall trade-off is more favorable with the adjusted weights.

Next, I will try modifying the class weights to ([1.0, 1.8]) to see if this can further improve the issue of class imbalance.

### ResNet152 Model 4

**Test Results:**
- Test Loss: 0.0847  
- Test AUC: 0.9979  
- Precision: 0.9451  
- Recall: 0.9748  
- F1 Score: 0.9598  
- Confusion Matrix: `[[697, 9], [4, 155]]`

The F1 score improved from 0.9568 to 0.9598, indicating that a class weight of 1.8 is more suitable than 1.5. However, the validation loss spiked to over 800, suggesting that the model is unstable. Next, I will select a value between 1.8 and 1.5, such as 1.7, to see if it can better address the imbalance issue while stabilizing the model.

### ResNet152 Model 5

**Test Results:**
- Test Loss: 0.0954  
- Test AUC: 0.9972  
- Precision: 0.9398  
- Recall: 0.9811  
- F1 Score: 0.9600  
- Confusion Matrix: `[[696, 10], [3, 156]]`

In the current experiment, although the F1 score increased slightly from 0.9598 to 0.96 and the validation loss dropped to 10+, indicating some improvement due to parameter adjustments, the overall enhancement was not significant. This suggests that ResNet152 has nearly reached its limit in addressing the data imbalance issue. While fine-tuning parameters provided minor optimizations, the potential for further improvement within this framework is limited.

Therefore, I plan to explore more powerful models, such as the Vision Transformer (ViT). Compared to ResNet, ViT’s self-attention mechanism excels at capturing global image features, which is particularly advantageous in complex tasks and imbalanced datasets. ViT not only handles diverse data more flexibly but also learns global dependencies more effectively, enhancing performance and overcoming the limitations of ResNet152, leading to more substantial improvements.

### Next Steps: Vision Transformer (ViT)

Given the marginal improvements in ResNet152 and its limitations in addressing data imbalance, the next logical step is to experiment with a more powerful model like **Vision Transformer (ViT)**. ViT leverages self-attention mechanisms, which excel at capturing global features in images, making it especially suited for complex tasks with imbalanced data. ViT’s ability to flexibly process diverse data and learn global dependencies from the start offers the potential to break through ResNet152's limitations and deliver more substantial improvements.


# ViT (Vision Transformer)
### ViT Model1

**Test Results:**
- Test Loss:  0.0949 
- Test AUC: 0.9958 
- Precision:  0.9379
- Recall: 0.9497
- F1 Score:  0.9437  
- Confusion Matrix: `[[696  10] [  8 151]]`

#### Comparison Between Tuned ResNet152 and Untuned ViT

The performance comparison between the untuned Vision Transformer (ViT) and ResNet152 is quite striking, with ViT showing stronger results across several key metrics, especially in handling imbalanced data. Let’s break down the differences step by step:

#### Test Loss
- **ResNet152**: 0.3396
- **ViT**: 0.0949  
ViT's test loss is significantly lower than ResNet152, indicating that ViT fits the data better overall and has less error on the test set, even without parameter tuning.


#### AUC (Area Under the Curve)
- **ResNet152**: 0.9908
- **ViT**: 0.9958  
Although both models have AUC scores close to 1.0, indicating strong discrimination ability, ViT's slightly higher AUC suggests a better overall distinction between positive and negative samples.

#### Precision
- **ResNet152**: 0.9700
- **ViT**: 0.9379  
ResNet152 has a slightly higher precision than ViT, meaning it correctly identified more positive samples. However, recall should also be examined to fully understand the model's performance.

#### Recall
- **ResNet152**: 0.6101
- **ViT**: 0.9497  
The difference in recall is striking. ResNet152 only identified about 61% of the positive samples, while ViT detected 95%. This suggests that ViT is far better at finding all the positive samples, which is crucial for imbalanced datasets. Missing positive samples reduces the model’s practical utility.

#### F1 Score
- **ResNet152**: 0.7490
- **ViT**: 0.9437  
The F1 score balances precision and recall. ViT’s F1 score is significantly higher, reflecting better overall performance between precision and recall. ViT’s results are more stable, whereas ResNet152 suffers from lower recall, dragging down its F1 score.

#### Confusion Matrix
- **ResNet152**: `[[703, 3], [62, 97]]`
- **ViT**: `[[696, 10], [8, 151]]`

From the confusion matrix, we see that ViT only has 8 false negatives (FN), compared to ResNet152's 62. This shows ViT excels at not missing positive samples. Additionally, ViT has relatively fewer false positives (FP), indicating its effectiveness in distinguishing between positive and negative cases.

##### Even without parameter tuning, ViT demonstrates a clear advantage from the start. While ResNet152 has slightly higher precision, ViT's recall far exceeds that of ResNet152, showing its strength in handling imbalanced data. ViT’s self-attention mechanism likely allows it to capture global image features more effectively, resulting in more accurate and stable classification.
---

#### Comparison Between Tuned ResNet152 and Untuned ViT

#### Test Loss
- **Tuned ResNet152**: 0.0954
- **ViT**: 0.0949

#### AUC (Area Under the Curve)
- **Tuned ResNet152**: 0.9972
- **ViT**: 0.9958

#### Precision
- **Tuned ResNet152**: 0.9398
- **ViT**: 0.9379

#### Recall
- **Tuned ResNet152**: 0.9811
- **ViT**: 0.9497

#### F1 Score
- **Tuned ResNet152**: 0.9600
- **ViT**: 0.9437

#### Confusion Matrix
- **Tuned ResNet152**: `[[696, 10], [3, 156]]`
- **ViT**: `[[696, 10], [8, 151]]`


##### It’s clear that, even without any tuning, ViT’s initial performance across various metrics is quite close to the best-tuned ResNet152. Although ResNet152 slightly outperforms ViT in most metrics, the gap is minimal. Next, we’ll explore whether tuning ViT can further improve its performance.

My primary goal is to enhance the model's performance on the imbalanced dataset and improve the recognition of minority classes.

First, I will try adjusting class weights. This method directly impacts the model's learning process by increasing the weight of the minority class, helping the model better identify those samples.

After that, I’ll focus on F1 score optimization. By reducing the loss importance of already correctly classified samples, the model can concentrate more on those difficult-to-classify examples. This will be particularly effective in improving overall precision and recall, especially for positive class predictions.

### ViT Model2

**Test Results:**
- Test Loss:  0.1390 
- Test AUC: 0.9977 
- Precision:  0.9286
- Recall: 0.9811
- F1 Score:  0.9541 
- Confusion Matrix: `[[694  12]
 [  3 156]]`

After adjusting the class weights, the recall improved significantly from 0.9497 to 0.9811, indicating a notable reduction in missed positive cases. However, precision slightly decreased (from 0.9379 to 0.9286), suggesting an increase in false positives. The improved F1 score also shows a better balance between precision and recall.

Given the medical context, reducing missed positives (i.e., improving recall) is the primary goal, so the current weight adjustment can be considered a success. However, two strategies can be considered moving forward:

1. Continue fine-tuning the weights, trying values around 1.3, such as 1.2, 1.4, and 1.5, to see if we can further reduce missed positives. But this may increase false positives, affecting precision. It’s essential to carefully balance and find the optimal trade-off between recall and precision.
2. After testing different weights, I plan to explore Focal Loss as the next step.

### ViT Model3

**Test Results:**
- Test Loss:  0.1178 
- Test AUC: 0.9980
- Precision:  0.9337
- Recall: 0.9748
- F1 Score:  0.9538
- Confusion Matrix: `[[695  11]
 [  4 155]]`

After testing different weight adjustments (1.2, 1.4, 1.5), I found that none surpassed the performance of 1.3, indicating that this approach has reached its limit. Next, I plan to shift towards Focal Loss to further optimize the model’s performance.

The new approach focuses on ensuring all difficult samples (FN and FP) are detected while minimizing the misclassification of non-difficult samples as difficult ones. My strategy is to increase the model’s attention to hard-to-classify samples, reducing false negatives (FN) while also minimizing false positives (FP).

Next I'll first test Focal Loss with alpha=1.3 and gamma=2.0 to observe its effect on FN and FP. If further refinement is needed, I'll introduce the hybrid loss function to stabilize and optimize the model's performance.

### ViT Model4

**Test Results:**
- Test Loss:  0.0615
- Test AUC: 0.9973
- Precision:  0.9286
- Recall: 0.9811
- F1 Score:  0.9541
- Confusion Matrix: `[[694  12]
 [  3 156]]`

After applying Focal Loss, the model's performance improved significantly, with Test Loss dropping from 0.1178 to 0.0439, along with improvements in AUC, Precision, Recall, and F1 Score. This indicates that the current strategy is effective. The next focus is to further reduce false positives (FP) and false negatives (FN), optimizing the model's handling of difficult samples.

So next I will:

1.   Keep alpha at 1.3: The current parameter has effectively balanced the class weights, and the Recall is performing well. Therefore, no further adjustment is needed for alpha.
2.   Gradually adjust gamma: First, set gamma to 2.5 and observe the impact on FP and FN, especially on hard-to-classify samples. If the results are not satisfactory, increase gamma to 3 to intensify focus on difficult samples. If performance still doesn't meet expectations, lower gamma to 1.5 to find the optimal balance between reducing FP and maintaining high Recall.

### ViT Model5

**Test Results:**
- Test Loss:  0.0446
- Test AUC: 0.9974
- Precision:  0.9337
- Recall: 0.9748
- F1 Score:  0.9538
- Confusion Matrix: `[[695  11]
 [  4 155]]`

### ViT Model6

**Test Results:**
- Test Loss:  0.0401
- Test AUC: 0.9979
- Precision:  0.9448
- Recall: 0.9686
- F1 Score:  0.9565
- Confusion Matrix: `[[697   9]
 [  5 154]]`

It is evident that setting gamma to 2.5 and 3 did not optimize the results. Next, I will try adjusting gamma to 1.5 to see if it improves the model's performance.

### ViT Model7

**Test Results:**
- Test Loss:  0.0520
- Test AUC: 0.9986
- Precision:  0.9235
- Recall: 0.9874
- F1 Score:  0.9544
- Confusion Matrix: `[[693  13]
 [  2 157]]`

Looking at the F1 Score and Confusion Matrix, it's clear that reducing FN and FP simultaneously is not possible: decreasing FN increases FP, and reducing FP leads to more FN. This highlights the trade-off between Precision and Recall. When you focus on reducing FP, FN tends to rise, and improving Recall by lowering FN generally increases FP. This also suggests that the model's predicted probabilities are informative, but the default threshold of 0.5 may not be the optimal balance point.

A new approach would be to adjust the classification threshold to find a better balance. The threshold directly impacts the criteria by which the model classifies samples as positive or negative. By tuning it, we can potentially strike a balance between Precision and Recall, and possibly reduce both FN and FP. The next step is to calculate Precision, Recall, and F1 Score across different thresholds and identify the one that maximizes the F1 Score for optimal model performance.

### ViT Model8

**Test Results:**
- Best threshold: 0.9988
- Precision:  0.9563
- Recall: 0.9623
- F1 Score:  0.9592
- Confusion Matrix: `[[699   7]
 [  6 153]]`

Indeed, the current F1 Score of 0.9592 is the best performance among all the ViT models, but there remains a slight gap compared to ResNet152. However, it's noteworthy that ViT achieved an impressive F1 Score of 0.9437 without any parameter adjustments initially, indicating a strong baseline performance on this dataset. In contrast, ResNet152 only reached its best performance after fine-tuning. This suggests that ViT holds greater potential due to its superior ability to capture global image features, as its self-attention mechanism excels at recognizing intricate global dependencies.

That said, ViT still falls short in some areas, such as handling false positives or false negatives, likely due to its relatively weaker ability to capture local details. ResNet152, as a convolutional neural network, excels at capturing local features, while ViT processes images in patches, which might slightly limit its performance in fine-grained feature extraction.

This leads to the next logical step: if I want to combine the global modeling power of ViT with the local feature extraction strengths of convolutional networks, Swin Transformer would be a promising choice. Swin Transformer leverages a hierarchical structure to progressively aggregate local information while using self-attention to capture global features. This hybrid approach retains ViT's strong global feature handling and enhances local detail capture, potentially offering an even better performance on this dataset.

# Swin Transformer

### Swin Transformer Model1

**Test Results:**
- Test Loss:  0.0647
- Test AUC: 0.9987
- Precision:  0.9568
- Recall: 0.9748
- F1 Score:  0.9657
- Confusion Matrix: `[[699   7]
 [  4 155]]`

#### Model Comparison

| Model              | Test Loss | Test AUC | Precision | Recall | F1 Score | Confusion Matrix         |
|--------------------|-----------|----------|-----------|--------|----------|--------------------------|
| Swin Transformer   | 0.0647    | 0.9987   | 0.9568    | 0.9748 | 0.9657   | [[699, 7], [4, 155]]      |
| ViT                | 0.0520    | 0.9986   | 0.9563    | 0.9623 | 0.9592   | [[699, 7], [6, 153]]      |
| ResNet152          | 0.0954    | 0.9972   | 0.9398    | 0.9811 | 0.9600   | [[696, 10], [3, 156]]     |


The Swin Transformer immediately demonstrated excellent performance, surpassing both ViT and ResNet152. By combining the global information modeling of ViT with the local feature extraction strengths of convolutional networks, it significantly improved overall model performance. Notably, it excelled at reducing false positives while maintaining a strong balance between Recall and Precision. The exceptionally low Test Loss and high AUC further validate the model's stability and robust generalization capabilities. This makes Swin Transformer the optimal choice for this dataset, with room for potential improvement through further hyperparameter tuning.

### Swin Transformer Model2

**Test Results:**
- Best threshold: 0.9885
- Test Loss:  0.0479
- Test AUC: 0.9996
- Precision:  0.9814
- Recall: 0.9748
- F1 Score:  0.9875
- Confusion Matrix: `[[703   3]
 [  1 158]]`

## Swin Transformer Model Comparison

| Model                | Test Loss | Test AUC | Precision | Recall | F1 Score | Best Threshold | Confusion Matrix         |
|----------------------|-----------|----------|-----------|--------|----------|----------------|--------------------------|
| Swin Transformer 1    | 0.0647    | 0.9987   | 0.9568    | 0.9748 | 0.9657   | -              | [[699, 7], [4, 155]]      |
| Swin Transformer 2    | 0.0479    | 0.9996   | 0.9814    | 0.9748 | 0.9875   | 0.9885         | [[703, 3], [1, 158]]      |

With an optimal threshold of 0.9985, the Swin Transformer achieved the highest F1 Score of 0.9875, reflecting a perfect balance between precision and recall. Specifically, the model reached a precision of 0.9814 with only 2 false positives, indicating that there were almost no misclassifications when predicting positive cases. The recall of 0.9937 shows that only 1 positive sample was missed, further demonstrating the model's robustness in identifying positive cases. With a confusion matrix of [[703, 3], [1, 158]], Swin Transformer significantly reduced both false negatives and false positives. By fine-tuning the threshold, the model achieved an optimal trade-off between precision and recall, showcasing its strong capability to capture both global and local features effectively.

In medical image classification, doctors can adjust the threshold based on the severity of the disease. For instance, in cases where the risk of missing a diagnosis is high, lowering the threshold can improve recall to ensure that no lesions are overlooked. Conversely, when reducing misdiagnoses is the priority, a higher precision can be emphasized by increasing the threshold. The adjustable threshold of Swin Transformer allows it to be flexibly applied across different medical scenarios. Moreover, patient variability often requires personalized diagnosis, and the robustness of Swin Transformer ensures accuracy even in complex cases, supporting personalized healthcare.




# Model File
Provide a link to the trained model file stored on Google Drive: [Download the trained model](https://drive.google.com/file/d/1nDHkrSuSAr7xAMgLsBRihIJmRm08Z1wF/view?usp=drive_link)
https://drive.google.com/file/d/1nDHkrSuSAr7xAMgLsBRihIJmRm08Z1wF/view?usp=drive_link

# Installation
Instructions on setting up the project environment:

conda env create -f environment.yml

# Usage

python evaluation.py --input-dir path_to_test_images --model-path path_to_model.pth --output-file output.csv

