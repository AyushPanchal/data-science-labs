Hello everyone, this is Ayush Panchal.

I'm here to explain my project, *CashewGuard: AI for Cashew Disease Classification* using convolutional neural networks and transfer learning.

I worked on this project with my teammates, Pooja Dave and Avanti Thale.

Let’s begin by understanding the problem and then discuss the potential solution.

As we know, managing diseases during the farming process can be challenging. Today, I’ll focus on cashew plants specifically.

Our objective is to identify various diseases affecting cashew plants, such as anthracnose, red rust, and gummosis, among others. The goal is to design an intelligent system that can classify the disease by analyzing an image of the infected leaf or trunk.

---

To achieve this, we leverage the power of computer vision, deep learning, and transfer learning techniques.

The Python libraries we used for this project include:
- **NumPy** and **Pandas** for data manipulation.
- **Matplotlib** and **Seaborn** for visualization.
- **OpenCV** for computer vision tasks.
- **Scikit-learn** for accuracy metrics and data splitting.
- **TensorFlow** for transfer learning and deep learning.

**Step 1: Data Collection**
We started by collecting images of healthy and infected cashew plants from Mendeley.com. The dataset link is provided in the video description.

**Step 2: Exploratory Data Analysis (EDA)**
We loaded the images and their respective class labels into variables for analysis. During EDA, we observed that the dataset was balanced, except for the gummosis class.

Here are example images from each class:
1. Red rust
2. Anthracnose
3. Healthy
4. Gummosis (trunk of the cashew tree)
5. Leaf miner

Notably, anthracnose and leaf miner images look quite similar.

**Step 3: Data Preprocessing**
We split the dataset into training and testing subsets and performed one-hot encoding. Since deep learning algorithms process numbers, this step was crucial for converting class labels into numerical format.

**Step 4: Transfer Learning**
Transfer learning is a technique that allows us to use pre-trained models for our specific tasks.

Deep convolutional neural network models can take days or weeks to train on large datasets. With transfer learning, we reuse the weights from pre-trained models, like those developed for ImageNet. These models can be downloaded and customized for new applications.

In this project, we used the **EfficientNetB0** model, initialized with weights from the ImageNet dataset. We set the `include_top` parameter to *False* to exclude the pre-built output layer, allowing us to add a custom output layer for our use case.

**Model Compilation**
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam for gradient descent
- **Metrics:** Accuracy

**Callback Functions**
- TensorBoard: For real-time visualization of accuracy and loss during training.
- ModelCheckpoint: To save the best model.
- ReduceLROnPlateau: To adjust the learning rate when a plateau is detected.

After 15 minutes of training, our model achieved a training accuracy of 99% and a validation accuracy of 98%.

Here’s a graph showing the changes in accuracy and loss during training.

**Step 5: Model Evaluation**
We evaluated the model using a classification report and confusion matrix.

The confusion matrix revealed that the model sometimes confused anthracnose with leaf miner, likely due to their visual similarity. This issue could be addressed by augmenting the dataset with more images for each class.

**Advanced Visualizations**
For each test image, we displayed:
- Predicted label: Green text for correct predictions and red for incorrect ones.
- Confidence percentage: Indicating the model’s certainty.

---

**Future Scope**
This model has the potential for real-world applications:
1. **Platform Integration:** Integrate the model into mobile or web applications, allowing users to upload images for disease detection.
2. **Expansion to Other Crops:** Extend this application to identify diseases in plants like tomatoes, maize, and potatoes.
3. **Autonomous Drone Systems:** Develop drones equipped with this model to monitor crop fields and detect diseases in real time.

Our tool can significantly benefit farmers by improving disease management and reducing crop loss.

Thank you for watching until the end! Please leave a like if you enjoyed the presentation, and share your suggestions or questions in the comments.

See you in the next video. Bye!

