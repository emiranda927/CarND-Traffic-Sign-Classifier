#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. The submission includes the project code.

You're reading it! Here is a link to my [project code](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas and numpy libraries (.shape and np.unique) to calculate summary statistics of the traffic signs data set:

* The size of  the training set is 27839
* The size of the validation set is 12630
* The size of test set is 6960
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

![](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/Visualizations/Sign_CID_Visualization.png) 
![](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/Visualizations/Original_DataSet_Vis.png) 

Each image was randomly selected from the training data and paired with the appropriate Class ID. A dictionary pairing the keys and values was created for ease of identifying and displaying results later in the implementation. 

The bar chart shows the frequency distribution of the dataset. It is clear from this chart that some of the classes were underrepresented. The maximum number of images in a class was 1607 and and the minimum number of images in a class was 141. That is a large delta that  could cause issues with certain classes during training. This issue was rectified through data augmentation (see pre-processing section for details).

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The following is a summary of the pre-processing steps I took:

| Processing Technique	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Image Filtering  		| Applied a Median Filter to images				| 
| Augmentation	     	| Extended original data with above processing 	|
| Micro-Rotations		| Rotated every image by a random delta (4x)	|
| Augmentation	      	| Extended previous data with above processing	|
| Grayscale			    | Converted augmented data to grayscale			|
| Normalization			| Used sklearn preprocessing.scale to normalize	|
| Shuffle				| Shuffled augmented dataset					|

When training, I envisioned that the CNN would benefit from having more generalized images to train on. Denoising is a common operation used in image preprocessing which I utilized to accomplish this. I applied a median filter (ndimage.median_filter) to denoise the training set images and blur them slightly. I chose a median filter over a gaussian blur because the median filter preserves the edges of the image better than a guassian filter. I added these filtered images directly onto the original dataset as an extension of what we already had. The one conern I have with employing the denoise operation are the computational resources required with increasing dataset sizes

The next problem I wanted to tackle was the issue of training example class disparity. In order to rectify this, I applied random "micro-rotations" to all images in a class until each Class-ID had 5000 images in the training set. The reason I used micro-rotations (as opposed to vertical/horizontal flipping) was to preserve the integrity of every training example. I could guarantee the invariance of each image while avoiding the introduction of redundant examples into the training set. This technique can be applied to any image used in training a convolutional neural network without having to split off select training id's that only work with specific geometric transformations.

The results of employing the micro-rotation operations was in increase in the dataset from 27,839 examples to 215,000 training examples. Below is the frequency distribution of the augmented dataset:

![](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/Visualizations/Augmented_DataSet_Vis.png) 

The last two steps were to transfrom the augmented dataset to grayscale and normalize to mean 0 and unit standard deviation ([preprocessing.scale()](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)). The best reason to transfrom the dataset into grayscale is to reduce the computational load (going from 3 channels to 1) without a noticeable loss in performance. It's been shown that color has a negligible impact on training CNN's.

The result of augmenting the dataset, compared with simply normalizing the images, was an increase from 94.3% validation accuracy to 99.1% validation accuracy.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        					| 
|:-----------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							|
| Pre-Processing   	| 32x32x1 Grayscale image   					|
| Convolution 		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU				|												|
| Max pooling	    | 2x2 stride,  outputs 14x14x6	 				|
| Convolution 		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU				|												|
| Max pooling	    | 2x2 stride,  outputs 5x5x16	 				|
| Flatten			| Input 5x5x16, outputs 400						|
| Fully connected	| Input 400, outputs 120						|
| RELU				| 	        									|
| Fully Connected	| Input 120, outputs 84							|
| RELU				| 	        									|
| Dropout			| 55% keep probability							|
| FC/Logits			| Input 84, outputs # classes					|
 
This is a pretty standard LeNet architecture. The biggest change within the atchitecture is choosing to include Dropout on the second fully connected layer. Dropout slightly improved the accuracy of the training model, but greatly improved the convergence and prevented the model from making large jumps in validation accuracy.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 25 Epochs and a batch size of 128. My initial training rounds used a batch size of 64, but the final augmented data set had many more data points and using a batch size of 64 had a negligible impact on performance while slowing down the training time. The training seemed to plateau around the 15th Epoch.

I used the Adam Optimizer instead of the traditional Gradient Descent Optimizer. Although the Adam Optimizer is more computationally intensive, it has the advantage of implementing momentum (moving averages of parameters), which allows the optimizer to converge faster with larger step sizes. I used a learning rate of 0.001. Any deviation from this learning rate seemed to hinder the  validation accuracy. I relied mostly on the Adam Optimizer to converge the parameters appropriately.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 99.1% 
* test set accuracy of 93.8%

My initial model was based on what I learned in the LeNet lab. As such, I stuck with the LeNet-5 architecture on my first run-through which provided approximately 91% validation set accuracy. The first architecture included 2x Convolution, 2x Max Pooling, and 3x Fully Connected layers. Normalizing and converting my training set to grayscale bumped up the validation accuracy to approximately 94%.

The next iteration was mostly spent tuning hyperparameters like Epoch Size, Batch Size, and learning rate. I learned that the number of Epochs past 25 had very little gain on validation accuracy, but reducing the batch size actually did slightly improve my accuracy by ~1%. Tuning the learning rate away from 0.001 only caused my model to diverge and become less accurate so I mostly left that alone. It seemed that I needed to take other steps to improve my model past 94%.

That is when I turned to dropout. For a while, I felt as if dropout wasn't doing anything to help with improving the validation accuracy. In some iterations, while tuning the keep_prob hyperparameter, it seemed as if lowering the keep_prob parameter caused my model more grief than improvement. That's when I decided to augment my dataset. After augmentation and dramatically increasing the size of the dataset is when dropout really started to shine. I jumped from a 94% validation accuracy to a ~98% validation accuracy. Re-implementing Dropout on the augmented dataset caused the validation accuracy to jump up another percentage point to 99.1%-99.2% It was at this point that I became satisfied with the model's training.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/Visualizations/DownloadedTrafficSigns.PNG) 

I chose these images because I felt like they were the most similar to what the dataset had. My biggest concern was the 3rd image: the no passing sign. The picture seemed to be taken at dusk and had a weird light artifact in the bottom corner of the image. Otherwise, I felt these images were very "well behaved" and should be easily classified by the model.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction		| 
|:---------------------:|:---------------------:| 
| Speed Limit (30 kph)	| Speed Limit (30 kph)	| 
| Right of Way 			| Right of Way			|
| No Passing			| No Passing			|
| Stop Sign	      		| Stop Sign				|
| Speed Limit (70 kph)	| Speed Limit (70 kph)	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I believe this is inline with the test accuracy of 93.8% simply because of how well-behaved the images I chose were. If a more challenging image was used in place of the stop sign, for example, the classifier accuracy could have dropped to 80%. In fact, I tried just that by introducing a "No Entry" sign (see below) that was positioned behind another sign in place of the stop sign and the classifier was not able to predict it correctly.

![](https://github.com/emiranda927/CarND-Traffic-Sign-Classifier/blob/master/no_entry.png)

The classifier though the entry sign was most likely a "Yield Sign", which is triangular in shape. Because the No-Entry sign is behind a triangular sign, this introduces an interesting problem for the classifier. Namely, how do we classify an image in this situation if it can fool the classifier into thinking it has a different shape? That is one limitation and difficulty I encountered while doing this excercise.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all the images, the model was completely confident that it chose the correct image (probability of 1.0).

The top 5 classes predicted by the classifier for each image (softmax probabilities) are:

| Image			        |     Softmax Probability (% Confidence)		|
|:----------------------|:---------------------------------------------:|
| No passing			| No passing (w/ 100.0% confidence)				|
| No Passing			| Speed limit (20km/h) (w/ 0.0% confidence)		|
| No passing			| Speed limit (30km/h) (w/ 0.0% confidence)		|
| No passing			| Speed limit (50km/h) (w/ 0.0% confidence)		|
| No passing			| Speed limit (60km/h) (w/ 0.0% confidence)		|
| Stop					| Stop (w/ 100.0% confidence)					|
| Stop					| Keep right (w/ 0.0% confidence)				|
| Stop					| Speed limit (120km/h) (w/ 0.0% confidence)	|
| Stop					| No vehicles (w/ 0.0% confidence)				|
| Stop					| Yield (w/ 0.0% confidence)					|
| Speed limit (70km/h)	| Speed limit (70km/h) (w/ 100.0% confidence)	|
| Speed limit (70km/h)	| Speed limit (30km/h) (w/ 0.0% confidence)		|
| Speed limit (70km/h)	| Speed limit (80km/h) (w/ 0.0% confidence)		|
| Speed limit (70km/h)	| Stop (w/ 0.0% confidence)						|
| Speed limit (70km/h)	| Speed limit (50km/h) (w/ 0.0% confidence)		|
| Right-of-way			| Right-of-way (w/ 100.0% confidence)			|
| Right-of-way			| Beware of ice/snow (w/ 0.0% confidence)		|
| Right-of-way			| Pedestrians (w/ 0.0% confidence)				|
| Right-of-way			| Dangerous curve-right (w/ 0.0% confidence)	|
| Right-of-way			| Turn right ahead (w/ 0.0% confidence)			|
| Speed limit (30km/h)	| Speed limit (30km/h) (w/ 100.0% confidence)	|
| Speed limit (30km/h)	| Speed limit (120km/h) (w/ 0.0% confidence)	|
| Speed limit (30km/h)	| Roundabout mandatory (w/ 0.0% confidence)		|
| Speed limit (30km/h)	| Speed limit (60km/h) (w/ 0.0% confidence)		|
| Speed limit (30km/h)	| Speed limit (50km/h) (w/ 0.0% confidence)		|

I should emphasize here that this was not the case the entire time. As I iterated through improving my model, I started with a 20% classification accuracy, moved up to 40%, then 60%, than 80% and finally 100% accuracy. The softmax probabilties were also not always 100% confidence. For example, the classifier often classified he 70 km/h traffic sign as a 30 km/h traffic sign with 75% confidence. The second softmax probability for this case was the correct prediction of a 70 km/h traffic sign with ~24% probability.

Augmenting the dataset to the extent that I did is what ultimately pushed the performance of these additional images to the 100% mark.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
