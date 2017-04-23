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

![](..//Visualizations/Sign_CID_Visualization.png) 
![](..//Visualizations/Original_DataSet_Vis.png) 

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
|						|												|

When training, I envisioned that the CNN would benefit from having more generalized images to train on. Denoising is a common operation used in image preprocessing. In order to facilitate this, I applied a median filter (ndimage.median_filter) to smooth the edges of the training set images and blur them slightly. I chose a median filter over a gaussian blur because the median filter preserves the edges of the image better than a guassian filter. I added these filtered images directly onto the original dataset as an extension of what we already had. The one conern I have with employing the denoise operation are the computational resources required with increasing dataset sizes

The next problem I wanted to tackle was the issue of training example class disparity. In order to rectify this, I applied random "micro-rotations" to all images in a class until each Class-ID had 5000 images in the training set. The reason I used micro-rotations (as opposed to vertical/horizontal flipping) was to preserve the integrity of every training example. I could guarantee the invariance of each image while avoiding the introduction of redundant examples into the training set. This technique can be applied to any image used in training a convolutional neural network without having to split off select training id's that only work with specific geometric transformations.

The results of employing the micro-rotation operations was in increase in the dataset from 27,839 examples to 215,000 training examples. 

The last two steps were to transfrom the augmented dataset to grayscale and normalize to mean 0 and unit standard deviation. The best reason to transfrom the dataset into grayscale is to reduce the computational load (going from 3 channels to 1) without a noticeable loss in performance. It's been shown that color has a negligible impact on training CNN's.

The result of augmenting the dataset, compared with simply normalizing the images, was an increase from 94.3% validation accuracy to 99.1% validation accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        					| 
|:-----------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							| 
| Convolution 3x3   | 1x1 stride, same padding, outputs 32x32x64 	|
| RELU				|												|
| Max pooling	    | 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	| etc.      									|
| Fully connected	| etc.        									|
| Softmax			| etc.        									|
|					|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
