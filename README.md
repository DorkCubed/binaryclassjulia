# Introduction
In this script, I have used the flux.jl library to train a neural network to determine the difference between signal “s” and background “b” points. The script involves processing of the csv to form a dataframe, training the model, and checking its accuracy.
***
### Getting the data
The provided data had an unequal representation of signal and background points, and I wanted to avoid the model being biased. Furthermore, I wanted the model to accommodate a wide range of signal to background ratios. This rendered the inbuilt data loaders suboptimal - as even after shuffling, the model would end up biased.

Thus, the data preprocessing involved several steps - 

* Dividing the original dataframe in two - one for “s” type points (achieved by the `sdf` function) and one for “b” type points (achieved via the `bdf` function)
* Turning the dataframes into matrices. This is done with the `x_train` function. The function also has two additional features - 
    * Adding a 4th element - the distance from the center of the normalized x, y, z coordinates. This both helped with accuracy of the model and visualization of the data.
    * Splitting the dataframe from the `a`th element to the `sz`th element.
* Getting a random batch of coordinates from the vector and appending the answers to it. The `giverand` function achieves this. 
    * At this point, there’s no need to read the class of the coordinates from the dataframe (which is very slow), as they have already been split according to it. The class is appended to the random matrix based on the `sorb` input to the `giverand` function.
* Two matrices - one of "s" type, which has a random number of elements between 200 and 300, and another of "b" type, also with a random size, are joined and then shuffled. This is done within the training function itself.

The data is now ready to train the model with!
***
### The model
#### Architecture
The architecture of the model is quite simple, consisting of 5 layers, an increasing dropout rate and normalization across the leaky ReLU layers. I will provide more details below, but I must confess that most of it is intuition and experimentation rather than the theoretical “best”.

* Layer 1 - The first layer has 4 inputs and 16 outputs. I avoided giving too high of an output number to avoid overfitting and ease the computational workload. The layer uses the `leakyReLU` activation function to avoid the problems ReLU might have with a non-normalized database. The layer is then normalized using `batchnorm`. This is followed by a `dropout` rate of 0.1 for overfitting protection.
* Layer 2 - This layer has 16 inputs and 32 outputs. This layer was added when 16 layers failed to provide me with a satisfactory loss. This layer uses `tanh` as it both normalizes the data and seems to provide the best results so far. This layer has a `dropout` rate of 0.2.
* Layer 3 - This layer has 32 inputs and 12 outputs. This layer follows the `tanh` with a `leakyReLU` function, providing the `leakyReLU` with balanced inputs and greatly reduces the influence of the negative outputs of the `tanh` layer. This is followed by normalization using `batchnorm`.
* Layer 4 - This layer has 12 inputs and 3 outputs - This layer both exists for a gradual reduction in parameters and as the first `sigmoid` function. It outputs a value which is then normalized using `batchnorm`.
* Layer 5 - The final layer goes from 3 inputs to a single output. This layer has a `sigmoid` function, the output of which can be compared with the output of the binary classification task. 
#### Training
The training is done via the `train!` function of the flux library. The `binarycrossentropy` function is the go-to loss function for a binary classification task, and seemed to provide me with the best results. The only bit of uniqueness in this part comes from my decision to train the model twice using two different optimization algorithms.

First, the model is trained via `RMSProp` at a high learning rate of 5e-3. This is both to avoid a bias towards 0 that `Adam` might create and to create a smoother learning curve.

Then, the model is trained via `Adam` at a slower learning rate of 1e-4. This now helps the model to possibly settle at a better solution and makes convergence faster.

So far, this seems to allow the model to achieve better performance than doubling the number of epochs with one algorithm.

The model is now ready to test!
***
### Evaluation