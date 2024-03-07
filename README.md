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
