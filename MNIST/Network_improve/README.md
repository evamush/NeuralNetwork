# L42-Starter-Pack

This repository contains the starter pack for the neural networks practical of the [Machine Learning and Algorithms for Data Mining](http://www.cl.cam.ac.uk/teaching/1617/L42/) (L42) module of Part III of the Computer Science Tripos and the MPhil in Advanced Computer Science at the [University of Cambridge](http://www.cam.ac.uk).

The purpose of this practical exercise is leveraging the [Keras](http://keras.io) framework to build, prototype and deploy deep neural networks on the [notMNIST](http://yaroslavvb.blogspot.co.uk/2011/09/notmnist-dataset.html) dataset (classifying 28x28 grayscale glyphs into ten classes (A-J)) in order to maximise accuracy.

The setup has been adapted from my collaboration with [Nenad Bauk](https://github.com/fifiman) for the [mgcsweek](http://www.csnedelja.mg.edu.rs) seminar at the [High School of Mathematics](http://www.mg.edu.rs) in Belgrade.

## Dependencies

The two primary dependencies are:

- [TensorFlow](https://www.tensorflow.org)
- [Keras](http://keras.io)

Their installation should take care of any further dependencies needed as well.

## Contents

Aside from the dataset (which is provided within the repository), three Python scripts are provided:

- `data.py` - Loads and preprocesses the dataset: normalises the data to [0, 1] range and performs a training/validation split;
- `model.py` - Contains the Keras specification of the neural network;
- `main.py` - Combines the two: loads the dataset, fetches the model, and trains it with the specified parameters (batch size, number of epochs, SGD optimizer).

The primary files to experiment with are `model.py` (for implementing a better-performing model) and `main.py` (for modifying the three training schedule hyperparameters.

## Usage

To begin, extract the dataset:

    $ tar xzvf notMNIST_small.tar.gz

Afterwards, launching the training subroutine is simply performed by running the `main` script:

    $ python main.py
    
Before starting the training, the summary of the deployed neural network (along with its trainable parameter count) will be pretty-printed for convenience.

## License

MIT
