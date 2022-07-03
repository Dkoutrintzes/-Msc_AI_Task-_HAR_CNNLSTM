# CNN-LSTM Architecture for Human Action Recognition Using Skeletal Representation

# Create images from Skeletal data

The codes makeRGB.py and makeRGBver2P.py are used to create images out from skeletal data.
To run:
```
python makeRGB(/ver2P).py <Skeletal Dataset Path> <Save Path>
```

# CNN-LSTM model

The CNNLSTM_model.py contains the arcitectures of the the models, including the default CNN model.
To call the models
```
create_LSTM_CNN(input_shape,number_of_actions)
create_LSTM_CNN_2P(input_shape,number_of_actions)
create_MOS_CNN(input_shape,number_of_actions)
```
# Traing

The files Cnn.py, Generator.py, Model.py contain the functions used for training.

The file Cnn.py contains the main function.
Model.py contains a class model that we use for training process.
Generator.py contains a datagenerator class.

# Evaluation

The file findbest.py is used to evaluate every model saved from every epoch while training.
The Record performance can return the accuracy, precision, recall and f1 scores from the best model that the findbest returned 

