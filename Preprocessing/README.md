# Preprocessing the CSV files that contain the Skeletal information

## 1.Data Normalization
Data Normalization is a simply code that moves the start of the movement on the center of the image. 
This can bring benefits to methods that use the locations of skeleton joint on each frame. 

```

>> python DataNormilization.py <DataPath> <NewDatapath>

```

## 2.Checks
For missing data and corect if any data point between two persons is on wrong body. For both cases the NTU RGB+D dataset dosent have any problematic 

```

>> python CheckAndRepair.py <DataPath> <NewDatapath>

```

## 3.Smoothing 
An Experiment to smooth the skeleton joints, that tremble between frames while the person is not moving,using the bezier curve. There is 3 gifs examples. 
First the Before is the default skeletons, after is simply apling the Bezier curve and after2 is when i tried to represent time compine in x dimantion.
