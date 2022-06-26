from BezierLocal import Bezier
import matplotlib.pyplot as plt
import numpy as np
import csv
from math import sqrt
data = []
with open('Action_041_L_001_001_001.csv','r') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    data.append(line)
print(np.shape(data))
finaldata = np.zeros([len(data),75])

for joint in range(25):
  x = []
  y = []
  z = []
  tempx = []
  tempy = []
  points = []
  counter = 0
  newdata = []
  for i in range(len(data)):
    x.append(float(data[i][joint*3]))
    y.append(float(data[i][(joint *3) + 1]))
    z.append(float(data[i][(joint *3) + 2]))

    tempx.append(float(data[i][joint*3]))
    tempy.append(float(data[i][(joint *3) + 1]))

    points.append([float(data[i][joint*3]),float(data[i][(joint *3) + 1])])
    counter += 1
    if counter == 20:
      counter = 0 
      t_points = np.arange(0, 1, 0.01) 
      points = np.array(points)
      curve = Bezier.Curve(t_points, points)

      for i in range(len(tempx)):
        xl = tempx[i]
        yl = tempy[i]

        dist = 100000000000
    

        for p in curve:
          tempdist = sqrt((xl - p[0])**2 + (yl - p[1])**2)
          
          if dist > tempdist:
            dist = tempdist
            xc = p[0]
            yc = p[1]
        
        newdata.append([xc,yc])

      tempx = []
      tempy = []
      points = []

      points2 = []
      for i in range(len(x)):
        points2.append([x[i],y[i]])
      points2 = np.array(points2)     

      plt.figure()
      plt.plot(
        points2[:, 0],  # x-coordinates.
        points2[:, 1],  # y-coordinates.
        'r'           # Styling (red, circles, dotted).
      )
      dd = np.array(newdata)
      plt.plot(
        dd[:, 0],  # x-coordinates.
        dd[:, 1],  # y-coordinates.
        'g:'           # Styling (red, circles, dotted).
      )

      plt.grid()
      plt.show()
  for l in range(len(newdata)):
    finaldata[l][joint*3] = newdata[l][0] 
    finaldata[l][joint*3 + 1] = newdata[l][1] 
    finaldata[l][joint*3 + 2] = z[l] 


with open('Action_041_L_001_001_001_S.csv','w',newline = '') as csvwrite:
  writer = csv.writer(csvwrite)
  for line in finaldata:
    writer.writerow(line)
