from argparse import Action
import numpy as np
import os.path as path
import os
#from scipy.misc import imsave
from scipy import interpolate
import imageio
import time


PathResActionsSeq= 'F:\Work\Action recognition NTU\Datasets\Skeletons_Data_Norm_zero'
PathResActionsSeqImages= 'F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P'

def scaleback(data,skels):
    xaction,yaction,channels=np.shape(data)
    newdata = np.zeros((skels,199,3), dtype=np.uint8)
    step = yaction/199
    
    for i in range(skels):
        a = 0
        for y in range(199):
            print(a,y)
            print(newdata[i][y],data[i][int(a)])
             
            newdata[i][y] = data[i][int(a)]
            a += step
    return newdata

def zero_pad(data,skels):
    #xaction,yaction=np.shape(data)
    newdata = np.zeros((skels,199,3), dtype=np.uint8)
    for i in range(len(data)):
        for y in range(len(data[0])):
            #print(newdata[i][y],data[i][y])
            newdata[i][y] = data[i][y]
    return newdata
    
    

    
            

        # except Exception as a:
        #     print("Error in reading file: ",name,a)


try:
    os.mkdir(PathResActionsSeqImages)
except:
    print('Folder exist')
files = os.listdir(PathResActionsSeq);
#files.sort(key=lambda f: int(filter(str.isdigit, f)))
minv=[0,0,0]
maxv=[0,0,0]
nmax=0
oldmax=0.1
oldmin=(-0.2);
newmax=255.0
newmin=0.0
num=200
oldrange=(oldmax-oldmin)
newrange=(newmax-newmin)  
for name in files:
        
        #if name=='2524.csv':
        # try:    
            PathResActions=path.join(PathResActionsSeq,name);
            PathResActionsImages=path.join(PathResActionsSeqImages,name);
            if os.path.isfile(PathResActionsImages.replace('.csv','.png')):
                print('Hi Mark')
                continue
            #open the action file
            ActionFile=np.genfromtxt(PathResActions, delimiter=',');

            try:
                 xaction,yaction=np.shape(ActionFile);
                 print(xaction,yaction)
            except:
                 xaction=1;
                 yaction=150;
            #red is x, green y , blue z
            skels = 25
            if len(ActionFile[0]) == 150:
                skels = 50

            if xaction != 0:
                    if nmax<xaction:
                        nmax=xaction


                    #print (signal_shape)
                    #print (str(xaction)+'xaction')
                    shape1 = (xaction,skels)

                    red_pos = np.ndarray(shape1)
                    red_pos = np.zeros(shape1)

                    green_pos = np.ndarray(shape1)
                    green_pos = np.zeros(shape1)

                    blue_pos = np.ndarray(shape1)
                    blue_pos = np.zeros(shape1)

                    for j in range(0,xaction):
                        y=0;
                        x=0;
                        while y<skels*3:
                                red_pos[j,x] =  ActionFile[j][y]
                                green_pos[j,x]= ActionFile[j][y+1]
                                blue_pos[j,x]= ActionFile[j][y+2]
                                y=y+3
                                x=x+1

                    zr = red_pos
                    zg = green_pos
                    zb = blue_pos
                    if xaction>=num:
                        dif=num
                    else:
                        dif=num-xaction
                    shape = (skels,dif)



                        #if signal_shape >= num:


                    #rgb size
                    rgb = np.zeros ((skels,xaction-1, 3), dtype=np.uint8)
                    redcord = np.zeros((skels,xaction-1,1), dtype=np.uint8)
                    greencord = np.zeros((skels,xaction-1,1), dtype=np.uint8)
                    bluecord = np.zeros((skels,xaction-1,1), dtype=np.uint8)
                    red = np.zeros((skels,xaction-1,1))
                    green = np.zeros((skels,xaction-1,1))
                    blue = np.zeros((skels,xaction-1,1))
                    for i in range(0,xaction-1):
                            #for every frame
                            y=0;
                            x=0;

                            while y<skels*3:

                                    #redcord=(((ActionFile[i,y]-oldmin)*newrange)/oldrange)+newmin
                         


                                    red[x,i]=(zr[i+1,x]-zr[i,x])
                                    green[x,i]=(zg[i+1,x]-zg[i,x])
                                    blue[x,i]=(zb[i+1,x]-zb[i,x])
                                    #red -0.27

                                    if(red[x,i]>0.1):
                                        red[x,i]=0.1
                                    if(green[x,i]>0.1):
                                        green[x,i]=0.1
                                    if(blue[x,i]>0.1):
                                        blue[x,i]=0.1


                                    if(red[x,i]<-0.2):
                                        red[x,i]=-0.2
                                    if(green[x,i]<-0.2):
                                        green[x,i]=-0.2
                                    if(blue[x,i]<-0.2):
                                        blue[x,i]=-0.2

                                    #redcord[x,i]=(((red[x,i]-oldmin)*newrange)/oldrange)+newmin
                                    redcord[x,i]=((red[x,i]-oldmin)/(oldmax-oldmin))*255.0

                                    #green
                                    #greencord[x,i]=(((green[x,i]-oldmin)*newrange)/oldrange)+newmin
                                    greencord[x,i]=((green[x,i]-oldmin)/(oldmax-oldmin))*255.0

                                    #blue
                                    #bluecord[x,i]=(((blue[x,i]-oldmin)*newrange)/oldrange)+newmin
                                    bluecord[x,i]=((blue[x,i]-oldmin)/(oldmax-oldmin))*255.0


                                    #print 'frame '+str(i)+' '+str(redcord[x,i])+'=redcord['+str(x)+','+str(i)+']='+str(ActionFile[i,y])+'-'+str(redcord[x,i-1]);


                                    rgb[x,i][0] =redcord[x,i]#red
                                    rgb[x,i][1] =greencord[x,i]#green
                                    rgb[x,i][2] =bluecord[x,i]#blue
                                    #time.sleep(2.0)
                                    y=y+3;
                                    x=x+1;
                            #("something")
                            #time.sleep(5.5)
                            #print("something")




                    if xaction < 200:
                        rgb=zero_pad(rgb,skels)
                    else:
                        rgb=scaleback(rgb,skels)

                                        
                    if skels == 25:
                        newrgb = np.zeros((50,199,3), dtype=np.uint8)
                        for fi in range(len(rgb)):
                            for fy in range(len(rgb[0])):
                                newrgb[fi][fy] = rgb[fi][fy]
                        rgb = newrgb

                    print(np.shape(rgb))
                    imageio.imwrite(PathResActionsImages.replace('.csv','.png'),rgb.transpose(1, 0, 2));
