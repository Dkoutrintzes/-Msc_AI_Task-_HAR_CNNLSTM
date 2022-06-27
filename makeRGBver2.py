import numpy as np
import os.path as path
import os
#from scipy.misc import imsave
from scipy import interpolate
import imageio
import time
Dataset_Folder = 'F:\Work\Action recognition NTU\Datasets\Skeletons_Data_Norm_zero'
PathResActionsSeq= 'F:\Work\Action recognition NTU\Datasets\Skeletons_Data_Norm_zero'
PathResActionsSeqImages= 'F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver'

def scaleback(data):
    xaction,yaction,channels=np.shape(data)
    newdata = np.zeros((25,199,3), dtype=np.uint8)
    step = yaction/199
    
    for i in range(25):
        a = 0
        for y in range(199):
            print(a,y)
            print(newdata[i][y],data[i][int(a)])
             
            newdata[i][y] = data[i][int(a)]
            a += step
    return newdata

def zero_pad(data):
    #xaction,yaction=np.shape(data)
    newdata = np.zeros((25,199,3), dtype=np.uint8)
    for i in range(len(data)):
        for y in range(len(data[0])):
            #print(newdata[i][y],data[i][y])
            newdata[i][y] = data[i][y]
    return newdata
    
    



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
            if xaction != 0:
                    if nmax<xaction:
                        nmax=xaction


                    
                    

                    #print (signal_shape)
                    #print (str(xaction)+'xaction')
                    shape1 = (xaction,25)

                    red_pos = np.ndarray(shape1)
                    red_pos = np.zeros(shape1)

                    green_pos = np.ndarray(shape1)
                    green_pos = np.zeros(shape1)

                    blue_pos = np.ndarray(shape1)
                    blue_pos = np.zeros(shape1)

                    for j in range(0,xaction):
                        y=0;
                        x=0;
                        while y<75:
                                red_pos[j,x] =  ActionFile[j][y]
                                green_pos[j,x]= ActionFile[j][y+1]
                                blue_pos[j,x]= ActionFile[j][y+2]
                                y=y+3
                                x=x+1

                    zr = red_pos
                    zg = green_pos
                    zb = blue_pos
                    br = []
                    bg = []
                    bb = []
                    TransposeRedPos=red_pos.T
                    TransposegreenPos=green_pos.T
                    TransposeBluePos=blue_pos.T
                    if xaction>=num:
                        dif=num
                    else:
                        dif=num-xaction
                    shape = (25,dif)
                    dr = np.zeros(shape)
                    dg= np.zeros(shape)
                    db = np.zeros(shape)
                    # #start of interpolation
                    # if 1:
                    #     for k in range(0,25):
                    #             arr2Red = np.array(TransposeRedPos[k])#each row
                    #             arr2Green = np.array(TransposegreenPos[k])
                    #             arr2Blue = np.array(TransposeBluePos[k])

                    #             arr2_interpRed = interpolate.interp1d(np.arange(arr2Red.size), arr2Red)
                    #             arr2_interpGreen = interpolate.interp1d(np.arange(arr2Green.size), arr2Green)
                    #             arr2_interpBlue = interpolate.interp1d(np.arange(arr2Blue.size), arr2Blue)


                    #             arr2_stretchRed = arr2_interpRed(np.linspace(0,arr2Red.size-1,num))
                    #             arr2_stretchGreen = arr2_interpGreen(np.linspace(0,arr2Green.size-1,num))
                    #             arr2_stretchBlue = arr2_interpBlue(np.linspace(0,arr2Blue.size-1,num))

                    #             br = np.concatenate((bfr, arr2_stretchRed), axis=0)
                    #             bg = np.concatenate((bg, arr2_stretchGreen), axis=0)
                    #             bb = np.concatenate((bb, arr2_stretchBlue), axis=0)
                    #     zr = np.reshape(br,(25,num))
                    #     zb = np.reshape(bg,(25,num))
                    #     zg = np.reshape(bb,(25,num))
                    #     zr=zr.T
                    #     zg=zg.T
                    #     zb=zb.T
		        #znew = z.T


                        #if signal_shape >= num:


                    #rgb size
                    rgb = np.zeros ((25,xaction-1, 3), dtype=np.uint8)
                    redcord = np.zeros((25,xaction-1,1), dtype=np.uint8)
                    greencord = np.zeros((25,xaction-1,1), dtype=np.uint8)
                    bluecord = np.zeros((25,xaction-1,1), dtype=np.uint8)
                    red = np.zeros((25,xaction-1,1))
                    green = np.zeros((25,xaction-1,1))
                    blue = np.zeros((25,xaction-1,1))
                    for i in range(0,xaction-1):
                            #for every frame
                            y=0;
                            x=0;

                            while y<75:

                                    #redcord=(((ActionFile[i,y]-oldmin)*newrange)/oldrange)+newmin
                                    '''        
                                    #if i is 0:
                                    #red
                                    redcord[x,0]=(((zr[i,x]-oldmin)*newrange)/oldrange)+newmin
                                    red[x,0]=zr[i,x]
                                   
                                    #green
                                    greencord[x,0]=(((zg[i,x]-oldmin)*newrange)/oldrange)+newmin
                                    green[x,0]=zg[i,x] 
                                    #blue
                                    bluecord[x,0]=(((zb[i,x]-oldmin)*newrange)/oldrange)+newmin
                                    blue[x,0]=zb[i,x]  
                                    
                                           
                                    #print 'frame 0 is '+'redcord= '+str(redcord[x,0])
                                    else:
                                            
                                    '''

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
                                    '''
                                    #min
                                    if red[x,i]<minv[0]:
                                        minv[0]=red[x,i]
                                    
                                    if green[x,i]<minv[1]:
                                        minv[1]=green[x,i]
                                    
                                    if blue[x,i]<minv[2]:
                                        minv[2]=blue[x,i]
                                    #max
                                    
                                    if red[x,i]>maxv[0]:
                                        maxv[0]=red[x,i]
                                    
                                    if green[x,i]>maxv[1]:
                                        maxv[1]=green[x,i]
                                    
                                    if blue[x,i]>maxv[2]:
                                        maxv[2]=blue[x,i]
                                    '''
                                    #time.sleep(2.0)
                                    y=y+3;
                                    x=x+1;
                            #("something")
                            #time.sleep(5.5)
                            #print("something")
                    print(type(rgb))
                    if xaction < 200:
                        rgb=zero_pad(rgb)
                    else:
                        rgb=scaleback(rgb)
                    print(np.shape(rgb))
                    imageio.imwrite(PathResActionsImages.replace('.csv','.png'),rgb.transpose(1, 0, 2));

                    #imsave(PathResActionsImages+'.png', rgb);
                    #for i in minv:
                    #        print (str(i)+' min')
                    #for i in maxv:
                    #        print (str(i)+' max')
                    #np.savetxt(PathResActionsImages+'rgb.csv', rgb, delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'red.csv', redcord.reshape(25,145), delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'green.csv', greencord.reshape(25,150), delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'blue.csv', bluecord.reshape(25,150), delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'redcol.csv', red.reshape(25,149), delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'greencol.csv', green.reshape(25,149), delimiter=',',fmt='%1.6f');
                    #np.savetxt(PathResActionsImages+'bluecol.csv', blue.reshape(25,149), delimiter=',',fmt='%1.6f');
        # except Exception as a:
        #     print("Error in reading file: ",name,a)

#print(nmax)
