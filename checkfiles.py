import os
import csv
datapath = 'F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver'
newfolder = 'ZPLabels'

try:
    os.mkdir(newfolder)
except:
    pass


def check(name):
    data = []
    counter = 0
    with open(os.path.join('NewLabels',name),'r') as file:
        reader = csv.reader(file)
        for line in reader:
            #print(line)
            if os.path.isfile(os.path.join(datapath,line[0])):
                data.append(line)
            else:
                #print(line)
                counter += 1

    print(str(counter) + ' Files found missing')
    

    with open(os.path.join(newfolder,name),'w',newline='') as newfile:
        writer = csv.writer(newfile)
        for line in data:
            writer.writerow(line)


check('CS.csv')
check('CST.csv')
