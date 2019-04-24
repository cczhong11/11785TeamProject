import xml.etree.ElementTree as ET
import os
def find_in_folder(folder):
    for filename in os.listdir(folder):
        
        if filename[-3:]=="xml":
            tree = ET.parse(folder+filename)
            root = tree.getroot()
            k = 0
            for child in root:
                if child.tag == 'object':
                    k+=1
                    
                    
            print("{},{}".format(filename,k))
find_in_folder("/home/andrewdeeplearningisawesome/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00120000/")