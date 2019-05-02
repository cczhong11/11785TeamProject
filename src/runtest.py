import os

not_moving_videos = "not_moving_filter.txt"
moving_videos = "moving_filter.txt"
error = "error.txt"
def find_good_data(cc=0):
    c = 0
    with open(moving_videos) as f:
        for l in f:
            if c<cc:
                c+=1
                continue
            l = l.strip()
            name = l
            os.chdir("../PyTorch-YOLOv3")
            try:
                os.system("python3 detect.py --root_dir=../../{} --mode=origin".format(l))
                #print(l[21:])
                os.chdir("output/"+l[21:])
                with open("acc.txt") as f1:
                    ll = f1.readline().strip()
                    ll = float(ll)
                    if ll < 0.4:
                        name = ""
                os.chdir("../../../../src")
                if name!="":
                    with open("moving_filter.txt","a") as f2:
                        f2.write(name+"\n") 
            except:
                os.chdir("../../../../src")
                with open("errorx.txt","a") as f2:
                        f2.write(name+"\n") 
            print(c)
            c += 1

def run_code(mode,threshold,filename):
    c = 0
    with open(filename) as f:
        for l in f:
            if c>threshold:
                break
            l = l.strip()
            name = l
            os.chdir("../PyTorch-YOLOv3")
            try:
                os.system("python3 detect.py --root_dir=../../{0} --mode={1} --img_size=608 --output_dir=output_{1}".format(l,mode))
                os.chdir("../src")
            except:
                os.chdir("../src")
                with open("errorx.txt","a") as f2:
                        f2.write(name+"\n") 
            print(c)
            c += 1

def find_lower_frame():
    rs = {}
    with open(not_moving_videos) as f:
        for l in f:
            l = l.strip()
            ll = os.listdir("../../{0}".format(l))
            rs[l] = len(ll)
    rs = sorted(rs.items(),key=lambda x:x[1])
    with open(not_moving_videos,'w') as f:
        for l in rs:
            if l[1]<230:
                f.write(l[0]+"\n")
    rs = {}
    with open(moving_videos) as f:
        for l in f:
            l = l.strip()
            ll = os.listdir("../../{0}".format(l))
            rs[l] = len(ll)
    rs = sorted(rs.items(),key=lambda x:x[1])
    with open(moving_videos,'w') as f:
        for l in rs:
            if l[1]<230:
                f.write(l[0]+"\n")

def compare_acc_time(mode1,mode2,filename,threshold):
    pass

find_lower_frame()
#run_code("origin",20,not_moving_videos)