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
                if mode=="origin":
                    os.system("python3 detect.py --root_dir=../../{0} --mode={1} --img_size=608 --output_dir=output_{1}/".format(l,mode))
                else:
                    os.system("python3 detect.py --root_dir=../../{0} --mode={1} --output_dir=output_{1}/".format(l,mode))
                os.chdir("../src")
            except Exception as e:
                print(e)
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
            if l[1]>50:
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
            if l[1]>50:
                f.write(l[0]+"\n")

def read_infere_time(mode,filename):
    total_time = 0.0
    with open("../PyTorch-YOLOv3/output_{}/output/{}/infer_time.txt".format(mode,filename)) as f:
        for line in f:
            line = line.strip()   
            if line[0] == "T":
                total_time = (line.split()[2])
            #if line[0] == "A":
            #    self.averge_time = line.split()[2]
    return total_time
def read_acc_time(mode,filename):
    acc = 0.0
    with open("../PyTorch-YOLOv3/output_{}/output/{}/acc.txt".format(mode,filename)) as f:
        acc = float(f.readline())
    return acc
def compare_acc_time(mode1,mode2,filename,threshold):
    total_time1 = 0.0
    total_time2 = 0.0
    acc1 = 0
    acc2 = 0
    
    total_time1 = read_infere_time(mode1,filename[21:])
    total_time2 = read_infere_time(mode2,filename[21:])
    acc1 = read_acc_time(mode1,filename[21:])
    acc2 = read_acc_time(mode2,filename[21:])
    print(" total time 1 is {}".format(total_time1))
    print(" total time 2 is {}".format(total_time2))
    print(" acc 1 is {}".format(acc1))
    print(" acc 2 is {}".format(acc2))
        
    
#find_lower_frame()
run_code("origin",30,not_moving_videos)
run_code("cropped",30,not_moving_videos)
run_code("psnr",30,not_moving_videos)
run_code("net",30,not_moving_videos)
#compare_acc_time("origin","cropped","image/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00051019",0)