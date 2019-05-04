import os
import datetime
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
                    os.system("python3 detect.py --root_dir=../../{0} --mode={1} --img_size=1024 --output_dir=output_{1}/".format(l,mode))
                elif mode=="lk":
                    os.system("python3 detect.py --root_dir=../../{0} --mode={1} --img_size=1024 --output_dir=output_{1}/ --use_cuda=0".format(l,mode))
                elif mode=="origin_cpu":
                    os.system("python3 detect.py --root_dir=../../{0} --mode=origin --img_size=608 --output_dir=output_{1}/ --use_cuda=0".format(l,mode))
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
    return datetime.datetime.strptime(total_time,"%H:%M:%S.%f").timestamp()-datetime.datetime(1900,1,1,0,0,0).timestamp()
def read_acc_time(mode,filename):
    acc = 0.0
    with open("../PyTorch-YOLOv3/output_{}/output/{}/acc.txt".format(mode,filename)) as f:
        acc = float(f.readline())
    return acc
def compare_acc_time(mode1,mode2,filename):
    total_time1 = 0.0
    total_time2 = 0.0
    acc1 = 0
    acc2 = 0
    
    total_time1 = read_infere_time(mode1,filename[21:])
    total_time2 = read_infere_time(mode2,filename[21:])
    
    acc1 = read_acc_time(mode1,filename[21:])
    acc2 = read_acc_time(mode2,filename[21:])
    rs = 0.0
    if acc2!=0:
        rs = float(acc1)/float(acc2)
    
    return total_time1/total_time2,rs
        
def compare_mode(mode1,mode2,filename, threshold):
    c = 0
    ttsp = 0.0
    taccsp = 0.0

    with open(filename) as f:
        for l in f:
            if c>threshold:
                break
            l = l.strip()
            tsp,accsp = compare_acc_time(mode1,mode2,l)
            ttsp+=tsp
            taccsp+=accsp
            c+=1
    print("mode 1:{} mode2:{} time speed up:{} acc improve: {}".format(mode1,mode2,ttsp/c,taccsp/c))
def run_baseline(mode):
    run_code(mode,30,not_moving_videos)
    run_code(mode,10,moving_videos)
def run_baseline_cpu(mode):
    run_code(mode,10,not_moving_videos)
    #run_code(mode,10,moving_videos)
#find_lower_frame()

#compare_acc_time("origin","cropped","image/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00051019",0)
#compare_mode("origin","cropped",not_moving_videos,30)
#compare_mode("origin","psnr",not_moving_videos,30)
#compare_mode("origin","net",not_moving_videos,30)
#run_baseline("cropped")
#run_baseline("psnr")
#run_baseline("net")

run_baseline_cpu("origin_cpu")
run_baseline_cpu("lk")