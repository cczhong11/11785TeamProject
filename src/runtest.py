import os

not_moving_videos = "no_moving.txt"
error = "error.txt"
def find_good_data(cc=0):
    c = 0
    with open(error) as f:
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
                    with open("not_moving_filter.txt","a") as f2:
                        f2.write(name+"\n") 
            except:
                os.chdir("../../../../src")
                with open("errorx.txt","a") as f2:
                        f2.write(name+"\n") 
            print(c)
            c += 1
find_good_data(0)
