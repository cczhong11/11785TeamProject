# report for one video
# total time
# average time
# interefence object frames / total frames
# average similarity between 1 / 5 / 20 / 50
import os
import datetime

PATH = "/Users/tczhong/Downloads/punch"


def get_overlap(ponit1, point2):
    x1,y1,x2,y2 = ponit1
    x3,y3,x4,y4 = point2
    dx = min(x2,x4) - max(x1,x3)
    dy = min(y2,y4) - max(y1,y3)
    if dx < 0 or dy < 0:
        return 0
    else:
        return dx * dy / ((x2-x1)*(y2-y1))

def get_simiarity(arr1, arr2):
    avg = 0
    cnt = 0
    for point1 in arr1:
        for point2 in arr2:
            space = get_overlap(point1, point2)
            
            if space > 0:
                avg += space
                cnt += 1
    if cnt == 0:
        return 0
    return avg / cnt

class Statistics(object):
    def __init__(self, rootpath,video=None):
        self.rootpath = rootpath
        self.video = video
        if video!=None:
            self.path = rootpath+"/"+video
        self.filenames = {} # frames: filename
        self.objects = {} # frames:{name:point}
        self.objects_count = {} # frame:obj_count

    def reset(self):
        self.filenames = {} # frames: filename
        self.objects = {} # frames:{name:point}
        self.objects_count = {} # frame:obj_count
    def read_timefile(self):
        with open(self.path+"/infer_time.txt") as f:
            for line in f:
                line = line.strip()
                
                if line[0] == "T":
                    self.total_time = line.split()[2]
                if line[0] == "A":
                    self.averge_time = line.split()[2]

    # read txt file and save data
    def read_txt(self,frame):
        with open(self.path + "/"+self.filenames[frame]) as f:
            line = f.readline()
            line = line.strip()
            objs = line.split(";")
            rs = {}
            count = len(objs)
            
            for obj in objs:
                if len(obj)==0:
                    continue
                name = obj.split(":")[0]
                value = obj.split(":")[1]
                points = [float(x) for x in value.split(",")]
                if name not in rs:
                    rs[name] = []
                rs[name].append(points)
        self.objects[frame] = rs
        self.objects_count[frame] = count    

    # read folder structure
    def read_folder(self):
        pngcnt = 0
        self.read_timefile()
        for file in os.listdir(self.path):
            if file.endswith(".txt"):
                frame = file.split(".")
                if frame[0] != "infer_time":
                    self.filenames[int(frame[0])] = file
            if file.endswith(".png"):
                pngcnt += 1
        self.pngcnt = pngcnt
        for f in self.filenames:
            self.read_txt(f)
    
    
    
    def find_simiarity(self, time1, time2):
        if time1 not in self.objects or time2 not in self.objects:
            return -1
        obj1 = self.objects[time1]
        obj2 = self.objects[time2]
        avg = 0
        cnt = 0
        for name in obj1:
            arr1 = obj1[name]
            if name in obj2:
                arr2 = obj2[name]
                score = get_simiarity(arr1, arr2)
                if score > 0:
                    avg += score
                    cnt += 1
        if cnt == 0:
            return 0
        return avg / cnt

    def get_avg_stats(self,n):
        s = 0
        cnt = 0
        for k in self.objects:
            if k+n in self.objects:
                s += self.find_simiarity(k,k+n)
                cnt+=1
        if cnt == 0:
            return 0
        return s/cnt

    def set_path(self,video):
        self.video = video
        self.path = self.rootpath+"/"+video

    def get_stats(self):
        f = open(self.path+"/report.csv",'w')
        avg1 = self.get_avg_stats(1)
        avg5 = self.get_avg_stats(5)
        avg10 = self.get_avg_stats(10)
        avg20 = self.get_avg_stats(20)
        f.write("{},{},{},{}\n".format(avg1,avg5,avg10,avg20))
        f.write("{},{}\n".format(self.total_time,self.averge_time))
        f.write("{},{},{}\n".format(len(self.filenames),self.pngcnt,len(self.filenames)/self.pngcnt))
        f.close()

    def average_all(self):
        rs = [0 for i in range(7)]
        cnt = 0
        for l in os.listdir(self.rootpath):
            if l.startswith("."):
                continue
            self.path = self.rootpath + "/" + l
            with open(self.path+"/report.csv") as f:
                line1 = f.readline().strip()
                line2 = f.readline().strip()
                line3 = f.readline().strip()
                
                a1,a5,a10,a20 = line1.split(",")
                
                ttime,atime = line2.split(",")
                _,_,ratio = line3.split(",")
                rs[0] += float(a1)
                rs[1] += float(a5)
                rs[2] += float(a10)
                rs[3] += float(a20)
                rs[4] += datetime.datetime.strptime(ttime,"%H:%M:%S.%f").timestamp()-datetime.datetime(1900,1,1,0,0,0).timestamp()
                rs[5] +=  datetime.datetime.strptime(atime,"%H:%M:%S.%f").timestamp()-datetime.datetime(1900,1,1,0,0,0).timestamp()
                rs[6] += float(ratio)
            cnt += 1
        rs = [i/cnt for i in rs]

        with open(self.rootpath+"/report.csv",'w') as f:
            f.write("{},{},{},{}\n".format(rs[0],rs[1],rs[2],rs[3]))
            f.write("{},{}\n".format(rs[4],rs[5]))
            f.write("{}\n".format(rs[6]))

def get_single_report_csv():
    s = Statistics(PATH)
    for l in os.listdir(PATH):
        s.set_path(l)
        s.reset()
        s.read_folder()
        s.get_stats()

def get_all_report_csv():
    s = Statistics(PATH)
    s.average_all()

get_single_report_csv()
get_all_report_csv()