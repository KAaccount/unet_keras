from datetime import datetime
import os
import json
import glob
import pathlib

class DataProc():
       
    def __init__(self):
        RES_DIR= "./result/"
        
        jobid=datetime.now().strftime("%Y/%m/%d%H%M%S").replace("/","")
        os.mkdir(RES_DIR+jobid)
        self.respath=RES_DIR+jobid
        print("### ",jobid)
        pass

    def generate_voc(self):

        DATA_DIR = '/work/DEVELOP/dataset/VOCdevkit/VOC2007/'

        segfilelist=os.listdir(DATA_DIR+"/SegmentationClass")
        imgfilelist=os.listdir(DATA_DIR+"/JPEGImages")

        tgtimgfiles=[]
        tgtsegfiles=[]
        for  segfile in segfilelist:
            tmpfile=segfile.replace(".png",".jpg")
            if tmpfile  in  imgfilelist:
                print(tmpfile,segfile)
                tgtimgfiles.append(DATA_DIR+"/JPEGImages/"+tmpfile)
                tgtsegfiles.append(DATA_DIR+"/SegmentationClass/"+segfile)
            else:
                continue

        allnum=len(tgtsegfiles)
        trainnum=int(allnum*0.7)
        #valnum=int(allnum*0.2)
        valnum=128
        testnum=allnum-trainnum-valnum

        train_imgfps=tgtimgfiles[0:trainnum]
        train_mskfps=tgtsegfiles[0:trainnum]
        valid_imgfps=tgtimgfiles[trainnum:trainnum+valnum]
        valid_mskfps=tgtsegfiles[trainnum:trainnum+valnum]
        test_imgfps=tgtimgfiles[trainnum+valnum:]
        test_mskfps=tgtsegfiles[trainnum+valnum:]

        print(allnum)
        print(len(train_imgfps),len(valid_imgfps),len(test_imgfps))
        print(len(train_mskfps),len(valid_mskfps),len(test_mskfps))
        print(test_imgfps)

        #json.dump(train_imgfps, respath+"/train_imgfps" )
        json.dump(train_imgfps, open(self.respath+"/train_imgfps.json","w") )
        json.dump(train_mskfps, open(self.respath+"/train_mskfps.json","w") )
        json.dump(valid_imgfps, open(self.respath+"/valid_imgfps.json","w") )
        json.dump(valid_mskfps, open(self.respath+"/valid_mskfps.json","w") )
        json.dump(test_imgfps,  open(self.respath+"/test_imgfps.json","w") )
        json.dump(test_mskfps,  open(self.respath+"/test_mskfps.json","w") )


    def generate_cvd(self):
        DATA_DIR = '/work/DEVELOP/dataset/camvid/SegNet-Tutorial/CamVid/'

        train_imgfps=sorted(glob.glob(DATA_DIR+"train"+"/*"))
        train_mskfps=sorted(glob.glob(DATA_DIR+"trainannot"+"/*"))
        valid_imgfps=sorted(glob.glob(DATA_DIR+"val"+"/*"))
        valid_mskfps=sorted(glob.glob(DATA_DIR+"valannot"+"/*"))
        test_imgfps =sorted(glob.glob(DATA_DIR+"test"+"/*"))
        test_mskfps =sorted(glob.glob(DATA_DIR+"testannot"+"/*"))

        json.dump(train_imgfps, open(self.respath+"/train_imgfps.json","w") )
        json.dump(train_mskfps, open(self.respath+"/train_mskfps.json","w") )
        json.dump(valid_imgfps, open(self.respath+"/valid_imgfps.json","w") )
        json.dump(valid_mskfps, open(self.respath+"/valid_mskfps.json","w") )
        json.dump(test_imgfps,  open(self.respath+"/test_imgfps.json","w") )
        json.dump(test_mskfps,  open(self.respath+"/test_mskfps.json","w") )



if __name__=="__main__":
    
    dataproc=DataProc()
    #dataproc.generate_voc()
    dataproc.generate_cvd()
