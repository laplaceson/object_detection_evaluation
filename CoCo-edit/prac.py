import random
import os 
import numpy as np,h5py
import cocoeval
import coco
import simplejson as json
import io
# annoformat={"id","image_id","area","bbox"}

	
annotationD=[]
annotationT=[]
image=[]
#bigD={"info": [],"images": [],"annotations": annotationD,"licenses": 0}
bigT={"info": [],"images": [],"annotations": annotationT,"licenses": 0,"categories":[]}

def loadData():
	train='imgIdx.mat'
	f=h5py.File(train,'r')
	imgidx= f['imgIdx']
	name=imgidx['name']
	label=imgidx['label']
	anno=imgidx['anno']
	#----------------------
	Data={'name':[],'label':[],'anno':[]}
	for i in range(len(name)):
		#print (f[name[i,0]].value.tobytes()[::2])
		Data['name'].append(f[name[i,0]].value.tobytes()[::2])
		#print f[name[i,0]].value.tobytes()[::2]
		#image.append(f[name[i,0]].value.tobytes()[::2]) 
		Data['label'].append(f[label[i,0]].value)
		if(f[label[i,0]].value!=0):
			#print (list(zip(*f[anno[i,0]].value)))
			Data['anno'].append(map(list,zip(*f[anno[i,0]].value)))
		else:
			Data['anno'].append([])
	#print (list(map(list,Data['anno'])))
	return Data
#---------------------------------------------
def model(Data): #format Detection result
	AvgIOU=0
	DataSize=len(Data['anno'])
	countD=0
	countT=0
	#--------------------
	with open('ChangeName.json') as json_data:
		NAMEs = json.load(json_data)
    #-------------------
	for i in range(DataSize):#for every pic
		tmps=random.randint(0,10)
		if(tmps>0):
			for j in range(tmps):#rand num of box for every img
				rx1=random.randint(0,640)
				ry1=random.randint(0,480)
				w=random.randint(0,640-rx1)
				h=random.randint(0,480-ry1)
				#annotationD.append({"id":countD,"image_id":NAMEs["after"][i],"category_id":-1,"bbox":[rx1,ry1,w,h],"score":100.,"area":w*h})
				annotationD.append({"id":countD,"image_id":NAMEs["after"][i],"category_id":-1,"bbox":[rx1,ry1,w,h],"score":100.,"area":w*h})
				countD+=1
		else:
			annotationD.append({"id":countD,"image_id":NAMEs["after"][i],"category_id":-1,"bbox":[],"score":100.,"area":0.})
			countD+=1

		#print (list(map(list,Data['anno'][i])))
		tmp=list(map(list,Data['anno'][i]))##?????????????????????????
		if(len(tmp)):
			for k in tmp:
				annotationT.append({"id":countT+1,"image_id":NAMEs["after"][i],"category_id":-1,"bbox":[k[0],k[1],k[2]-k[0],k[3]-k[1]],"score":100.,"iscrowd":0,"area":100000})
				countT+=1
		else:
			annotationT.append({"id":countT+1,"image_id":NAMEs["after"][i],"category_id":-1,"bbox":[],"score":100.,"iscrowd":0,"area":0.})
			countT+=1
		#image.append({"id":i," file_name ": Data['name'][i]})# h w
# --------------------------------------------
# Data=loadData()
# model(Data)#for annotation
# #-------------------------------------------for categories
# with open('instances_val2017.json','r') as data_file:    
#     DTcatg = json.load(data_file)
# #bigT["categories"]=DTcatg["categories"]
# bigT['categories'].append({"supercategory":"person","id":-1,"name":"person"})
# # #-------------------------------------------
# for dirPath, dirNames, fileNames in os.walk("./img/"):
# 	print ("Loading Image Data...")
# for i in range(len(fileNames)):
# 	image.append({"id":int(fileNames[i]),"file_name":fileNames[i]+".jpg"})
# bigT["images"]=image
# #---------------------------------------------------
# # Write JSON file
# with open("DataGT.json", "w") as outfile:
#     json.dump(bigT, outfile, indent=4)
# with open("DataDT.json", "w") as outfile:
#     json.dump(annotationD, outfile, indent=4)

#-------------------------------
selfT=coco.COCO("DataGT.json")
selfD=selfT.loadRes("DataDT.json")
E=cocoeval.COCOeval(selfT,selfD,'bbox')
E._prepare();
E.evaluate();
E.accumulate();
E.summarize();

