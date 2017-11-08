#-*- coding: utf-8 -*-
import os 
import numpy as np,h5py
import simplejson as json
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
#--------------------------------------------------------------------------------------------------
#calculate IOU between box and box
def calcIOU(BBox_x1, BBox_y1, BBox_x2, BBox_y2, BBox_gt_x1, BBox_gt_y1, BBox_gt_x2, BBox_gt_y2):
	x1=BBox_x1;
	y1=BBox_y1;
	width1=BBox_x2-BBox_x1;
	height1=BBox_y2-BBox_y1;

	x2=BBox_gt_x1;
	y2=BBox_gt_y1;
	width2=BBox_gt_x2-BBox_gt_x1;
	height2=BBox_gt_y2-BBox_gt_y1;

	endx=max(x1+width1,x2+width2);
	startx=min(x1,x2);
	width=width1+width2-(endx-startx);

	endy=max(y1+height1,y2+height2);
	starty=min(y1,y2);
	height=height1+height2-(endy-starty);
	if (width<=0 or height<=0):
		ratio=0;
	else:
		Area=width*height;
		AreaUnit=width1*height1+width2*height2-Area;
		ratio=Area/AreaUnit;
	return ratio 
#------------------------------------------------------------------------------------------------------
#return ious between Gt and DT (maybe mutiple DTbox or GTbox)
def modelrandblk(arrayG,arrayD):
	numofG=len(arrayG)
	numofD=len(arrayD)
	NN=[]
	if(numofG==0 and numofD==0):
		NN.append([[1.]])
	else:
		GG=copy.deepcopy(arrayG)
		DD=copy.deepcopy(arrayD)
		if(numofG==0):
			GG=[[0,0,0,0]]
		if(numofD==0):
			DD=[[0,0,0,0]]
		for D in DD:
			tmpdir=[]
			for G in GG:
				tmpiou=calcIOU(D[0],D[1],D[2],D[3],G[0],G[1],G[2],G[3])
				tmpdir.append(tmpiou)
			NN.append(tmpdir)
	return NN
#--------------------------------------------------------------------------------------------------------------
#read GroundTruth data from imgidx.mat file
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
		Data['name'].append(f[name[i,0]].value.tobytes()[::2].decode())
		Data['label'].append(f[label[i,0]].value)
		if(f[label[i,0]].value!=0):
			Data['anno'].append(map(list,zip(*f[anno[i,0]].value)))
		else:
			Data['anno'].append([])
	return Data
#-------------------------------------------------------------------------------------------------------
#randomly generate detection result for verification
def createdetectformat(Data):
	DTid=[]
	DTanno=[]
	Detectdicct={"id":DTid,"anno":DTanno}
	for i,itr in enumerate(list(Data['anno'])):
		rndresult=random.randint(0,10)
		DTid.append(i)
		if(rndresult>0):
			tmp=[]
			for j in range(rndresult):#generate result for per img
				rx1=random.randint(0,640)
				ry1=random.randint(0,480)
				tmp.append([rx1,ry1,random.randint(rx1,640),random.randint(ry1,480)])#x1 y1 x2 y2
			DTanno.append(tmp)
		else:
			DTanno.append([])
	with open("DataDTTRY.json", "w") as outfile:
		json.dump(Detectdicct, outfile, indent=4)
#-----------------------------------------------------------------------------------------------------------
#return matching result from an image
def matchperimg(iou,thres):
	mch=[]
	if(iou==[[[1.]]]):
		mch.append([1])
	else:
		gtm=[0 for i in range(len(iou[0]))]
		for indd,D in enumerate(iou):
			tmpG=[0 for i in range(len(D))]
			thr=thres
			thrind=-1
			for indg,G in enumerate(D):
				if(gtm[indg]>0):
					continue
				if(float(G)<thr):
					continue
				thr=G
				thrind=indg
				tmpG[indg]=1
			mch.append(tmpG)
	return mch
#-----------------------------------------------------------------------------------------------------------
def evaluate(Data):
	with open('DataDTTRY2.json') as data_file:#load Detection data from transform('DataDTTRY2.json')or geneation as above("DataDTTRY.json")
		dataDT = json.load(data_file) 

	GlistDataAnn=list(Data['anno'])#from DataGT.json
	DlistDataAnn=list(dataDT['anno'])#from transCo2my.py or generate above 

	xxx=copy.deepcopy(GlistDataAnn)
	gtig = []
	for i in xxx:#count GT nunmber
		if(i==[]):
			gtig.append(0)
		else:
			for j in i:
				gtig.append(0)
	#iteratively calculate IOU in per image 
	totalIOU=[]
	count=0
	for cnt,(D,G) in enumerate(zip(DlistDataAnn,GlistDataAnn)):
		xxx=copy.deepcopy(G)
		#IOU=modelrandblk(list(G),list(xxx))#using GroundTruth to GroundTruth (precision should be 100%)
		#print(list(G));print(list(D))
		IOU=modelrandblk(list(G),list(D))#using GroundTruth vs Detection result(randomly generate) result should be 0.X%

		count+=len(D)

		totalIOU.append(IOU)

	threshold=0.5 #IOU threshold can change here
	totalmch=[]
	for iou in totalIOU:
		totalmch.append(matchperimg(iou,threshold))

	#matching ious
	dtm=[]
	for m in totalmch:
		for D in m:
			if(any(D)):
				dtm.append(1)
			else:
				dtm.append(0)

	#calculating precision and recall(on single IOU)
	tps =np.array(dtm)
	fps =np.logical_not(tps)
	tp_sum = np.cumsum(tps).astype(dtype=np.float)
	fp_sum = np.cumsum(fps).astype(dtype=np.float)
	pr = tp_sum / (tp_sum+fp_sum+np.spacing(1))

	nd = len(tps)
	rc = tp_sum/len(gtig)#/GT num
	#print(tp_sum)

	R=np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
	q  = np.zeros((len(R),))
	pr = pr.tolist(); q = q.tolist()
	for i in range(nd-1, 0, -1):
		if pr[i] > pr[i-1]:
			pr[i-1] = pr[i]
	inds = np.searchsorted(rc, R, side='left')

	recall=0
	if nd:
		recall = rc[-1]
	print(recall)
	try:
		for ri, pi in enumerate(inds):
			q[ri] = pr[pi]
	except:
		pass

	precision= np.array(q)
	zzz=np.mean(precision, axis=0)*100
	recall*=100

	print('precision in IOU='+str(threshold)+'::   '+str(zzz))
	print('recall in IOU='+str(threshold)+'::   '+str(recall))


#---------------------------------

Data=loadData()#loading GT
#createdetectformat(Data)#random generate DT
evaluate(Data)