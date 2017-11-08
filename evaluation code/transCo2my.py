import simplejson as json
import copy
#change another formats(coco detection result) to our result format 
#checking if the result is same as CoCo
#--------------------------------
#read data from DataDt json file
with open('DataDT.json','r') as data_file:
	Data = json.load(data_file)
#--------------------------------
#out format: include its id and annotation(boundingbox[x,y,x2,y2])
DTanno=[]
DTid=[]
Detectdicct={"id":DTid,"anno":DTanno}
#---------------------------------
tmpnum=0
tmpanno=[]
cnt=0
for i in Data:
	if((i['image_id']-100000)==tmpnum):
		#print(ccc,(i['image_id']-100000))
		tmpanno.append([i['bbox'][0],i['bbox'][1],i['bbox'][2]+i['bbox'][0],i['bbox'][3]+i['bbox'][1]])#CoCo annotation in bbox is [x,y,w,h],while ours is [x,y,x2,y2]
	else:
		#print(ccc,(i['image_id']-100000))
		DTid.append(cnt)
		cnt+=1
		tmpxxx=copy.deepcopy(tmpanno)
		del tmpanno[:]
		DTanno.append(tmpxxx)
		if(i['bbox']!=[]):
			tmpanno.append([i['bbox'][0],i['bbox'][1],i['bbox'][2]+i['bbox'][0],i['bbox'][3]+i['bbox'][1]])
		tmpnum=i['image_id']-100000
DTanno.append(tmpanno)
DTid.append(cnt)
#--------------------------------------
#write transform file to another json
with open("DataDTTRY2.json", "w") as outfile:
	json.dump(Detectdicct, outfile, indent=4)
