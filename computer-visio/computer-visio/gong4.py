
# def a():
    # return
# res=a()
# if not res:
    # print ("#######")

# file_open=open("/home/congleng/1.txt")  
# content=file_open.readlines()
# if content=="":
    # print ("#######")
# file_open.close()


import os
import chardet
for item in os.listdir("/home/congleng/work/electrocardiogram/样本数据_20170308/样本数据"):
    #print (item)
    print chardet.detect(item)