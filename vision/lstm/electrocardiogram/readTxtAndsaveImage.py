# -*-coding:utf-8-*-
from numpy import zeros
import matplotlib 
import matplotlib.pyplot as plt
import os

#读取文件内数据为矩阵
def file2matrix(filename):
    file_open=open(filename)
    content=file_open.readlines()
    file_open.close()
    if len(content)==0:
        return
    else:
        #去除content两边的空格
        content=content[0].strip().split(" ")
        number=len(content)
        returnMat=zeros((number,2))
        index=0
        for i in content:
            returnMat[index,0]=index
            returnMat[index,1]=int(i)
            index+=1
        return returnMat

#遍历某目录下指定后缀形式的文件
def eachFile(filedir,postfix):
    filenames=os.listdir(filedir)
    res=[]
    for filename in filenames:
        #使用py3则不需要对os.listdir(filedir)下的中文目录名解码
        # basename=os.path.basename(filename)
        # try:
        #     filename=basename.decode("gb2312")
        # except UnicodeDecodeError:
        #     filename=basename.decode("utf-8")
        if os.path.splitext(filename)[1] == postfix:
            filename=os.path.join(filedir,filename)
            res.append(filename)
    return res

#遍历某目录下的文件或者目录
def eachFileOrDir(folderFullName):
    nextFileOrDirs=os.listdir(folderFullName)
    index=0
    for nextFileOrDir in nextFileOrDirs:
        #使用py3则不需要对os.listdir(filedir)下的中文目录名解码
        # basename=os.path.basename(nextFileOrDir)
        # try:
        #     nextFileOrDir=basename.decode("gb2312")
        # except UnicodeDecodeError:
        #     nextFileOrDir=basename.decode("utf-8")
        nextFileOrDir=os.path.join(folderFullName,nextFileOrDir)
        nextFileOrDirs[index]=nextFileOrDir
        index+=1
    return nextFileOrDirs

#读取指定后缀形式的文件内容，并将内容画出
def readTxtAndsaveImage(dir,postfix,rows):
    nextDirs=eachFileOrDir(dir)
    print nextDirs
    for nextDir in nextDirs:
        print nextDir
        filenames=eachFile(nextDir,postfix)
        for filename in filenames:
            plt_filename(filename,postfix,rows)
#画图
def plt_filename(filename,postfix,rows):
    dataMat=file2matrix(filename)
    if dataMat is not None:
        fig=plt.figure(figsize=(rows*8,1*8))
        # ax=fig.add_subplot(1,1,1)
        plt.plot(dataMat[:,0],dataMat[:,1])
        # for i in range(rows):
        #     ax=fig.add_subplot(1,rows,i+1)
        #     ax.scatter(dataMat[len(dataMat)/rows*i:len(dataMat)/rows*(i+1),0],dataMat[len(dataMat)/rows*i:len(dataMat)/rows*(i+1),1])
        #plt.show()
        plt.savefig(filename.split(postfix)[0]+'.png')
        plt.close()
    else:
        return

if __name__ == "__main__":
    readTxtAndsaveImage('/home/congleng/work/electrocardiogram/样本数据_20170308/样本数据','.txt',14)
