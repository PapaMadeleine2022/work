from __future__ import division
import os
from PIL import Image, ImageFont, ImageDraw
import xml.dom.minidom
import numpy as np

ImgPath = 'C:\\Users\\Seven\\Desktop\\VOC\\JPEGImages\\'
AnnoPath = 'C:\\Users\\Seven\\Desktop\\VOC\\Annotations\\'
ProcessedPath = 'C:\\Users\\Seven\\Desktop\\VOC\\ProcessedPath\\'

if not os.path.exists(ProcessedPath):
    os.makedirs(ProcessedPath)

imagelist = os.listdir(ImgPath)



for image in imagelist:
    try:
        print ('a new image:', image)
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'

        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement

        filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')

        print (imgfile)
        im = Image.open(imgfile)
        dr = ImageDraw.Draw(im)
        # get a font
        fnt = ImageFont.truetype("arial.ttf", 12, encoding="unic")
        i = 1
        for object in objectlist:
            namelist =object.getElementsByTagName('name')
            # print 'namelist:',namelist
            objectname = namelist[0].text
            # print objectname
            print (objectname)
            bndbox=[]
            bndbox.append(object.getElementsByTagName('bndbox')[0])
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].text)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].text)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].text)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].text)
                w = x2 - x1
                h = y2 - y1

                dr.rectangle(((x1, y1),(x2, y2)), outline = "blue")
                dr.text((x1, y1-10), objectname, font=fnt, fill=(255,255,255,128))

        im.show()
        im.save(ProcessedPath + image)
    except Exception as e:
        print (e)

