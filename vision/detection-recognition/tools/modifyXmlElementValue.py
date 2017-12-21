from __future__ import division
import os
from xml.etree.ElementTree import parse
from xlrd import open_workbook
from collections import OrderedDict


AnnoPath = 'C:\\Users\\Seven\\Desktop\\VOC\\Annotations\\'
ProcessedPath = 'C:\\Users\\Seven\\Desktop\\VOC\\ProcessedPath\\'

if not os.path.exists(ProcessedPath):
    os.makedirs(ProcessedPath)

## read the correspondent key and value from xxx.xlsx to OrderedDict()
dict_product=OrderedDict()
wb = open_workbook('C:\\Users\\Seven\\Desktop\\VOC\\产品名称与条形码20171213.xlsx')
for sheet in wb.sheets():
    number_of_rows = sheet.nrows
    for row in range(1, number_of_rows):
        key  = (sheet.cell(row, 1).value)
        value = (sheet.cell(row, 2).value)
        try:
            value = str(value)
        except ValueError:
            pass
        finally:
            dict_product[key] = value

for fileItem in os.listdir(AnnoPath):
    try:
        fileItem_pre, ext = os.path.splitext(fileItem)
        if ext=='.xml':
            xmlfile = AnnoPath + fileItem
            doc = parse(xmlfile)
            root = doc.getroot()
            objectlist = root.findall('object')

            for object in objectlist:
                name =object.find('name')
                objectname = name.text
                if objectname in dict_product:
                    name.text = dict_product[objectname]
                # print objectname
                print (objectname)

            doc.write(ProcessedPath + fileItem, xml_declaration=True)

    except Exception as e:
        print (e)
