# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:03:47 2020

@author: huijianpzh
"""
from xml.dom import minidom
import xml.etree.ElementTree as ET

tree = ET.parse("template.xml")

root = tree.getroot()

def write_xml(input_,output_):
    
    provider = r"中电慧遥"
    author = r"huijian"
    pluginname = r"地物标注"
    pluginclass= r"标注"
    
    #1. 创建dom树对象
    doc = minidom.Document()

    #2. 创建根结点，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)
    
    # source
    source_node = doc.createElement("source")
    filename_node = doc.createElement("filename")
    filename_val = doc.createTextNode(input_) # 文件名称
    filename_node.appendChild(filename_val)
    origin_node = doc.createElement("origin")
    origin_val = doc.createTextNode('GF2/GF3')
    origin_node.appendChild(origin_val)
    
    source_node.appendChild(filename_node)
    source_node.appendChild(origin_node)
    root_node.appendChild(source_node)
    
    # research
    research_node = doc.createElement("research")
    version_node = doc.createElement("version")
    version_val = doc.createTextNode("4.0")
    version_node.appendChild(version_val)
    provider_node = doc.createElement("provider")
    provider_val = doc.createTextNode(provider)
    provider_node.appendChild(provider_val)
    author_node = doc.createElement("author")
    author_val = doc.createTextNode(author)
    author_node.appendChild(author_val)
    pluginname_node = doc.createElement("pluginname")
    pluginname_val = doc.createTextNode(pluginname)
    pluginname_node.appendChild(pluginname_val)
    pluginclass_node = doc.createElement("pluginclass")
    pluginclass_val = doc.createTextNode(pluginclass)
    pluginclass_node.appendChild(pluginclass_val)
    time_node = doc.createElement("time")
    time_val = doc.createTextNode("2020-07-2020-11")
    time_node.appendChild(time_val)
    
    research_node.appendChild(version_node)
    research_node.appendChild(provider_node)
    research_node.appendChild(author_node)
    research_node.appendChild(pluginname_node)
    research_node.appendChild(pluginclass_node)
    research_node.appendChild(time_node)
    root_node.appendChild(research_node)
    
    
    # segmentation
    segmentation_node = doc.createElement("segmentation")
    resultfile_node = doc.createElement("resultfile")
    resultfile_val = doc.createTextNode(output_) # 输出结果名称 
    resultfile_node.appendChild(resultfile_val)
    
    segmentation_node.appendChild(resultfile_node)
    root_node.appendChild(segmentation_node)
    
    with open("huijian.xml", "w", encoding="utf-8") as f:
        # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")
def write_xml2():
    #1. 创建dom树对象
    doc = minidom.Document()

    #2. 创建根结点，并用dom对象添加根结点
    root_node = doc.createElement("annotation")
    doc.appendChild(root_node)

    #3. 创建结点，结点包含一个文本结点, 再将结点加入到根结点
    folder_node = doc.createElement("folder")
    folder_value = doc.createTextNode('user')
    folder_node.appendChild(folder_value)
    root_node.appendChild(folder_node)

    filename_node = doc.createElement("filename")
    filename_value = doc.createTextNode('0000001.jpg')
    filename_node.appendChild(filename_value)
    root_node.appendChild(filename_node)

    path_node = doc.createElement("path")
    path_value = doc.createTextNode('/home')
    path_node.appendChild(path_value)
    root_node.appendChild(path_node)

    source_node = doc.createElement("source")
    database_node = doc.createElement("database")
    database_node.appendChild(doc.createTextNode("Unknown"))
    source_node.appendChild(database_node)
    root_node.appendChild(source_node)

    size_node = doc.createElement("size")
    for item, value in zip(["width", "height", "depth"], [1920, 1080, 3]):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        size_node.appendChild(elem)
    root_node.appendChild(size_node)

    seg_node = doc.createElement("segmented")
    seg_node.appendChild(doc.createTextNode(str(0)))
    root_node.appendChild(seg_node)

    obj_node = doc.createElement("object")
    name_node = doc.createElement("name")
    name_node.appendChild(doc.createTextNode("boat"))
    obj_node.appendChild(name_node)

    pose_node = doc.createElement("pose")
    pose_node.appendChild(doc.createTextNode("Unspecified"))
    obj_node.appendChild(pose_node)

    trun_node = doc.createElement("truncated")
    trun_node.appendChild(doc.createTextNode(str(1)))
    obj_node.appendChild(trun_node)

    trun_node = doc.createElement("difficult")
    trun_node.appendChild(doc.createTextNode(str(0)))
    obj_node.appendChild(trun_node)

    bndbox_node = doc.createElement("bndbox")
    for item, value in zip(["xmin", "ymin", "xmax", "ymax"], [103, 1, 634, 402]):
        elem = doc.createElement(item)
        elem.appendChild(doc.createTextNode(str(value)))
        bndbox_node.appendChild(elem)
    obj_node.appendChild(bndbox_node)
    root_node.appendChild(obj_node)


    with open("huijian.xml", "w", encoding="utf-8") as f:
        # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")

    # 每一个结点对象（包括dom对象本身）都有输出XML内容的方法，如：toxml()--字符串, toprettyxml()--美化树形格式。
    # print(doc.toxml(encoding="utf-8"))  # 输出字符串
    # print(doc.toprettyxml(indent='', addindent='\t', newl='\n', encoding="utf-8"))   #输出带格式的字符串
    # doc.writexml() #将prettyxml字符串写入文件
write_xml(input_="1.tf",output_="1_gt.png")


    
