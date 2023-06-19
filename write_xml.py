# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET





if __name__ == '__main__':
    flows = ET.Element("flows")
    interval = ET.SubElement(flows, "interval", begin='0', end='3599')
    
    ET.SubElement(interval, "flow", name="blah", flows='2321')
    ET.SubElement(interval, "flow", name="asdfasd", flows='2321')
    
    tree = ET.ElementTree(flows)
    tree.write("filename.xml")
