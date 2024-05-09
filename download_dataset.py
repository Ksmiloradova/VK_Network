
import pathlib

Path = pathlib.Path

from ogb.lsc import MAG240MDataset


import urllib.request
from urllib.request import urlopen
import ssl
import json
ssl._create_default_https_context = ssl._create_unverified_context

directory = 'D:\\2\\'  
print("Path(directory): ", Path(directory))
# files = [f for f in Path(directory).iterdir() if f.is_file()]
# print("files: ", files)
# file_list = [f.name for f in files]
folder = Path(directory)
dataset = MAG240MDataset(root = folder)
print(dataset)