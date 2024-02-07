# Configuration file
import sys
import socket

"""
Configure device
"""


hostname = socket.gethostname()
CConfig=dict()
CConfig['hostname']=hostname
if hostname=='DESKTOP-K5BEHE3':         
    CConfig['device']='local'
    CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
    CConfig['datapath']='C:/DLData'
    CConfig['datapath_source']='C:/DLData'
    CConfig['replace_dataset']=('', '')
    CConfig['OS']='WIN'
elif hostname=='erde':
    CConfig['device']='erde'
    CConfig['srcpath']='/data/Code/RS/src'
    CConfig['datapath']='/data/datasets'
    CConfig['datapath_source']='/data/datasets'
    CConfig['replace_dataset']=('C:/DLdata/OneraChangeNet/patches_center', '/data/datasets/OneraChangeNet/patches_center')
    CConfig['OS']='LINUX'
elif hostname=='DESKTOP-458KDQP':
    CConfig['device']='local'
    CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
    CConfig['datapath']='C:/DLData'
    CConfig['datapath_source']='C:/DLData'
    CConfig['replace_dataset']=('', '')
    CConfig['OS']='LINUX'
elif hostname=='berni-pc':
    CConfig['device']='local'
    CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
    CConfig['datapath']='C:/DLData'
    CConfig['datapath_source']='C:/DLData'
    CConfig['replace_dataset']=('', '')
    CConfig['OS']='WIN'
elif hostname=='foellmer':
    CConfig['device']='local'
    #CConfig['srcpath']='/mnt/SSD2/cloud_data/Projects/DL/Code/src' 
    #CConfig['datapath']='/mnt/SSD2/cloud_data/Projects/DL/Code/src/datasets'
    CConfig['srcpath']='/mnt/SSD2/cloud_data/Projects/CTP/src' 
    CConfig['datapath']='/mnt/SSD2/cloud_data/Projects/CTP/src/datasets'
    CConfig['OS']='LINUX'
    sys.path.append('/mnt/SSD2/cloud_data/Projects/DL/Code/src')
elif 'sc' in hostname:
    CConfig['device']='local'
    CConfig['srcpath']='/home/foellmeb/DL/Code/DL/src' 
    CConfig['datapath']='/home/foellmeb/DL/Code/DL/src/datasets'
    CConfig['OS']='LINUX'
    sys.path.append('/home/foellmeb/DL/Code/DL/src')
elif 'kirk' in hostname:
    CConfig['device']='local'
    CConfig['srcpath']='/home/bernhard/code/CTP/src' 
    CConfig['datapath']='/home/bernhard/code/CTP/src/datasets'
    CConfig['OS']='LINUX'
    sys.path.append('/home/bernhard/code/CTP/src')
else:
    raise ValueError('Hostname is not known! Please add your host settings to the config.py list in the config folder.')


# hostname = socket.gethostname()
# CConfig=dict()
# CConfig['hostname']=hostname
# if hostname=='DESKTOP-K5BEHE3':         
#     CConfig['device']='local'
#     CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
#     CConfig['datapath']='C:/DLData'
#     CConfig['datapath_source']='C:/DLData'
#     CConfig['replace_dataset']=('', '')
#     CConfig['OS']='WIN'
# elif hostname=='erde':
#     CConfig['device']='erde'
#     CConfig['srcpath']='/data/Code/RS/src'
#     CConfig['datapath']='/data/datasets'
#     CConfig['datapath_source']='/data/datasets'
#     CConfig['replace_dataset']=('C:/DLdata/OneraChangeNet/patches_center', '/data/datasets/OneraChangeNet/patches_center')
#     CConfig['OS']='LINUX'
# elif hostname=='DESKTOP-458KDQP':
#     CConfig['device']='local'
#     CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
#     CConfig['datapath']='C:/DLData'
#     CConfig['datapath_source']='C:/DLData'
#     CConfig['replace_dataset']=('', '')
#     CConfig['OS']='LINUX'
# elif hostname=='berni-pc':
#     CConfig['device']='local'
#     CConfig['srcpath']='H:/cloud/cloud_data/Projects/DL/Code/src' 
#     CConfig['datapath']='C:/DLData'
#     CConfig['datapath_source']='C:/DLData'
#     CConfig['replace_dataset']=('', '')
#     CConfig['OS']='WIN'
# elif hostname=='foellmer':
#     CConfig['device']='local'
#     #CConfig['srcpath']='/mnt/SSD2/cloud_data/Projects/DL/Code/src' 
#     #CConfig['datapath']='/mnt/SSD2/cloud_data/Projects/DL/Code/src/datasets'
#     CConfig['srcpath']='/mnt/SSD2/cloud_data/Projects/CTP/src' 
#     CConfig['datapath']='/mnt/SSD2/cloud_data/Projects/CTP/src/datasets'
#     CConfig['OS']='LINUX'
#     sys.path.append('/mnt/SSD2/cloud_data/Projects/DL/Code/src')
# elif 'sc' in hostname:
#     CConfig['device']='local'
#     CConfig['srcpath']='/home/foellmeb/DL/Code/DL/src' 
#     CConfig['datapath']='/home/foellmeb/DL/Code/DL/src/datasets'
#     CConfig['OS']='LINUX'
#     sys.path.append('/home/foellmeb/DL/Code/DL/src')
# elif 'kirk' in hostname:
#     CConfig['device']='local'
#     CConfig['srcpath']='/home/bernhard/code/CTP/src' 
#     CConfig['datapath']='/home/bernhard/code/CTP/src/datasets'
#     CConfig['OS']='LINUX'
#     sys.path.append('/home/bernhard/code/CTP/src')
# else:
#     raise ValueError('Hostname is not known! Please add your host settings to the config.py list in the config folder.')
