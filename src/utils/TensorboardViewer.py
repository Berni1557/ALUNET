# -*- coding: utf-8 -*-
import os
import sys
import_DL_path = 'H:/cloud/cloud_data/Projects/DeepLearning/imports'
sys.path.append(import_DL_path)
import threading
from subprocess import call
from utils.YAML import YAML
from utils.SaveState import SaveState
import subprocess
import keyboard

def launchTensorBoard(tensorBoardLog_path, tensorBoard_path, update_time=None):
    print('command', tensorBoard_path + ' --logdir=' + tensorBoardLog_path + ' --host localhost --port=6006')
    #os.system(tensorBoard_path + ' --logdir=' + '"' + tensorBoardLog_path + '"' + ' --host localhost --port=6006')

    if update_time:
        while True:
            process = subprocess.Popen([tensorBoard_path, '--logdir', tensorBoardLog_path, '--host', 'localhost', '--port','6006'])
            print('Press button "ctrl + n" to restart tensorboard visualization.')
            keyboard.wait('ctrl + n')
            process.kill()
            print('Restarting TensorBoard')
        else:
            process = subprocess.Popen([tensorBoard_path, '--logdir', tensorBoardLog_path, '--host', 'localhost', '--port','6006'])
    return

    
class TensorboardViewer:

    yml = YAML()
    props = None
    
    def __init__(self, settingsfilepath, mode=SaveState.SAVE, update_time=None):
        #if os.path.isfile(settingsfilepath):
        self.update_time = update_time
        if mode==SaveState.LOAD:
            self.props = self.yml.load(settingsfilepath)
        elif mode==SaveState.SAVE:
            self.props = dict(
                firefox_path = "C:/Program Files/Mozilla Firefox/firefox.exe",
                IP = "localhost:6006",
                tensorBoard_path = "C:/Anaconda3/envs/env_RS/Scripts/tensorboard.exe"
            )
            self.yml.save(self.props, settingsfilepath)
        else:
            raise ValueError('Mode ' + mode + ' is not defined.')

    def createYML(self, settingsfilepath):
        self.yml.save(self.props, settingsfilepath) 
        
    def start(self, tensorBoardLog_path='/logs', tensorBoardPath=None):
        if not tensorBoardPath:
            tensorBoard_path=self.props['tensorBoard_path']
        t = threading.Thread(target=launchTensorBoard, args=([tensorBoardLog_path, tensorBoard_path, self.update_time]))
        t.start()
        call([self.props['firefox_path'], "-new-window", self.props['IP']])  
