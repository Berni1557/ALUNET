# -*- coding: utf-8 -*-
import yaml
from collections import defaultdict
from enum import Enum

class YAML_MODE(Enum):
    """ Enum for different yml modes
    """ 
    DEFAULT = 1     # Use default properties
    LOAD = 2        # load properties from yml file
    SAVE = 3        # Use default properties and save to yml file
    UPDATE = 4      # Update properties from file and save to new yml file

class YAML:
    """ YAML class to save and read yml files
    """ 
    def __init__(self):
        """ Init YAML class
        """
        pass        

    def update(self, dictionary, filepath):
        """ Update existing yml file by values of disctionary
        
        :param dictionary: Dictionary with values to update
        :type dictionary: dict
        :param filepath: Filepath to yml file
        :type filepath: str
        """
        
        # Load properties
        props = self.load(filepath)
        # Set new values
        for key in dictionary.keys():
            props[key] = dictionary[key]
        # Save properties
        self.save(props, filepath)
        return props
            
    def save(self, dictionary, filepath):
        """ Save dictionary to yml file
        
        :param dictionary: Dictionary to save ti yml file
        :type dictionary: dict
        :param filepath: Filepath to yml file
        :type filepath: str
        """
        with open(filepath, 'w') as filepath:
            yaml.dump(dict(dictionary), filepath, default_flow_style=False)

    def load(self, filepath):
        """ Load dictionary to yml file
        
        :param filepath: Filepath to yml file
        :type filepath: str
        """
        dictionary = None
        with open(filepath, 'r') as stream:
            docs = yaml.load_all(stream, Loader=yaml.FullLoader)
            #print('docs', docs)
            dictionary = dict()
            for doc in docs:
                if doc:
                    for k,v in doc.items():
                        #print(k, "->", v)
                        dictionary[k]=v
        return defaultdict(lambda: None, dictionary)


#######################################################################

## Create yml file
#filepath = 'H:/cloud/cloud_data/Projects/DL/Code/src/tmp/props.yml'
#file = YAML()
## Save yml file
#props = defaultdict(lambda: None)
#props['var01'] = 1
#props['var02'] = '123'
#file.save(props, filepath)
## Load properties
#props_load = file.load(filepath)
## Update properties
#props_update = defaultdict(lambda: None)
#props_update['var02'] = '12345'
#file.update(props_update, filepath)
#props_new = file.load(filepath)
