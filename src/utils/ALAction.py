# -*- coding: utf-8 -*-
import os, sys
import json
import numpy as np
#from modules.XAL.ALSample import ALSample

class ALAction():
  """ALAction
  ALAction - Active learning action class
  """

  def __init__(self):
    """
    Init class
    """
    self.id = None
    self.idp = None
    self.name = None
    self.imagename = None
    self.pseudoname = None
    self.refinename = None
    self.maskname = None
    self.imageext = None
    self.pseudoext = None
    self.refineext = None
    self.maskext = None
    self.fip_image = None
    self.fip_pseudo = None
    self.fip_refine = None
    self.fip_mask = None
    self.status = None
    self.command = None
    self.slice = None
    self.bboxLbsOrg=None
    self.bboxUbsOrg=None
    self.bboxLbsOrgW=None
    self.bboxUbsOrgW=None
    self.dim=None
    self.msg = None
    self.label = []
    self.info = dict()

  def serialize(self):
    d=dict()
    d['id']=self.id
    d['idp']=self.idp
    d['name'] = self.name
    d['imagename'] = self.imagename
    d['pseudoname'] = self.pseudoname
    d['refinename'] = self.refinename
    d['maskname'] = self.maskname
    d['imageext'] = self.imageext
    d['pseudoext'] = self.pseudoext
    d['refineext'] = self.refineext
    d['maskext'] = self.maskext
    d['fip_image'] = self.fip_image
    d['fip_pseudo'] = self.fip_pseudo
    d['fip_refine'] = self.fip_refine
    d['fip_mask'] = self.fip_mask
    d['status'] = self.status
    d['command'] = self.command
    d['slice'] = self.slice
    d['bboxLbsOrg'] = [int(x) for x in list(self.bboxLbsOrg)]
    d['bboxUbsOrg'] = [int(x) for x in list(self.bboxUbsOrg)]
    d['bboxLbsOrgW'] = [float(x) for x in list(self.bboxLbsOrgW)]
    d['bboxUbsOrgW'] = [float(x) for x in list(self.bboxUbsOrgW)]
    d['dim'] = self.dim
    d['msg'] = self.msg
    d['label'] = self.label
    d['info'] = self.info
    return d

  @staticmethod
  def deserialize(d):
    action = ALAction()
    action.id=d['id']
    action.idp=d['idp']
    action.name = d['name']
    action.imagename = d['imagename']
    action.pseudoname = d['pseudoname']
    action.refinename = d['refinename']
    action.maskname = d['maskname']
    action.imageext = d['imageext']
    action.pseudoext = d['pseudoext']
    action.refineext = d['refineext']
    action.maskext = d['maskext']
    action.fip_image = d['fip_image']
    action.fip_pseudo = d['fip_pseudo']
    action.fip_refine = d['fip_refine']
    action.fip_mask = d['fip_mask']
    action.status = d['status']
    action.command = d['command']
    action.slice = d['slice']
    action.bboxLbsOrg = np.array(d['bboxLbsOrg'])
    action.bboxUbsOrg = np.array(d['bboxUbsOrg'])
    action.bboxLbsOrgW = np.array(d['bboxLbsOrgW'])
    action.bboxUbsOrgW = np.array(d['bboxUbsOrgW'])
    action.dim = d['dim']
    action.msg = d['msg']
    action.label = d['label']
    action.info = d['info']
    return action

  @staticmethod
  def save(fip_actionlist, actionlist):
    """
    Save actionlist
    """
    alist = []
    for action in actionlist:
      alist.append(action.serialize())
    with open(fip_actionlist, 'w') as file:
        file.write(json.dumps(alist, indent=4))

  @staticmethod
  def load(fip_actionlist):
    """
    Load actionlist
    """

    with open(fip_actionlist, 'r') as file:
      data = json.load(file)
    alist = []
    for d in data:
      alist.append(ALAction.deserialize(d))
    return alist

# #####
# def tmp():
#     a=ALAction()
#     a.name='actionname'
#     a.status='OPEN'
#     a.slice=25
#     a.imagename='imagename'
#     a.pseudoname='segname'
#     a.refinename='segname'
#     a.maskname='maskname'
#     a.fip_image='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839.mhd'
#     a.fip_pseudo='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label.nrrd'
#     a.fip_refine='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label-correct.nrrd'
#     a.fip_mask='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label.nrrd'

#     b=ALAction()
#     b.name='actionname'
#     b.status='OPEN'
#     b.slice=25
#     b.imagename='imagename'
#     b.pseudoname='segname'
#     b.refinename='segname'
#     b.maskname='maskname'
#     b.fip_image='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839.mhd'
#     b.fip_pseudo='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label.nrrd'
#     b.fip_refine='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label-correct.nrrd'
#     b.fip_mask='01-BER-0010_1.2.392.200036.9116.2.6.1.37.2426555318.1460675933.966839-label.nrrd'


#     fip_actionlist='/mnt/SSD2/cloud_data/Projects/CTP/src/modules/XAL/XALabeler/XALabeler/data/sample/actionlist.json'
#     actionlist=[a,b]
#     ALAction.save(fip_actionlist, actionlist)



