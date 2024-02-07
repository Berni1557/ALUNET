# -*- coding: utf-8 -*-
import os, sys
#from config.config import CConfig
import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
# import datetime
from collections import defaultdict
from glob import glob
from utils.helper import splitFilePath, splitFolderPath
import numpy as np
# import medpy
# import subprocess
import cv2
# import pydicom
from scipy.ndimage import zoom
import nrrd
#from helper.helper import compute_one_hot_np, compute_one_hot_noclass_np
from utils.helper import compute_one_hot_np
from SimpleITK import ConnectedComponentImageFilter
import shutil
import torch 
import skimage
from skimage.transform import resize
from typing import Callable, Iterator, Union, Optional
import math

class CTImage(object):
    """Create CT image
    
    Attributes
    ----------
    name : str
        Name of the image
    image_sitk : sitk.Image
        SimpleITK image 
    image_label : sitk.Image
        Label of the image
    filepath : str
        Filepath of the image
    """
    
    name = ''           #: Image name
    image_sitk = None   #: SimpleITK image object
    image_label = None  #: Label of the image
    filepath = ''       #: Filepath of the image
    
    def __init__(self, fip_arr=None, axis='CWH'):
        """Init

        Parameters
        ----------
        fip : str, optional
            File path to the image that will be loaded immediately
        """
        
        self.info = defaultdict(lambda: None)
        if isinstance(fip_arr, np.ndarray):
            self.setImage(fip_arr, axis=axis)
        elif isinstance(fip_arr, torch.Tensor):
            self.setImage(fip_arr.numpy(), axis=axis)
        elif isinstance(fip_arr, str):
            self.load(fip_arr)
        else:
            pass

        
    def load(self, path:str='H:/cloud/cloud_data/Projects/DL/Code/src/tmp/ct/VAV4P2CTI.mhd', image_format:str=None):
        """Load image from file

        Load an image from file or folder.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        path : str, optional
            Filepath to the image or folderpath to the image
        image_format : str
            File fomat extension ('mhd', 'nrrd', 'png', 'dcm') if None, extension is guessed (default is None)
        """
        
        if os.path.isdir(path):
            # Read from folder
            if image_format is None:
                file = glob(path + '/*')[0]
                _, _, ext = splitFilePath(file)
                image_format = ext[1:]
            
            # Read mhd file
            if image_format=='mhd':
                file = glob(path + '/*.mhd')[0]
                if os.path.exists(file):
                    self.image_sitk = sitk.ReadImage(file)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            # Read nrrd file
            elif image_format=='nrrd':
                file = glob(path + '/*.nrrd')[0]
                if os.path.exists(file):
                    self.image_sitk = sitk.ReadImage(file)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            # Read png file
            elif image_format=='png':
                file = glob(path + '/*.png')[0]
                if os.path.exists(file):
                    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    self.image_sitk = sitk.GetImageFromArray(image)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            # Read dicom file (.dcm)
            elif image_format=='dcm':
                if len(glob(path + '/*.dcm'))==1:
                    file = glob(path + '/*.dcm')[0]
                    if os.path.exists(file):
                        self.image_sitk = sitk.ReadImage(file)
                        _, filename, _ = splitFilePath(file)
                        self.name = filename
                        self.filepath = path
                    else:
                        raise ValueError('Filepath: ' + path + ' does not exist.')
                else:
                    filepathDcm = glob(path + '/*.dcm')[0]
                    file_reader = sitk.ImageFileReader()
                    file_reader.SetFileName(filepathDcm)
                    file_reader.ReadImageInformation()
                    series_ID = file_reader.GetMetaData('0020|000e')
                    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_ID)
                    data = sitk.ReadImage(sorted_file_names)
                    self.image_sitk = data
        else:
            # Reads the image using SimpleITK
            if os.path.exists(path):
                self.image_sitk = sitk.ReadImage(path)
                _, filename, _ = splitFilePath(path)
                self.name = filename
                self.filepath = path
            else:
                raise ValueError('Filepath: ' + path + ' does not exist.')


    def save(self, path:str, image_format:str='mhd', dtype=None):
        """Save image to file

        Load an image from file or folder.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        path : str, optional
            Filepath to the folder of the image
        image_format : str
            File fomat extension ('mhd', 'nrrd', 'png', 'nii')
        dtype : sitk Pixel Types
            Pixel Type of the image (e.g. sitk.sitkUInt16)
        """
        
        if dtype is not None:
            self.image_sitk = sitk.Cast(self.image_sitk, dtype)
        if image_format=='mhd' or image_format=='nrrd' or image_format=='nii':
            if path[-3:]=='mhd' or path[-4:]=='nrrd' or path[-3:]=='nii' or path[-6:]=='nii.gz':
                filepath = path
                sitk.WriteImage(self.image_sitk, filepath, True)
            else:
                sitk.WriteImage(self.image_sitk, os.path.join(path, self.name + '.mhd'), True)
        elif image_format=='png':
            image = self.image()
            cv2.imwrite(os.path.join(path, self.name + '.png'), image) 
        else:
            raise ValueError('Image format ' + image_format + ' is not implemented.')
            
    def image(self, axis='CWH'):
        """Get numpy array

        Convert sitk image into numpy image

        Parameters
        ----------
        axis : str, optional
            Convert array into the 3d order (e.g. CWH for channel first) (default is CWH-channel, width, hight)

        Returns
        -------
        ndarray
            Numpy array from sitk image 
        """
        
        im = sitk.GetArrayFromImage(self.image_sitk)
        im = self.swapaxes(im, 'CWH', axis)
        return im
    
    def setImage(self, arr, axis='CWH'):
        """Set image

        Set image from numpy or torch array

        Parameters
        ----------
        arr : ndarray
            Input array of image
        axis : str
            Order of channels (e.g. CWH for channel first) (default is CWH-channel, width, hight)
        """
        
        if type(arr)==torch.Tensor:
            arr = arr.numpy()
        if not axis is None:
            if axis == 'WH':
                arr = np.expand_dims(arr, axis=0)
                axis='CWH'
            arr = self.swapaxes(arr, axis, 'CWH')
        image_sitk_new = sitk.GetImageFromArray(arr)
        # Copy image information
        if self.image_sitk:
            image_sitk_new.CopyInformation(self.image_sitk)
        # Set image
        self.image_sitk = image_sitk_new
        self.shape = arr.shape
        del arr

    def copy(self):
        """Copy image

        Create deep copy of the image

        Returns
        -------
        CTImage
            Copy of the image
        """
        
        image = CTImage()
        image.setImage(self.image(axis='WHC'), axis='WHC')
        image.copyInformationFrom(self)
        return image
    
    def convert(self, dtype):
        """Convert data type

        Convert image to other datatype

        Parameters
        ----------
        dtype : sitk.dtype
            SimpleITK datatype (e.g. sitk.sitkUInt16)
        Returns
        -------
        CTImage
            Image with different data type
        """
        
        self.image_sitk = sitk.Cast(self.image_sitk, dtype)

    
    def copySpacingFrom(self, image):
        """Copy spacing

        Copy spacing from CTImage

        Parameters
        ----------
        image : CTImage
            Image with spacing
        """
        self.image_sitk.SetSpacing(image.image_sitk.GetSpacing())


    def copyInformationFrom(self, image):
        """Copy information

        Copy information from CTImage

        Parameters
        ----------
        image : CTImage
            Image with spacing
        """
        self.image_sitk.CopyInformation(image.image_sitk)
        
    def scale(self, scale_factor, interpolation=cv2.INTER_CUBIC):
        """Scale image

        Scale the image to different resolution

        Parameters
        ----------
        scale_factor : float, tuple
            Scale factor can be a float scaling factor (e.g. 2.0) or a tuple of the target resolution (e.g. (256,256,3))
        interpolation : cv2 interpolation, tuple
            Interpolation (e.g. cv2.INTER_CUBIC)  
        """
        if type(scale_factor)==float:
            arr = self.image(axis='WHC')
            spacing = self.image_sitk.GetSpacing()
            height, width = arr.shape[:2]
            width_sc = int(round(width*scale_factor))
            height_sc = int(round(height*scale_factor))
            arr_sc = cv2.resize(arr, dsize=(width_sc, height_sc), interpolation=interpolation)
            self.image_sitk=None
            self.setImage(arr_sc, axis='WHC')
            scale_factor_inv = 1/scale_factor
            self.image_sitk.SetSpacing((spacing[0]*scale_factor_inv, spacing[1]*scale_factor_inv, spacing[2]))
        elif type(scale_factor)==tuple:
            arr = self.image(axis='WHC')
            spacing = self.image_sitk.GetSpacing()
            height, width, depth = arr.shape
            arr_sc = zoom(arr, scale_factor)
            self.image_sitk=None
            self.setImage(arr_sc, axis='WHC')
            self.image_sitk.SetSpacing((spacing[0]*scale_factor[0], spacing[1]*scale_factor[1], spacing[2]*scale_factor[2]))
        else:
            raise ValueError('Scaling method not defined.')
        
        
    def transformIndexToPhysicalPoint(self, pixList, label=None):
        """Transform index coodinates to physical points

        Parameters
        ----------
        pixList : list of tuple
            List of indices of positions e.g. [(100,10,56), (100,200,40)]
        
        Returns
        -------
        List of PhysicalPoint
            list of PhysicalPoint
        """
        if type(pixList) == list:
            pointList=[]
            for pix in pixList:
                coordinates = self.image_sitk.TransformIndexToPhysicalPoint(pix)
                point = PhysicalPoint(coordinates)
                if label is not None:
                    point.label = label
                pointList.append(point)
        else:
            pointList=[]
            for i in range(pixList.shape[0]):
                coordinates = self.image_sitk.TransformIndexToPhysicalPoint((int(pixList[i,0]), int(pixList[i,1]), int(pixList[i,2])))
                pointList.append(PhysicalPoint(coordinates))
                
        return pointList
    
    def transformPixelPointToPhysicalPoint(self, pixList, label=None):
        """Transform pixel coodinates to physical points

        Parameters
        ----------
        pixList : list of tuples
            List of PixelPoint objects
        
        Returns
        -------
        list of PhysicalPoint
            List of PhysicalPoints
        """
        
        if type(pixList) == list:
            pointList=[]
            for pix in pixList:
                coordinates = self.image_sitk.TransformIndexToPhysicalPoint(pix.coordinates)
                point = PhysicalPoint(coordinates)
                if label is not None:
                    point.label = label
                pointList.append(point)
        else:
            pointList=[]
            for i in range(pixList.shape[0]):
                coordinates = self.image_sitk.TransformIndexToPhysicalPoint((int(pixList[i,0]), int(pixList[i,1]), int(pixList[i,2])))
                pointList.append(PhysicalPoint(coordinates))
                
        return pointList
    
    def transformPhysicalPointToIndex(self, pointList):
        """Transform physical points to index coodinates 

        Parameters
        ----------
        pointList : list of PhysicalPoint
            List of PhysicalPointss
        
        Returns
        -------
        list of tuple
            List of Indices
        """
        pixList=[]
        for point in pointList:
            pix = self.image_sitk.TransformPhysicalPointToIndex(point.coordinates)
            pixList.append(pix)
        return pixList

    def transformPhysicalPointToPixelPoint(self, pointList):
        """Transform physical points to pixel points

        Parameters
        ----------
        pointList : list of PhysicalPoint
            List of PhysicalPoints
        
        Returns
        -------
        list of PixelPoint
            List of PixelPoints
        """
        pixList=[]
        for point in pointList:
            pix = self.image_sitk.TransformPhysicalPointToIndex(point.coordinates)
            pixList.append(PixelPoint(pix))
        return pixList
        
    
    @staticmethod
    def swapaxes(image, axis_in='WHC', axis_out='CWH'):
        """Swap axex
        
        Swap axex e.g. from 'WHC' to 'CWH'

        Parameters
        ----------
        image : CTImage
            Input image
        axis_in : str
            Axis of the original image (e.g. 'WHC')
        axis_out : str
            Axis of the target image (e.g. 'CWH')        
        Returns
        -------
        CTImage
            CTImage with swaped axis
        """
        
        if axis_in=='WHC' and axis_out=='CWH':
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 1, 2)
        elif axis_in=='CWH' and axis_out=='WHC':
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 0, 1)
        elif axis_in=='CWH' and axis_out=='CWH':
            image = image
        elif axis_in=='WHC' and axis_out=='WHC':
            image = image
        elif axis_in=='WH' and axis_out=='WHC':
            image = np.expand_dims(image, axis=2)
        elif axis_in=='CWH' and axis_out=='HWC':
            image = np.swapaxes(image, 0, 2)
            image = image
        else:
            raise ValueError('Tranformation from ' + axis_in + + ' to ' + axis_out + ' does not exist.')
        return image
        

    
    # def setLabel(self, arr, axis='CWH'):
    #     #atlas.atlas.image_sitk.CopyInformation(atlas.atlas.image_sitk)
    #     if self.image_label is None:
    #         arr = self.swapaxes(arr, axis, 'CWH')
    #         self.image_label = sitk.GetImageFromArray(arr)
    #         if self.image_sitk:
    #             print('Select label metadata from image:sitk.')
    #             self.image_label.CopyInformation(self.image_sitk)
                                                 
    # def label(self, axis='CWH'):
    #     if self.image_label:
    #         im = sitk.GetArrayFromImage(self.image_label)
    #         im = self.swapaxes(im, 'CWH', axis)
    #         return im
    #     else:
    #         return None
    
    # def findLabel(self, filenameRange=[0,6], folderpathReference=''):
        
    #     #filepath = 'H:/cloud/cloud_data/Projects/DL/Code/src/ct/CTRegistration/data/images/TRV4P8CTI.mhd'
    #     #folderpathReference = 'H:/cloud/cloud_data/Projects/DL/Code/src/ct/CTRegistration/data/references'
        
    #     _, filename, _ = splitFilePath(self.filepath)
    #     references = glob(folderpathReference + '/*.mhd')
    #     for ref in references:
    #         _, filenameRef, _ = splitFilePath(ref)
    #         if filenameRef[filenameRange[0]:filenameRange[1]] == filename[filenameRange[0]:filenameRange[1]]:
    #             return ref
    #     return None

    # def loadLabel(self, filepath='H:/cloud/cloud_data/Projects/DL/Code/src/tmp/ct/VAV4P2CTI.mhd'):
    #     """ Load image label
        
    #     :param filepath: Filepath to image or folderpath of an image
    #     :type filepath: str
    #     """
        
    #     if filepath:
    #         if os.path.isdir(filepath):
    #             # Reads the image from folder
    #             file = glob(filepath + '/*.mhd')[0]
    #             self.image_label = sitk.ReadImage(file)
    #         else:
    #             # Reads the image using SimpleITK
    #             self.image_label = sitk.ReadImage(filepath)     
    #     else:
    #           raise ValueError('Filepath is empty.')
        

    # def saveLabel(self, filepath='H:/cloud/cloud_data/Projects/DL/Code/src/tmp/ct/VAV4P2CTI.mhd'):
    #     """ Save image
        
    #     :param filepath: Filepath to image
    #     :type filepath: str
    #     """

    #     sitk.WriteImage(self.image_label, filepath, True) 

    def resize(self, shape=(None, 256, 256), order=3):
        """Resize image
        
        Resize image to target shape

        Parameters
        ----------
        shape : tuple
            Shape of the target image
        order : int
            The order of the spline interpolation, default is 0 if
            image.dtype is bool and 1 otherwise. The order has to be in
            the range 0-5. See `skimage.transform.warp` for detail.
        """
        arr = sitk.GetArrayFromImage(self.image_sitk)
        dtype = arr.dtype
        r0 = arr.shape[0] if shape[0] is None else shape[0]
        r1 = arr.shape[1] if shape[1] is None else shape[1]
        r2 = arr.shape[2] if shape[2] is None else shape[2]
        arr = resize(arr, (r0, r1, r2), order=order, preserve_range=True).astype(dtype)
        self.image_sitk = sitk.GetImageFromArray(arr)
    
    def showImageJ(self, ImageJPath='/home/bernifoellmer/bin/Fiji.app/ImageJ-linux64'):
        """Show CT image in ImageJ

        Parameters
        ----------
        ImageJPath : str
            Filepath to ImageJPath app
        """
        # Create temporary folder and save image in tmp folder
        folderpath_tmp = os.path.join(CConfig['srcpath'], 'tmp_show')
        os.makedirs(folderpath_tmp, exist_ok=True)
        filepath_tmp = os.path.join(folderpath_tmp, 'tmp.mhd')
        self.save(filepath_tmp)
        # Open ImageJ and image
        command = ImageJPath + ' ' + filepath_tmp
        os.system(command)
        # Delete temporary folder
        if os.path.exists(folderpath_tmp):
            shutil.rmtree(folderpath_tmp)

        

    # def showImage3D(self, xslices=[], yslices=[], zslices=[], title=None, margin=0.05,dpi=80):
    #     img_xslices = [self.image_sitk[s, :, :] for s in xslices]
    #     img_yslices = [self.image_sitk[:, s, :] for s in yslices]
    #     img_zslices = [self.image_sitk[:, :, s] for s in zslices]
    
    #     maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))
    
    
    #     img_null = sitk.Image([0, 0], self.image_sitk.GetPixelID(),
    #                           self.image_sitk.GetNumberOfComponentsPerPixel())
    
    #     img_slices = []
    #     d = 0
    
    #     if len(img_xslices):
    #         img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
    #         d += 1
    
    #     if len(img_yslices):
    #         img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
    #         d += 1
    
    #     if len(img_zslices):
    #         img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
    #         d += 1
    
    #     if maxlen != 0:
    #         if self.image_sitk.GetNumberOfComponentsPerPixel() == 1:
    #             img = sitk.Tile(img_slices, [maxlen, d])
    #         # TO DO check in code to get Tile Filter working with vector images
    #         else:
    #             img_comps = []
    #             for i in range(0, img.GetNumberOfComponentsPerPixel()):
    #                 img_slices_c = [sitk.VectorIndexSelectionCast(s, i)
    #                                 for s in img_slices]
    #                 img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
    #             img = sitk.Compose(img_comps)
    
    #     self.__showImage(img, title, margin, dpi)


    # def showImage(self, title=None, margin=0.05, dpi=80):
    #     self.__showImage(self.image_sitk, title=None, margin=0.05, dpi=80)

    # def __showImage(self, img, title=None, margin=0.05, dpi=80):
    #     nda = sitk.GetArrayFromImage(img)
    #     spacing = img.GetSpacing()
    
    #     if nda.ndim == 3:
    #         # fastest dim, either component or x
    #         c = nda.shape[-1]
    
    #         # the the number of components is 3 or 4 consider it an RGB image
    #         if c not in (3, 4):
    #             nda = nda[nda.shape[0] // 2, :, :]
    
    #     elif nda.ndim == 4:
    #         c = nda.shape[-1]
    
    #         if c not in (3, 4):
    #             raise RuntimeError("Unable to show 3D-vector Image")
    
    #         # take a z-slice
    #         nda = nda[nda.shape[0] // 2, :, :, :]
    
    #     xsize = nda.shape[1]
    #     ysize = nda.shape[0]
    
    #     # Make a figure big enough to accommodate an axis of xpixels by ypixels
    #     # as well as the ticklabels, etc...
    #     figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi
    
    #     plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
    #     ax = plt.gca()
    
    #     extent = (0, xsize * spacing[0], ysize * spacing[1], 0)
    
    #     t = ax.imshow(nda, extent=extent, interpolation=None)
    
    #     if nda.ndim == 2:
    #         t.set_cmap("gray")
    
    #     if(title):
    #         plt.title(title)
    
    #     plt.show()
        
    def showSlice(self, s, axis='CWH'):
        """Show image slice

        Parameters
        ----------
        s : int
            Slice number
        axis : str
            Order of axis e.g. 'CWH'
        """
        im = self.image(axis=axis)
        if s>=0 and s<im.shape[0]:
            plt.imshow(im[s])
        else:
            raise ValueError('Slice number not correct!')
        
        
    def extractRect(self, rect, axis='WHC'):
        """Extract patch from image using rectangle

        Parameters
        ----------
        rect : CTRect
            Rectangle ro crop a patch
        axis : str
            Order of axis e.g. 'WHC'
            
        Returns
        ----------
        CTPatch
            Croped patch
        """
        shape = self.image(axis=axis).shape
        if shape[0] >= rect.maxx and shape[1] >= rect.maxy and shape[2] >= rect.maxz:
            patch = CTPatch()
            patch.image_sitk = self.image_sitk[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz]
            patch.rect = rect
            return patch
        else:
            raise ValueError('CTRect is to big for the image!')

    def replaceValue(self, value_old, value_new): 
        """Replace value by another value in the image

        Parameters
        ----------
        value_old : Value which is replaced
            Old value which is replaced
        value_new : float
            New Value which replaces the old value
        """
        arr = self.image()
        arr[arr==value_old]=value_new
        self.setImage(arr)

        
    def maskToPhysicalPoint(self, mask, maskfilter=[]):
        """Transform pixel coodinates from mask with value specified in maskfilter to physical points

        Parameters
        ----------
        mask : ndarray
            Numpy array with mask information 
        maskfilter : list of float
            List of mask values which are trnasformed to physical points
            
        Returns
        ----------
        list of PhysicalPoint
            List of physicl points with values of mask
        """
        
        pointList=[]
        for maskf in maskfilter:
            idx = np.argwhere(mask == maskf)
            pixList=[]
            for i in range(idx.shape[0]):
                pixList.append((int(idx[i,1]), int(idx[i,0]), int(idx[i,2])))
            pointList = pointList + self.transformIndexToPhysicalPoint(pixList, label=maskf)
        return pointList

    def physicalPointToMask(self, pointList):
        """Transform pixel coodinates from PhysicalPoint to a mask

        Parameters
        ----------
        pointList : list of PhysicalPoint
            List of PhysicalPoint

        Returns
        ----------
        ndarray
            Mask of values from PhysicalPoints
        """
        
        pixListTrans = self.transformPhysicalPointToIndex(pointList)
        mask_trans = np.zeros(self.image('WHC').shape)
        for i, p in enumerate(pixListTrans):
            if p[0]>0 and p[0] < mask_trans.shape[0] and p[1]>0 and p[1] < mask_trans.shape[1] and p[2]>0 and p[2] < mask_trans.shape[2]:
                mask_trans[p[1], p[0], p[2]] = pointList[i].label
        return mask_trans

        
    def transform(self, transform, interpolation='FinalNearestNeighborInterpolator', default_value=0.0):
        """Transform image by transformation

        Parameters
        ----------
        transform : sitk.Transform
            Image transformation from sitk.Transformix library
        interpolation : str
            Interpolation after image transformation (default is 'FinalNearestNeighborInterpolator')
        default_value : float
            Default value for transformation (default is 0.0)

        Returns
        ----------
        CTImage
            Transformed image
        """
        for t in transform.parameterMapVector:
            t['ResampleInterpolator']=[interpolation]
        resampled = sitk.Transformix(self.image_sitk, transform.parameterMapVector)
        image_res = CTImage()
        image_res.image_sitk = resampled
        image_res.name = self.name
        return image_res
    
    def fillPatches(self, patches=[], stride=None, axis='CWH'):
        """Convert listof 3D patches to an aimage

        Parameters
        ----------
        patches : list of CTPatch
            List of CT patches
        stride : int
            Stride of the patches (default is None), If None, patach is set accorind to CTRect of the patch
        axis : str
            Order of axis e.g. 'WHC'

        Returns
        ----------
        CTImage
            Transformed image
        """
        if stride is None:  
            im = self.image(axis='WHC')
            for patch in patches:
                im[patch.rect.miny:patch.rect.maxy, patch.rect.minx:patch.rect.maxx, patch.rect.minz:patch.rect.maxz] = patch.image(axis='WHC')
        else:
            w = int(round((patches[0].image().shape[1] - stride[0]) / 2))
            h = int(round((patches[0].image().shape[2] - stride[1]) / 2))
            im = self.image(axis='WHC')
            for i, patch in enumerate(patches):
                pa = patch.image(axis='WHC')
                pa = pa[w:-w,h:-h,:]
                patch.rect.miny = patch.rect.miny+w
                patch.rect.maxy = patch.rect.maxy-w
                patch.rect.minx = patch.rect.minx+h
                patch.rect.maxx = patch.rect.maxx-h
                im[patch.rect.miny:patch.rect.maxy, patch.rect.minx:patch.rect.maxx, patch.rect.minz:patch.rect.maxz] = pa             
        self.setImage(im, axis='WHC')
        
    def getPatchList(self, patch_size=(64,64,1), stride=(64,64,1), rectROI=None, border=[64,64,64,64]):
        """Split imageinto pacthes

        Parameters
        ----------
        patch_size : tuple
            Tuple of patch size (axis=WHC)
        stride : tuple
            Stride of the patches (default is None), If None, stride is same as patch size
        rectROI : CTRect
            Rectangle that defines the ROI to extarct pacthes. If None, patches are extarcted from the entire image

        Returns
        ----------
        list of CTPatch
            List of patches
        """

        if stride is None:
            stride = patch_size
        if rectROI is None:
            shape = self.image(axis='WHC').shape
            rectROI = CTRect(0,0,0,shape[0],shape[1],shape[2])

        patchList=[]
        for i in range(rectROI.minx,  rectROI.maxx, stride[0]):
            for j in range(rectROI.miny, rectROI.maxy, stride[1]):
                for k in range(rectROI.minz, rectROI.maxz, stride[2]):
                    rect = CTRect(i, j, k, i + patch_size[0], j + patch_size[1], k + patch_size[2])
                    patch = self.extractRect(rect, axis='WHC')
                    if patch is not None:
                        patchList.append(patch)
        return patchList
    
    def getPatchList3D(self, patch_size=(128,128,128), stride=(64,64,64), rectROI=None, border=[128,128,128,128]):
        """Split imageinto pacthes

        Parameters
        ----------
        patch_size : tuple
            Tuple of patch size (axis=WHC)
        stride : tuple
            Stride of the patches (default is None), If None, stride is same as patch size
        rectROI : CTRect
            Rectangle that defines the ROI to extarct pacthes. If None, patches are extarcted from the entire image

        Returns
        ----------
        list of CTPatch
            List of patches
        """

        if stride is None:
            stride = patch_size
        if rectROI is None:
            shape = self.image(axis='CWH').shape
            sh0 = patch_size[0] + math.ceil((shape[0]-patch_size[0]) / stride[0]) * stride[0]
            sh1 = patch_size[1] + math.ceil((shape[1]-patch_size[1]) / stride[1]) * stride[1]
            sh2 = patch_size[2] + math.ceil((shape[2]-patch_size[2]) / stride[2]) * stride[2]
            #sh0 = math.ceil(patch_size[0] + shape[0] / stride[0]) * stride[0]
            #sh1 = math.ceil(shape[1] / stride[1]) * stride[1]
            #sh2 = math.ceil(shape[2] / stride[2]) * stride[2]
            arr = np.zeros((sh0, sh1, sh2))
            arr[0:shape[0], 0:shape[1], 0:shape[2]]=self.image(axis='CWH')
            rectROI = CTRect(0,0,0,sh0,sh1,sh2)
            

        patchList=[]
        for i in range(rectROI.minx,  rectROI.maxx-patch_size[0]+stride[0], stride[0]):
            for j in range(rectROI.miny, rectROI.maxy-patch_size[1]+stride[1], stride[1]):
                for k in range(rectROI.minz, rectROI.maxz-patch_size[2]+stride[2], stride[2]):
                    rect = CTRect(i, j, k, i + patch_size[0], j + patch_size[1], k + patch_size[2])
                    #patch = self.extractRect(rect, axis='WHC')
                    patch = CTPatch()
                    patch.setImage(arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz], axis='CWH')
                    #patch.image_sitk = sitk.GetImageFromArray(arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz])
                    #patch.image_sitk = arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz]
                    patch.rect = rect
                    #print(rect)
                    if patch is not None:
                        patchList.append(patch)
        return patchList

    @staticmethod
    def setPatchList3D(patchList, ref_tmp, stride=(64,64,64)):
        """Split imageinto pacthes

        Parameters
        ----------
        patch_size : tuple
            Tuple of patch size (axis=WHC)
        stride : tuple
            Stride of the patches (default is None), If None, stride is same as patch size
        rectROI : CTRect
            Rectangle that defines the ROI to extarct pacthes. If None, patches are extarcted from the entire image

        Returns
        ----------
        list of CTPatch
            List of patches
        """
        
        # patchList=patchListPred
        # ref = CTRef()
        # ref.setRef(np.zeros(image.image(axis='CWH').shape))
        shape = ref_tmp.ref().shape

        maxx=0
        maxy=0
        maxz=0
        for patch in patchList:
            maxx = max(maxx, patch.rect.maxx)
            maxy = max(maxy, patch.rect.maxy)
            maxz = max(maxz, patch.rect.maxz)
        maxc = patchList[0].image(axis='CWH').shape[0]
        arr = np.zeros((maxc, maxx, maxy,maxz))
        mask = np.zeros((maxc, maxx, maxy,maxz))
        for patch in patchList:
            p = patch.image(axis='CWH')
            arr[:,patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] = arr[:,patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] + p[:]
            mask[:,patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] = mask[:,patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] + np.ones(p.shape)
        arr = arr/mask
        
        arr=np.argmax(arr, axis=0)
        
        # maxx=0
        # maxy=0
        # maxz=0
        # for patch in patchList:
        #     maxx = max(maxx, patch.rect.maxx)
        #     maxy = max(maxy, patch.rect.maxy)
        #     maxz = max(maxz, patch.rect.maxz)
        # arr = np.zeros((maxx, maxy,maxz))
        # mask = np.zeros((maxx, maxy,maxz))
        # for patch in patchList:
        #     p = patch.image(axis='CWH')
        #     arr[patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] = arr[patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] + p
        #     mask[patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] = mask[patch.rect.minx:patch.rect.maxx, patch.rect.miny:patch.rect.maxy, patch.rect.minz:patch.rect.maxz] + np.ones(p.shape)
        # arr = arr/mask
        #arr = np.swapaxes(arr, 0, 2)
        #arr = np.swapaxes(arr, 1, 2)
            
        ref_tmp.setRef(arr[0:shape[0], 0:shape[1], 0:shape[2]])

        
        
        return ref_tmp
        

        # if stride is None:
        #     stride = patch_size
        # if rectROI is None:
        #     shape = self.image(axis='WHC').shape
        #     sh0 = math.ceil(shape[0] / stride[0]) * stride[0]
        #     sh1 = math.ceil(shape[1] / stride[1]) * stride[1]
        #     sh2 = math.ceil(shape[2] / stride[2]) * stride[2]
        #     arr = np.zeros((sh0, sh1, sh2))
        #     arr[0:shape[0], 0:shape[1], 0:shape[2]]=self.image(axis='WHC')
        #     rectROI = CTRect(0,0,0,sh0,sh1,sh2)
            

        # patchList=[]
        # for i in range(rectROI.minx,  rectROI.maxx, stride[0]):
        #     for j in range(rectROI.miny, rectROI.maxy, stride[1]):
        #         for k in range(rectROI.minz, rectROI.maxz, stride[2]):
        #             rect = CTRect(i, j, k, i + patch_size[0], j + patch_size[1], k + patch_size[2])
        #             #patch = self.extractRect(rect, axis='WHC')
        #             patch = CTPatch()
        #             patch.setImage(arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz], axis='WHC')
        #             #patch.image_sitk = sitk.GetImageFromArray(arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz])
        #             #patch.image_sitk = arr[rect.minx:rect.maxx, rect.miny:rect.maxy, rect.minz:rect.maxz]
        #             patch.rect = rect
        #             if patch is not None:
        #                 patchList.append(patch)
        # return patchList
    
    # @staticmethod
    # def drawMask(image, mask, axis='WHC'):
    #     image = np.float32(image)
    #     image = CTImage.swapaxes(image, axis_in=axis, axis_out='WHC')
    #     if image.shape[2]==1:
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
    #     image[:,:,0] = image[:,:,0] * (1-mask)
    #     image[:,:,1] = image[:,:,1] * (1-mask)
    #     image[:,:,2] = image[:,:,2] * (1-mask) + mask
    #     return image
            
            
class CTRef():
    """Create CT mask
    
    Attributes
    ----------
    refFormat : str
        Format of the mask:\n
        'norm'- normalized probabilities between 0.0 and 1.0 of shape (N,C,W,H)\n
        bin'- binary mask of shape (N,C,W,H) (one-hot) \n
        'num'- multi-class mask of shape (N,1,W,H) with integers for each class\n
    ref_sitk : sitk.Image
        SimpleITK image of the mask
    """
    
    #shape = None
    
    def __init__(self, fip_arr=None, refFormat='num'):
        """Init CTRef

        Parameters
        ----------
        fip : str, optional
            File path to the image that will be loaded immediately
        """
        self.refFormat = 'norm'
        self.ref_sitk=None
        if isinstance(fip_arr, np.ndarray):
            self.setRef(fip_arr, refFormat=refFormat)
        elif isinstance(fip_arr, torch.Tensor):
            self.setRef(fip_arr.numpy(), refFormat=refFormat)
        elif isinstance(fip_arr, str):
            self.load(fip_arr)
        else:
            pass

    def __mul__(self, arr_mult):
        """Multiply mask by an array (element wise)

        Parameters
        ----------
        arr_mult : ndarray
            Array that is multiplied by the image mask
        """
        arr = sitk.GetArrayFromImage(self.ref_sitk)
        if arr.shape==arr_mult.shape:
            arr = arr * arr_mult
            self.ref_sitk = sitk.GetImageFromArray(arr)
        else:
            raise ValueError('Image shape not equal to the array.')
        return self
    
    def normTobin(self):
        """Convert image mask from normalized probabilities to binary mask (one-hot)
        """
        if self.refFormat=='norm':
            arr = sitk.GetArrayFromImage(self.ref_sitk)
            arr = compute_one_hot_np(arr)
            self.ref_sitk = sitk.GetImageFromArray(arr)
            self.refFormat = 'bin'
        else:
            if self.refFormat=='bin':
                print('Reference format is already "bin"')
            else:
                raise ValueError('Image format has to be "norm"')
        return self
 
    # def normTobin_noclass(self):
    #     if self.refFormat=='norm':
    #         arr = sitk.GetArrayFromImage(self.ref_sitk)
    #         arr = compute_one_hot_noclass_np(arr)
    #         self.ref_sitk = sitk.GetImageFromArray(arr)
    #         self.refFormat = 'bin'
    #     else:
    #         if self.refFormat=='bin':
    #             print('Reference format is already "bin"')
    #         else:
    #             raise ValueError('Image format has to be "norm"')
    #     return self
    
    def binTonum(self):
        """Convert image mask from binary representation ((one-hot) to multi-class representation
        """
        if self.refFormat=='bin':
            arr = sitk.GetArrayFromImage(self.ref_sitk)
            arr_num = np.zeros((arr.shape[0],arr.shape[2], arr.shape[3]), dtype=np.int16)
            for i in range(arr.shape[1]):
                arr_num = arr_num + arr[:,i,:,:]*(i+1)
            self.ref_sitk = sitk.GetImageFromArray(arr_num)
            self.refFormat = 'num'
            self.shape = arr_num.shape
        else:
            if self.refFormat=='num':
                print('Reference format is already "num"')
            else:
                raise ValueError('Image format has to be "bin"')
            
        return self
    
    def numTobin(self, NumChannels, offset=1):
        """Convert image mask from multi-class representation to binary representation (one-hot)
        """
        if self.refFormat=='num':
            arr = sitk.GetArrayFromImage(self.ref_sitk)
            if arr.max()>NumChannels:
                raise ValueError('NumChannels must be greater or equal than the maxium value of the label.')
            arr_new = np.zeros((arr.shape[0], NumChannels, arr.shape[1], arr.shape[2]))
            for i in range(arr_new.shape[1]):
                arr_new[:,i,:,:] = arr==(i+offset)
            self.ref_sitk = sitk.GetImageFromArray(arr_new)
            self.refFormat = 'bin'
            self.shape = arr_new.shape
        else:
            if self.refFormat=='bin':
                print('Reference format is already "bin"')
            else:
                raise ValueError('Image format has to be "num"')
        return self
        
        
    def setRef(self, arr, refFormat='num'):
        """Set mask array

        Parameters
        ----------
        arr : ndarray
            Image mask
        refFormat : str
            Format of the mask:\n
            'norm'- normalized probabilities between 0.0 and 1.0 of shape (N,C,W,H)\n
            'bin'- binary mask of shape (N,C,W,H) (one-hot) \n
            'num'- multi-class mask of shape (N,1,W,H) with integers for each class\n
        """
        if refFormat=='norm' and arr.max()>1:
            raise ValueError('Image format "bin" requires values between [0,1]')
        if refFormat=='bin' and not arr.size == ((arr==0).sum() + (arr==1).sum()):
            raise ValueError('Image format "bin" requires binary values')
        ref_sitk_new = sitk.GetImageFromArray(arr)
        # Copy image information
        if self.ref_sitk:
            ref_sitk_new.CopyInformation(self.ref_sitk)
        # Set image
        self.ref_sitk = ref_sitk_new
        # Set refernce system which can be ('num', 'norm' or 'bin')
        self.refFormat = refFormat
        self.shape = arr.shape
        

    def ref(self, axis='CWH'):
        """Get mask ans ndrray

        Parameters
        ----------
        axis : str
            Order of axis e.g. 'WHC'

        Returns
        ----------
        ndarray
            Image mask as ndarray
        """
        ref = sitk.GetArrayFromImage(self.ref_sitk)
        ref = self.swapaxes(ref, 'CWH', axis)
        return ref
    
    @staticmethod
    def swapaxes(image, axis_in='WHC', axis_out='CWH'):
        """Swap axis
        
        Swap axis from axis_in to axis_out

        Parameters
        ----------
        axis_in : str
            Order of axis e.g. 'WHC' of the original image
        axis_out : str
            Order of axis e.g. 'CWH' of the target image

        Returns
        ----------
        ndarray
            Image with swaped axis
        """
        if axis_in=='WHC' and axis_out=='CWH':
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 1, 2)
        elif axis_in=='CWH' and axis_out=='WHC':
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 0, 1)
        elif axis_in=='CWH' and axis_out=='CWH':
            image = image
        elif axis_in=='WHC' and axis_out=='WHC':
            image = image
        elif axis_in=='WH' and axis_out=='WHC':
            image = np.expand_dims(image, axis=2)
        elif axis_in=='CWH' and axis_out=='HWC':
            image = np.swapaxes(image, 0, 2)
            image = image
        else:
            raise ValueError('Tranformation from ' + axis_in + + ' to ' + axis_out + ' does not exist.')
        return image
        
    # def setRefFormat(self, refFormat='num'):
    #     if self.refFormat=='num' and refFormat=='bin':
    #         arr = sitk.GetArrayFromImage(self.ref_sitk)
    #         #arr_new = np.array(arr.shape[0], arr.max(), arr.shape[2], arr.shape[3])
    #         arr_new = np.zeros((arr.shape[0], arr.max(), arr.shape[1], arr.shape[2]))
    #         for i in range(arr_new.shape[1]):
    #             arr_new[:,i,:,:] = arr==i
    #         ref_sitk_new = sitk.GetImageFromArray(arr_new)
    #         self.ref_sitk = ref_sitk_new
    #         self.refFormat='bin'
    #     elif self.refFormat=='bin' and refFormat=='num':
    #         pass
    
    # def showSlice(self, slice=0, channel=0):
    #     arr = self.ref()
    #     plt.imshow(arr[slice, channel])
        
    # def showImageJ(self, ImageJPath='/home/bernifoellmer/bin/Fiji.app/ImageJ-linux64'):
    #     # Create temporary folder and save image in tmp folder
    #     folderpath_tmp = os.path.join(CConfig['srcpath'], 'tmp_show')
    #     os.makedirs(folderpath_tmp, exist_ok=True)
    #     filepath_tmp = os.path.join(folderpath_tmp, 'tmp.mhd')
    #     self.save(filepath_tmp)
    #     # Open ImageJ and image
    #     command = ImageJPath + ' ' + filepath_tmp
    #     os.system(command)
    #     # Delete temporary folder
    #     if os.path.exists(folderpath_tmp):
    #         shutil.rmtree(folderpath_tmp)

    def copyInformationFrom(self, image):
        """Copy information

        Copy information from CTRef

        Parameters
        ----------
        image : CTRef
            CtRef with image information (orientation, spacing, etc.)
        """
        if isinstance(image, CTImage):
            self.ref_sitk.CopyInformation(image.image_sitk)  
        else:
            self.ref_sitk.CopyInformation(image.ref_sitk)  
        # try:        
        #     self.ref_sitk.CopyInformation(image.ref_sitk)               
        # except: 
        #     self.ref_sitk.CopyInformation(image.image_sitk)
            
        #self.ref_sitk.CopyInformation(image.ref_sitk)
        # if isinstance(image, CTImage):
        #     self.ref_sitk.CopyInformation(image.image_sitk)
        # else:
        #     self.ref_sitk.CopyInformation(image.ref_sitk)
    
    def load(self, path='H:/cloud/cloud_data/Projects/DL/Code/src/tmp/ct/VAV4P2CTI.mhd', image_format=None, refFormat='num'):
        """Load mask

        Load an image mask from file or folder.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        path : str, optional
            Filepath to the image or folderpath to the image
        image_format : str
            File fomat extension ('mhd', 'nrrd', 'png', 'dcm') if None, extension is guessed (default is None)
        refFormat : str
            Format of the mask:\n
            'norm'- normalized probabilities between 0.0 and 1.0 of shape (N,C,W,H)\n
            'bin'- binary mask of shape (N,C,W,H) (one-hot) \n
            'num'- multi-class mask of shape (N,1,W,H) with integers for each class\n
        """
        self.refFormat = refFormat
        
        if os.path.isdir(path):
            if image_format is None:
                file = glob(path + '/*')[0]
                _, _, ext = splitFilePath(file)
                image_format = ext[1:]
            
            # Reads the image from folder
            if image_format=='mhd':
                file = glob(path + '/*.mhd')[0]
                if os.path.exists(file):
                    self.ref_sitk = sitk.ReadImage(file)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                    self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            elif image_format=='nrrd':
                file = glob(path + '/*.nrrd')[0]
                if os.path.exists(file):
                    self.ref_sitk = sitk.ReadImage(file)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                    self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            elif image_format=='png':
                file = glob(path + '/*.png')[0]
                if os.path.exists(file):
                    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                    self.ref_sitk = sitk.GetImageFromArray(image)
                    _, filename, _ = splitFilePath(file)
                    self.name = filename
                    self.filepath = path
                    self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
                else:
                    raise ValueError('Filepath: ' + path + ' does not exist.')
            elif image_format=='dcm':
                if len(glob(path + '/*.dcm'))==1:
                    file = glob(path + '/*.dcm')[0]
                    if os.path.exists(file):
                        self.ref_sitk = sitk.ReadImage(file)
                        _, filename, _ = splitFilePath(file)
                        self.name = filename
                        self.filepath = path
                        self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
                    else:
                        raise ValueError('Filepath: ' + path + ' does not exist.')
                else:
                    filepathDcm = glob(path + '/*.dcm')[0]
                    file_reader = sitk.ImageFileReader()
                    file_reader.SetFileName(filepathDcm)
                    file_reader.ReadImageInformation()
                    series_ID = file_reader.GetMetaData('0020|000e')
                    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_ID)
                    data = sitk.ReadImage(sorted_file_names)
                    self.ref_sitk = data
                    self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
        else:
            # Reads the image using SimpleITK
            if os.path.exists(path):
                self.ref_sitk = sitk.ReadImage(path)
                _, filename, _ = splitFilePath(path)
                self.name = filename
                self.filepath = path
                self.shape = sitk.GetArrayFromImage(self.ref_sitk).shape
            else:
                raise ValueError('Filepath: ' + path + ' does not exist.')
                
    def save(self, path, image_format='mhd', dtype=sitk.sitkUInt8):
        """Save image mask to file

        Save an image mask to a file.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        folderpath : str
            Folderpath to the folder of the image mask
        image_format : str
            File fomat extension ('mhd', 'nrrd', 'png', 'nii')
        dtype : sitk Pixel Types
            Pixel Type of the image (e.g. sitk.sitkUInt16)
        """
        if dtype is not None:
            self.ref_sitk = sitk.Cast(self.ref_sitk, dtype)
        if image_format=='mhd' or image_format=='nrrd' or image_format=='nii' or image_format=='nii.gz':
            if path[-3:]=='mhd' or path[-4:]=='nrrd' or path[-3:]=='nii' or path[-6:]=='nii.gz':
                filepath = path
                self.ref_sitk = sitk.Cast(self.ref_sitk, dtype)
                sitk.WriteImage(self.ref_sitk, filepath, True)
            else:
                sitk.WriteImage(self.ref_sitk, os.path.join(path, self.name + '.' + image_format), True)
        elif image_format=='png':
            image = self.ref()
            cv2.imwrite(os.path.join(path, self.name + '.png'), image) 
        else:
            raise ValueError('Image format ' + image_format + ' is not implemented.')

            
    def resize(self, shape=(None, 256, 256), order=3):
        """Resize image
        
        Resize image to target shape

        Parameters
        ----------
        shape : tuple
            Shape of the target image
        order : int
            The order of the spline interpolation, default is 0 if
            image.dtype is bool and 1 otherwise. The order has to be in
            the range 0-5. See `skimage.transform.warp` for detail.
        """
        arr = sitk.GetArrayFromImage(self.ref_sitk)
        dtype = arr.dtype
        r0 = arr.shape[0] if shape[0] is None else shape[0]
        r1 = arr.shape[1] if shape[1] is None else shape[1]
        r2 = arr.shape[2] if shape[2] is None else shape[2]
        arr = resize(arr, (r0, r1, r2), order=order, preserve_range=True).astype(dtype)
        self.ref_sitk = sitk.GetImageFromArray(arr)
        
    # def save_png(self, folderpath, sliceNum=10, channel=0, color='R', filename=None, alpha=None):
    #     ref =self.ref()
    #     image = ref[sliceNum,channel,:,:]*255
    #     if filename is None:
    #         filename = self.name
    #     if color=='R':
    #         if alpha is None:
    #             imageC = cv2.merge((np.zeros((512,512), dtype=ref.dtype), np.zeros((512,512), dtype=ref.dtype), image))
    #         else:
    #             imageC = cv2.merge((np.zeros((512,512), dtype=ref.dtype), np.zeros((512,512), dtype=ref.dtype), image, alpha))
    #     if color=='G':
    #         if alpha is None:
    #             imageC = cv2.merge((np.zeros((512,512)), image, np.zeros((512,512))))
    #         else:
    #             imageC = cv2.merge((np.zeros((512,512)), np.zeros((512,512)), image, alpha))
    #     if color=='B':
    #         if alpha is None:
    #             imageC = cv2.merge(image, (np.zeros((512,512)), np.zeros((512,512))))
    #         else:
    #             imageC = cv2.merge((np.zeros((512,512)), np.zeros((512,512)), image, alpha))
    #     cv2.imwrite(os.path.join(folderpath, filename + '.png'), imageC) 

            
    # def saveChannel(self, folderpath, channel=0, image_format='mhd'):
    #     print('s0', self.ref().shape)
    #     scan_save = CTRef()
    #     scan_save.setRef(self.ref()[:,channel,:,:])
    #     print('s01', scan_save.shape)
    #     #scan_save.ref_sitk.CopyInformation(self.ref_sitk)
    #     #scan_save.name = self.name + '_' + str(channel)
    #     folderpathSpit, filenameSpit, fileExtSpit = splitFilePath(folderpath)
    #     folderpath = os.path.join(folderpathSpit, filenameSpit + '_' + str(channel) + '.' + image_format)
    #     print('folderpath123', folderpath)
    #     scan_save.save(folderpath, image_format=image_format)
    
    # def extractConnectedComponents(self, arr=None, min_size=None):
    #     """Extract connected components

    #     Parameters
    #     ----------
    #     axis_in : str
    #         Order of axis e.g. 'WHC' of the original image
    #     axis_out : str
    #         Order of axis e.g. 'CWH' of the target image

    #     Returns
    #     ----------
    #     ndarray
    #         Image with swaped axis
    #     """
    #     if arr is not None:
    #         ref_sitk = sitk.GetImageFromArray(arr)
    #     else:
    #         ref_sitk = self.ref_sitk
            
    #     # Filter small objects
    #     if min_size is not None:
    #         ref_sitk = sitk.GetImageFromArray((skimage.morphology.remove_small_objects(sitk.GetArrayFromImage(ref_sitk), min_size=min_size)))
        
    #     compFilter = ConnectedComponentImageFilter()
    #     labeled_sitk = compFilter.Execute(ref_sitk)
    #     labeled = sitk.GetArrayFromImage(labeled_sitk)
    #     return labeled
    
    # def extractConnectedComponents(self, arr=None, min_size=None):
    #     """Extract connected components

    #     Parameters
    #     ----------
    #     axis_in : str
    #         Order of axis e.g. 'WHC' of the original image
    #     axis_out : str
    #         Order of axis e.g. 'CWH' of the target image

    #     Returns
    #     ----------
    #     ndarray
    #         Image with swaped axis
    #     """
    #     if arr is not None:
    #         ref_sitk = sitk.GetImageFromArray(arr)
    #     else:
    #         ref_sitk = self.ref_sitk
            
    #     # Filter small objects
    #     if min_size is not None:
    #         ref_sitk = sitk.GetImageFromArray((skimage.morphology.remove_small_objects(sitk.GetArrayFromImage(ref_sitk), min_size=min_size)))
        
    #     compFilter = ConnectedComponentImageFilter()
    #     labeled_sitk = compFilter.Execute(ref_sitk)
    #     labeled = sitk.GetArrayFromImage(labeled_sitk)
    #     return labeled

    def replaceValue(self, value_old, value_new): 
        """Replace value by another value in the image

        Parameters
        ----------
        value_old : Value which is replaced
            Old value which is replaced
        value_new : float
            New Value which replaces the old value
        """
        arr = self.ref()
        arr[arr==value_old]=value_new
        self.setRef(arr)
        

            
# class CTRefObj():
#     """Create object
#     """
    
#     def __init__(self, fip=None):
#         """ Init CTRefObj
#         """
#         self.idx = np.array((), dtype=np.int16)
    
            
class PhysicalPoint():
    """Create physical point


    Attributes
    ----------
    coordinates : tuple
        Coordinates of the physical point (x,y,z)
    label : int
        Label of the physical point
    props : dict
        Additional properties of the physical point
    """
    
    coordinates = (0.0, 0.0, 0.0)
    label = None
    props = defaultdict(lambda: None, {})
    
    def __init__(self, coordinates=(0.0, 0.0, 0.0)):
        """Init PhysicalPoint
    
        Parameters
        ----------
        coordinates : tuple
            X, Y and Z coordinate e.g. (0.0,0.0,0.0)
        """
        self.coordinates = coordinates
        
    def scale(self, scale=(1,1,1)):
        """Scale the PhysicalPoint in all three diections
    
        Parameters
        ----------
        scale : tuple
            Scaling of the three directions X, Y and Z coordinate e.g. (0.5,1.0,1.0)
        """
        
        self.coordinates = (self.coordinates[0]*scale[0], self.coordinates[1]*scale[1], self.coordinates[2]*scale[2])
        
    @staticmethod
    def pointListToArray(pointList):
        """Convert a list of physical points into a numpy array with size (len(pointList) x 3)
    
        Parameters
        ----------
        pointList : list of PhysicalPoints
            List of physical points
            
        Returns
        -------
        ndarray
            3 x N array of point coordinates
        """
        
        points=[]
        for p in pointList:
            points.append(p.coordinates)
        return np.array(points)
    
    
class PixelPoint():
    """Create pixel point


    Attributes
    ----------
    coordinates : tuple
        Coordinates of the pixel point (x,y,z)
    label : int
        Label of the pixel point
    props : dict
        Additional properties of the physical point
    """
    
    
    coordinates = (0, 0, 0)
    label = None
    props = defaultdict(lambda: None, {})
    
    def __init__(self, coordinates=(0, 0, 0)):
        """Init PixelPoint
    
        Parameters
        ----------
        coordinates : tuple
            X, Y and Z coordinate e.g. (0,0,0)
        """
        
        self.coordinates = coordinates
        
    def scale(self, scale=(1,1,1)):
        """Scale the PhysicalPoint in all three diections
    
        Parameters
        ----------
        scale : tuple
            Scaling of the three directions X, Y and Z coordinate e.g. (0.5,1.0,1.0)
        """
        self.coordinates = (int(round(self.coordinates[0]*scale[0])), int(round(self.coordinates[1]*scale[1])), int(round(self.coordinates[2]*scale[2])))
        
    @staticmethod
    def pointListToArray(pointList):
        """Convert a list of pixel points into a numpy array with size (len(pointList) x 3)
    
        Parameters
        ----------
        pointList : list of PixelPoint
            List of PixelPoint
            
        Returns
        -------
        ndarray
            3 x N array of point coordinates
        """
        points=[]
        for p in pointList:
            points.append(p.coordinates)
        return np.array(points)    


class CTTransform():
    """Image transformation class

    Attributes
    ----------
    parameterMapVector : list of parameterMap
        ID of the patch
    name : str
        name of the pacth
    rect : CTRect
        Rectangle that defines the pacth position in the image
    """
    def __init__(self):
        """ Init CTImage
        """
        self.parameterMapVector = []
        #self.FromCTImage = None
        #self.ToCTImage = None
    
    def save(self, folderpath, filename):
        for i, trans in enumerate(self.parameterMapVector):
            filepath = os.path.join(folderpath, filename + '_' + str(i) + '.txt')
            sitk.WriteParameterFile(trans, filepath)
       
    def load(self, folderpath, filename=''):
        if os.path.isdir(folderpath):
            transFiles = glob(os.path.join(folderpath,filename + '*.txt'))
            for trans in transFiles:
                self.parameterMapVector.append(sitk.ReadParameterFile(trans))
        else:
            trans = folderpath
            self.parameterMapVector.append(sitk.ReadParameterFile(trans))            


class CTPatch(CTImage):
    """Patch of CT image

    Attributes
    ----------
    PatchID : int
        ID of the patch
    name : str
        name of the pacth
    rect : CTRect
        Rectangle that defines the pacth position in the image
    """
    
    def __init__(self, fip=None):
        """Init CTPatch
    
        Parameters
        ----------
        fip : str
            Filepath of the patch
        """
        super().__init__(fip)   
        self.PatchID = None                                     #: Id of the patch
        self.name = ''                                      #: Name of the patch
        self.rect = CTRect()

    def save(self, folderpath, image_format='mhd', dtype=None):
        """Save patch to file

        Load an image from file or folder.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        folderpath : str,
            Path to the folder of the patch
        image_format : str
            File fomat extension ('mhd', 'nrrd', 'png', 'nii')
        dtype : sitk Pixel Types
            Pixel Type of the image (e.g. sitk.sitkUInt16)
        """
        
        if dtype is not None:
            self.image_sitk = sitk.Cast(self.image_sitk, dtype)
        if image_format=='mhd':
            sitk.WriteImage(self.image_sitk, os.path.join(folderpath, self.name + '.mhd'), True)
        elif image_format=='nrrd':
            data = self.image(axis='HWC')
            nrrd.write(os.path.join(folderpath, self.name + '.nrrd'), data)
            #sitk.WriteImage(self.image_sitk, os.path.join(folderpath, self.name + '.nrrd')) 
        elif image_format=='png':
            image = self.image()
            cv2.imwrite(os.path.join(folderpath, self.name + '.png'), image) 
        else:
            raise ValueError('Image format ' + image_format + ' is not implemented.')
            
    @staticmethod
    def patchListToArray(patchList, axis='CWH'):
        """Convert list of pacthes to ndarray

        Load an image from file or folder.
        Supported file formats: dcm, mhd, nrrd, png

        Parameters
        ----------
        patchList : list of CTPatch,
            List of patches
        axis : str
            Order of axis e.g. 'WHC'
            
        Returns
        -------
        ndarray
            N x [axis] array of pacthes
        """
        shape = patchList[0].image().shape
        l = len(patchList)
        X = np.ones((l, shape[0], shape[1], shape[2]), np.float32)
        for i, patch in enumerate(patchList):
            X[i,:] = patch.image(axis=axis)
        return X

    @staticmethod
    def ArraytoPatchList(X, axis='CWH', patchListDefault=None):
        """Convert array of patches to list of pacthes

        Parameters
        ----------
        X : ndarray
            Array of patches
        axis : str
            Order of axis e.g. 'WHC'
        patchListDefault : list of CTPatch (optional)
            Predefined list of patches. Array is copied to the patches
            
        Returns
        -------
        list of CTPatch
            List of patches
        """
        if patchListDefault:
            patchList=[]
            for i in range(X.shape[0]):
                patch = CTPatch()
                patch.setImage(X[i,:])
                patch.rect = patchListDefault[i].rect
                patchList.append(patch)
        else:
            for i, patch in enumerate(patchList):
                patchList[i] = patch.setImage(X[i,:])
        return patchList        

    
class CTRect:
    """Rectangle

    Attributes
    ----------
    minx : int
        Minimal pixel position in x direction
    miny : int
        Minimal pixel position in y direction
    minz : int
        Minimal pixel position in z direction
    maxx : int
        Maximum pixel position in x direction
    maxy : int
        Maximum pixel position in y direction
    maxz : int
        Maximum pixel position in z direction
    """
    def __init__(self, minx=None, miny=None, minz=None, maxx=None, maxy=None, maxz=None):
        """Init rectangle

        Parameters
        ----------
        minx : int
            Minimal pixel position in x direction
        miny : int
            Minimal pixel position in y direction
        minz : int
            Minimal pixel position in z direction
        maxx : int
            Maximum pixel position in x direction
        maxy : int
            Maximum pixel position in y direction
        maxz : int
            Maximum pixel position in z direction
        """
        self.minx = minx                                        #: minx [px]
        self.miny = miny                                        #: miny [px]
        self.minz = minz                                        #: minz [px]
        self.maxx = maxx                                        #: maxx [px]
        self.maxy = maxy                                        #: maxy [px]
        self.maxz = maxz                                        #: maxz [px]
        
    def __str__(self):
        s = 'minx:'+str(self.minx)
        s = s+'\nminy:'+str(self.minz)
        s = s+'\nminz:'+str(self.miny)
        s = s+'\nmaxx:'+str(self.maxx)
        s = s+'\nmaxy:'+str(self.maxy)
        s = s+'\nmaxz:'+str(self.maxz)
        return s 
        
        
    def scale(self, ratio):
        """Scale rectangle 

        Parameters
        ----------
        ratio : float
            Sacling ration in all directions
            
        Returns
        -------
        CTRect
            Scaled rectangle
        """
        minx = int(np.around(self.minx*ratio))
        miny = int(np.around(self.miny*ratio))
        minz = int(np.around(self.minz*ratio))
        maxx = int(np.around(self.maxx*ratio))
        maxy = int(np.around(self.maxy*ratio))
        maxz = int(np.around(self.maxz*ratio))
        rect_scale = CTRect(minx, miny, maxx, maxy, minz, maxz)
        return rect_scale


# class CTSeries():
    
#     image = None
#     SeriesInstanceUID = ''
#     header = ''
    
#     def __init__(self, image, header, SeriesInstanceUID):
#         """ Init CTPatient
#         """
        
#         self.image = image
#         self.header = header
#         self.SeriesInstanceUID = SeriesInstanceUID
    
# class CTPatient():
#     """ CTPatient
#     """

#     StudyInstanceUID = None
#     series = None
#     image_CACS = None
#     PatientID = None
        
#     def __init__(self, StudyInstanceUID='', PatientID='', series=None):
#         """ Init CTPatient
#         """
        
#         self.StudyInstanceUID = StudyInstanceUID
#         self.PatientID = PatientID
#         self.series = series
#         self.image_CACS = CTImage()
        
#     def getSeriesInstanceUIDCACS(self, folderpath):
#         cacs = self.series[self.series['CACS'] == True]
#         SeriesInstanceUIDList = list(cacs.index)
#         return SeriesInstanceUIDList
    
#     def loadSeries(self, folderpath, SeriesInstanceUID, SOPInstanceUID):
#         folderpath_dcm = os.path.join(folderpath, self.StudyInstanceUID, SeriesInstanceUID)
#         filepathList = glob(folderpath_dcm + '/*.dcm')
        
#         # Read data
#         if len(filepathList)>0:
#             filepath = filepathList[0]
#             # Read the file's meta-information without reading bulk pixel data
#             file_reader = sitk.ImageFileReader()
#             file_reader.SetFileName(filepath)
#             file_reader.ReadImageInformation()
#             series_ID = file_reader.GetMetaData('0020|000e')
#             sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folderpath_dcm, series_ID)

#             # Check MultiSlice
#             ds = pydicom.dcmread(filepathList[0], force = False, defer_size = 256, specific_tags = ['NumberOfFrames'], stop_before_pixels = True)
#             try:        
#                 NumberOfFrames = ds.data_element('NumberOfFrames').value                            
#             except: 
#                 NumberOfFrames=''
# 				
#             if (not NumberOfFrames=='') and NumberOfFrames>1:
#                 MultiSlice = True   
#             else:
#                 MultiSlice = False
               
#             # Read the bulk pixel data
#             if MultiSlice:
#                 data = sitk.ReadImage(filepathList[0])
#             else:
#                 data = sitk.ReadImage(sorted_file_names)
#             image = CTImage()
#             image.image_sitk = data
#             header = 0
#             self.image_CACS=image
            
#             # Create series
#             series = CTSeries(image, header, SeriesInstanceUID)
#             return series
#         else:
#             return None
    
#     def open3DSlicer(self, folderpath, StudyInstanceUID, SeriesInstanceUID):
#         filepath3DSlicer = 'H:/ProgramFiles/Slicer 4.10.2/Slicer.exe'
#         folderpath_dcm = os.path.join(folderpath, StudyInstanceUID, SeriesInstanceUID)
#         filepath = glob(folderpath_dcm + '/*.dcm')[0]
#         filepath = filepath.replace("\\", '/')
#         command_default = """SLICER --python-code "slicer.util.loadVolume('FILEPATH', returnNode=True)"""
#         command_default = command_default.replace('SLICER', filepath3DSlicer)
#         command = command_default.replace('FILEPATH', filepath)
#         process = subprocess.Popen(command)
        
#     @staticmethod
#     def getPatientByStudyInstanceUID(patientsList, StudyInstanceUID):
#         for p in patientsList:
#             if p.StudyInstanceUID == StudyInstanceUID:
#                 return p

#     @staticmethod
#     def getPatientByPatientID(patientsList, PatientID):
#         for p in patientsList:
#             if p.PatientID == PatientID:
#                 return p
            
#     @staticmethod
#     def createPatientsFromList(filepath, sheet='linear'):
        
#         print('Reading excel sheet:', filepath)
#         df = pd.read_excel(filepath, sheet) 
#         PatientIDList = list(df['PatientID'].unique())
#         patientsList=[]
#         for i, ID in enumerate(PatientIDList[0:10]):
#             # Select patient data
#             series = df[df['PatientID']==ID]
#             series = series.set_index('SeriesInstanceUID')
#             # Create patient
#             StudyInstanceUID = series['StudyInstanceUID'][0]
#             patient = CTPatient(StudyInstanceUID, PatientID=ID, series=series)
#             patientsList.append(patient)
            
#         return patientsList
