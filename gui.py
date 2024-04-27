from utils import *
from MLThreadworkers import *
from Remote import *
from dataManipulation import *

from ttkthemes import ThemedTk
import tkinter as tk
import tkinter.ttk as ttk
from pygubu.widgets.pathchooserinput import PathChooserInput
from connectomics.config import *
import yaml
import os
from os.path import isdir,sep
from os import listdir,mkdir,getcwd
import h5py
from PIL import Image, ImageSequence
import skimage.io as skio
from skimage.color import label2rgb
import threading
from multiprocessing import Process
import random
import traceback
import json
import numpy as np
from connectomics.utils.process import binary_watershed,bc_watershed

import matplotlib as mpl
defaultMatplotlibBackend = mpl.get_backend()
defaultMatplotlibBackend = 'TkAgg'
from matplotlib import pyplot as plt

		
def writeH5(filename, dtarray, datasetname='vol0'): 
	fid=h5py.File(filename,'w')
	if isinstance(datasetname, (list,)):
		for i,dd in enumerate(datasetname):
			ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
			ds[:] = dtarray[i]
	else:
		ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
		ds[:] = dtarray
	fid.close()
	del dtarray

class TabguiApp():
	def __init__(self, master=None):
		self.root = master
		self.RefreshVariables(firstTime=True)

		self.tabHolder = ttk.Notebook(self.root,height=700)
		self.tabHolder.pack(fill='none',padx=3,pady=3,ipadx=1,ipady=1,side='top')

		self.frameDatacheck = ttk.Frame(self.tabHolder)

		self.labelDataImages = ttk.Label(self.frameDatacheck)
		self.labelDataImages.configure(text='Original Images(.tif):')
		self.labelDataImages.grid(column='0',row='0',sticky='w')

		self.pathchooserinputDataImages = PathChooserInput(self.frameDatacheck)
		self.pathchooserinputDataImages.configure(type='file')
		self.pathchooserinputDataImages.grid(column='1',row='0',sticky="e")

		self.labelDataLabel = ttk.Label(self.frameDatacheck)
		self.labelDataLabel.configure(text='Labelled Images(.tif):')
		self.labelDataLabel.grid(column='0', row='1',sticky="w")

		self.pathchooserinputDataLabel = PathChooserInput(self.frameDatacheck)
		self.pathchooserinputDataLabel.configure(type='file')
		self.pathchooserinputDataLabel.grid(column='1', row='1',sticky="e")

		self.labelDataglancerURL = ttk.Label(self.frameDatacheck, foreground="blue", cursor="hand2")
		self.labelDataglancerURL.configure(text="")
		self.labelDataglancerURL.grid(column="0", row="2", columnspan="2")

		self.buttonDataOpen = ttk.Button(self.frameDatacheck)
		self.buttonDataOpen.configure(text='Launch Neuroglancer')
		self.buttonDataOpen.grid(column='0', row='3', columnspan="2")
		self.buttonDataOpen.configure(command=self.openNeuroGlancer1)

		self.frameDatacheck.grid_columnconfigure(0, weight=1)
		self.frameDatacheck.grid_columnconfigure(1, weight=1)

		self.tabHolder.add(self.frameDatacheck, text="Data Check")


		self.frameTrainMaster = ttk.Frame(self.tabHolder)
		
		canvas = tk.Canvas(self.frameTrainMaster,borderwidth=0)
		vsb = ttk.Scrollbar(self.frameTrainMaster, orient="vertical", command=canvas.yview)

		self.frameTrain = ttk.Frame(canvas)
		
		vsb.pack(side="right", fill="y")                                      
		canvas.pack(side="left", fill="both", expand=True)               
		canvas.create_window((0, 0), window=self.frameTrain,anchor='nw')
		canvas.configure(yscrollcommand=vsb.set)
		def on_mousewheel(event):
			canvas.yview_scroll(-1*(event.delta//120), "units")
		canvas.bind_all("<MouseWheel>", on_mousewheel)

		def on_frame_configure(event):
			canvas.configure(scrollregion=canvas.bbox("all"))
		canvas.bind("<Configure>", on_frame_configure)

		self.pathChooserTrainImageStack = PathChooserInput(self.frameTrain)
		self.pathChooserTrainImageStack.configure(type='file')
		self.pathChooserTrainImageStack.grid(column='1',row='0',sticky='e')

		self.pathChooserTrainLabels = PathChooserInput(self.frameTrain)
		self.pathChooserTrainLabels.configure(type='file')
		self.pathChooserTrainLabels.grid(column='1',row='1',sticky='e')

		self.label1 = ttk.Label(self.frameTrain)
		self.label1.configure(text='Image Stack(.tif or .h5):')
		self.label1.grid(column='0', row='0',sticky='w')
		self.label2 = ttk.Label(self.frameTrain)
		self.label2.configure(text='Labels(.tif or .h5):')
		self.label2.grid(column='0', row='1',sticky='w')
		
		self.configChooserVariable = tk.StringVar(master)
		self.configChooserSelect = ttk.OptionMenu(self.frameTrain, self.configChooserVariable, None, *self.configs)
		self.configChooserSelect.grid(column='1', row='2',sticky='e')

		self.labelConfig = ttk.Label(self.frameTrain)
		self.labelConfig.configure(text='Training Config:')
		self.labelConfig.grid(column='0', row='2',sticky='w')
		
		self.labelTrainX = ttk.Label(self.frameTrain)
		self.labelTrainX.configure(text='X nm/pixel:')
		self.labelTrainX.grid(column='0', row='3',sticky='w')
		self.labelTrainY = ttk.Label(self.frameTrain)
		self.labelTrainY.configure(text='Y nm/pixel:')
		self.labelTrainY.grid(column='0', row='4',sticky='w')
		self.labelTrainZ = ttk.Label(self.frameTrain)
		self.labelTrainZ.configure(text='Z nm/pixel:')
		self.labelTrainZ.grid(column='0', row='5',sticky='w')

		self.entryTrainX = ttk.Entry(self.frameTrain)
		self.entryTrainX.grid(column='1', row='3',sticky='e')
		self.entryTrainY = ttk.Entry(self.frameTrain)
		self.entryTrainY.grid(column='1', row='4',sticky='e')
		self.entryTrainZ = ttk.Entry(self.frameTrain)
		self.entryTrainZ.grid(column='1', row='5',sticky='e')

		self.label4 = ttk.Label(self.frameTrain)
		self.label4.configure(text='# GPU:')
		self.label4.grid(column='0', row='6',sticky='w')

		self.label5 = ttk.Label(self.frameTrain)
		self.label5.configure(text='# CPU:')
		self.label5.grid(column='0', row='7',sticky='w')

		self.numBoxTrainGPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainGPU.configure(from_='0', increment='1', to='200')
		self.numBoxTrainGPU.delete('0', 'end')
		self.numBoxTrainGPU.insert('0', "1")
		self.numBoxTrainGPU.grid(column='1', row='6',sticky='e')

		self.numBoxTrainCPU = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainCPU.configure(from_='1', increment='1', to='500')
		self.numBoxTrainCPU.delete('0', 'end')
		self.numBoxTrainCPU.insert('0', "1")
		self.numBoxTrainCPU.grid(column='1', row='7',sticky='e')

		self.label17 = ttk.Label(self.frameTrain)
		self.label17.configure(text='Base LR:')
		self.label17.grid(column='0', row='8',sticky='w')

		self.label18 = ttk.Label(self.frameTrain)
		self.label18.configure(text='Iteration Step:')
		self.label18.grid(column='0', row='9',sticky='w')
		self.label19 = ttk.Label(self.frameTrain)
		self.label19.configure(text='Iteration Save:')
		self.label19.grid(column='0', row='10',sticky='w')
		self.label20 = ttk.Label(self.frameTrain)
		self.label20.configure(text='Iteration Total:')
		self.label20.grid(column='0', row='11',sticky='w')
		self.label21 = ttk.Label(self.frameTrain)
		self.label21.configure(text='Samples Per Batch:')
		self.label21.grid(column='0', row='12',sticky='w')

		self.numBoxTrainBaseLR = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainBaseLR.configure(increment='.0001', to='1')
		_text_ = '''.0001'''
		self.numBoxTrainBaseLR.delete('0', 'end')
		self.numBoxTrainBaseLR.insert('0', _text_)
		self.numBoxTrainBaseLR.grid(column='1', row='8',sticky='e')
		
		self.numBoxTrainIterationStep = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationStep.configure(from_='1', increment='1', to='100')
		_text_ = '''1'''
		self.numBoxTrainIterationStep.delete('0', 'end')
		self.numBoxTrainIterationStep.insert('0', _text_)
		self.numBoxTrainIterationStep.grid(column='1', row='9',sticky='e')

		self.numBoxTrainIterationSave = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationSave.configure(from_='1', increment='1000', to='1000000')
		_text_ = '''5000'''
		self.numBoxTrainIterationSave.delete('0', 'end')
		self.numBoxTrainIterationSave.insert('0', _text_)
		self.numBoxTrainIterationSave.grid(column='1', row='10',sticky='e')

		self.numBoxTrainIterationTotal = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainIterationTotal.configure(from_='1', increment='5000', to='10000000')
		_text_ = '''100000'''
		self.numBoxTrainIterationTotal.delete('0', 'end')
		self.numBoxTrainIterationTotal.insert('0', _text_)
		self.numBoxTrainIterationTotal.grid(column='1', row='11',sticky='e')

		self.numBoxTrainSamplesPerBatch = ttk.Spinbox(self.frameTrain)
		self.numBoxTrainSamplesPerBatch.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainSamplesPerBatch.delete('0', 'end')
		self.numBoxTrainSamplesPerBatch.insert('0', _text_)
		self.numBoxTrainSamplesPerBatch.grid(column='1', row='12',sticky='e')

		self.label_new1 = ttk.Label(self.frameTrain)
		self.label_new1.configure(text='Window Size:')
		self.label_new1.grid(column='0', row='13',sticky='w')

		self.entryWindowSize = ttk.Entry(self.frameTrain)
		self.entryWindowSize.delete('0', 'end')
		self.entryWindowSize.grid(column='1', row='13',sticky='e')

		self.label25 = ttk.Label(self.frameTrain)
		self.label25.configure(text='Name Model as:')
		self.label25.grid(column='0', row='14',sticky='w')

		self.entryTrainModelName = ttk.Entry(self.frameTrain)
		self.entryTrainModelName.grid(column='1', row='14',sticky='e')

		self.buttonTrainTrain = ttk.Button(self.frameTrain)
		self.buttonTrainTrain.configure(text='Train')
		self.buttonTrainTrain.grid(column='0', columnspan='2', row='15', pady=10)
		self.buttonTrainTrain.configure(command=self.trainTrainButtonPress)

		self.textTrainOutput = tk.Text(self.frameTrain,relief='solid')
		self.textTrainOutput.configure(height=10, width=50)
		_text_ = '''Training Progress Will Show Here'''
		self.textTrainOutput.insert('0.0', _text_)
		self.textTrainOutput.grid(column='0',columnspan='2',row='16',padx=5,pady=5)

		self.frameTrain.columnconfigure(0, weight=1)
		self.frameTrain.rowconfigure(0, weight=1)

		self.tabHolder.add(self.frameTrainMaster, text='Train')


		self.framePredict = ttk.Frame(self.tabHolder)

		self.pathChooserUseImageStack1= ttk.Label(self.framePredict)
		self.pathChooserUseImageStack1.configure(text='Image Stack(.tif or .h5):')
		self.pathChooserUseImageStack1.grid(column='0',row='0',sticky="w")

		self.pathChooserUseImageStack = FileChooser(self.framePredict, labelText='', mode='open')
		self.pathChooserUseImageStack.grid(column='1',row='0',sticky="e")

		self.pathChooserUseOutputFile1= ttk.Label(self.framePredict)
		self.pathChooserUseOutputFile1.configure(text='Output File:')
		self.pathChooserUseOutputFile1.grid(column='0',row='1',sticky="w")

		self.pathChooserUseOutputFile = FileChooser(self.framePredict, labelText='', mode='create')
		self.pathChooserUseOutputFile.grid(column='1',row='1',sticky="e")

		self.modelChooserVariable = tk.StringVar(master)
		self.modelChooserSelect = ttk.OptionMenu(self.framePredict, self.modelChooserVariable, None, *self.models)
		self.modelChooserSelect.grid(column='1', row='2',sticky="e")

		self.labelPredictModelChooser = ttk.Label(self.framePredict)
		self.labelPredictModelChooser.configure(text='Model to Use: ')
		self.labelPredictModelChooser.grid(column='0', row='2',sticky="w")

		self.entryUsePadSize = ttk.Entry(self.framePredict)
		_text_ = '''0,0,0'''
		self.entryUsePadSize.delete('0', 'end')
		self.entryUsePadSize.insert('0', _text_)
		self.entryUsePadSize.grid(column='1', row='6',sticky="e")

		self.entryUseAugMode = ttk.Entry(self.framePredict)
		_text_ = "'mean'"
		self.entryUseAugMode.delete('0', 'end')
		self.entryUseAugMode.insert('0', _text_)
		self.entryUseAugMode.grid(column='1', row='7',sticky="e")

		self.entryUseAugNum = ttk.Entry(self.framePredict)
		_text_ = '''None'''
		self.entryUseAugNum.delete('0', 'end')
		self.entryUseAugNum.insert('0', _text_)
		self.entryUseAugNum.grid(column='01', row='8',sticky="e")

		self.entryUseStride = ttk.Entry(self.framePredict)
		_text_ = '''1,128,128'''
		self.entryUseStride.delete('0', 'end')
		self.entryUseStride.insert('0', _text_)
		self.entryUseStride.grid(column='1', row='9',sticky="e")

		self.numBoxUseSamplesPerBatch = ttk.Spinbox(self.framePredict)
		self.numBoxUseSamplesPerBatch.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxUseSamplesPerBatch.delete('0', 'end')
		self.numBoxUseSamplesPerBatch.insert('0', _text_)
		self.numBoxUseSamplesPerBatch.grid(column='1', row='10',sticky="e")

		self.label29 = ttk.Label(self.framePredict)
		self.label29.configure(text='Pad Size')
		self.label29.grid(column='0', row='6',sticky="w")
		self.label30 = ttk.Label(self.framePredict)
		self.label30.configure(text='Aug Mode: ')
		self.label30.grid(column='0', row='7',sticky="w")
		self.label31 = ttk.Label(self.framePredict)
		self.label31.configure(text='Aug Num: ')
		self.label31.grid(column='0', row='8',sticky="w")
		self.label33 = ttk.Label(self.framePredict)
		self.label33.configure(text='Stride: ')
		self.label33.grid(column='0', row='9',sticky="w")
		self.label32 = ttk.Label(self.framePredict)
		self.label32.configure(text='Samples Per Batch: ')
		self.label32.grid(column='0', row='10',sticky="w")

		self.buttonUseLabel = ttk.Button(self.framePredict)
		self.buttonUseLabel.configure(text='Label')
		self.buttonUseLabel.grid(column='0', columnspan='2', row='11')
		self.buttonUseLabel.configure(command=self.UseModelLabelButtonPress)

		self.textUseOutput = tk.Text(self.framePredict,relief='solid',height=15)
		_text_ = '''Labelling Progress Will Show here'''
		self.textUseOutput.insert('0.0', _text_)
		self.textUseOutput.grid(column='0', columnspan='2', row='12',padx=5,pady=5,sticky='nsew')

		self.buttonNeuroInverts2 = ttk.Button(self.framePredict)
		self.buttonNeuroInverts2.configure(text="Semantic2D Post-Process")
		self.buttonNeuroInverts2.grid(column='0', row='13', columnspan="1")
		self.buttonNeuroInverts2.configure(command=self.semantic2dProcessor)

		self.buttonNeuroInverts3 = ttk.Button(self.framePredict)
		self.buttonNeuroInverts3.configure(text="Semantic3D Post-Process")
		self.buttonNeuroInverts3.grid(column='0', row='14', columnspan="1")
		self.buttonNeuroInverts3.configure(command=self.semantic3dProcessor)

		self.buttonNeuroInverti2 = ttk.Button(self.framePredict)
		self.buttonNeuroInverti2.configure(text="Instance2D Post-Process")
		self.buttonNeuroInverti2.grid(column='1', row='13', columnspan="1")
		self.buttonNeuroInverti2.configure(command=self.instance2dProcessor)	

		self.buttonNeuroInverti3 = ttk.Button(self.framePredict)
		self.buttonNeuroInverti3.configure(text="Instance3D Post-Process")
		self.buttonNeuroInverti3.grid(column='1', row='14', columnspan="1")
		self.buttonNeuroInverti3.configure(command=self.instance3dProcessor)

		self.tabHolder.add(self.framePredict, text='Auto Label')


		self.frameImage = ttk.Frame(self.tabHolder)

		self.fileChooserImageToolsInput1= ttk.Label(self.frameImage)
		self.fileChooserImageToolsInput1.configure(text='Folder to Combine(.tif files):')
		self.fileChooserImageToolsInput1.grid(column='0',row='0',sticky="w")

		self.fileChooserImageToolsInput = FileChooser(self.frameImage, labelText='', mode='folder', title='Folder with Files To Combine', buttonText='Choose Folder')
		self.fileChooserImageToolsInput.grid(column='1', row='0',sticky="e")

		self.fileChooserImageToolsOutput1= ttk.Label(self.frameImage)
		self.fileChooserImageToolsOutput1.configure(text='Output File:')
		self.fileChooserImageToolsOutput1.grid(column='0',row='1',sticky="w")

		self.fileChooserImageToolsOutput = FileChooser(self.frameImage, labelText='', mode='create', title='Output Filename', buttonText='Choose File')
		self.fileChooserImageToolsOutput.grid(column='1', row='1',sticky="e")

		self.buttonImageCombineTif = ttk.Button(self.frameImage)
		self.buttonImageCombineTif.configure(text='Combine Into TIF')
		self.buttonImageCombineTif.grid(column='0',columnspan='2',row='2')
		self.buttonImageCombineTif.configure(command=self.ImageToolsCombineImageButtonPressTif)

		self.buttonImageCombineTxt = ttk.Button(self.frameImage)
		self.buttonImageCombineTxt.configure(text='Combine Into TXT')
		self.buttonImageCombineTxt.grid(column='0',columnspan='2',row='3')
		self.buttonImageCombineTxt.configure(command=self.ImageToolsCombineImageButtonPressTxt)

		self.buttonImageCombineJson = ttk.Button(self.frameImage)
		self.buttonImageCombineJson.configure(text='Combine Into JSON')
		self.buttonImageCombineJson.grid(column='0',columnspan='2',row='4')
		self.buttonImageCombineJson.configure(command=self.ImageToolsCombineImageButtonPressJson)

		self.textImageTools = tk.Text(self.frameImage,relief='solid',height=15)
		_text_ = '''Image Tools Output Will Be Here'''
		self.textImageTools.insert('0.0', _text_)
		self.textImageTools.grid(column='0',columnspan='2',row='5',padx=5,pady=5,sticky='nsew')

		self.tabHolder.add(self.frameImage, text='Image Tools')


		self.frameOutputTools = ttk.Frame(self.tabHolder)

		self.fileChooserOutputStats1= ttk.Label(self.frameOutputTools)
		self.fileChooserOutputStats1.configure(text='Model Output(.h5):')
		self.fileChooserOutputStats1.grid(column='0',row='0',sticky="w")

		self.fileChooserOutputStats = FileChooser(master=self.frameOutputTools,labelText='',changeCallback=False, mode='open', title='', buttonText='Choose File')
		self.fileChooserOutputStats.grid(column='1',row='0',sticky="e")

		self.fileChooserOutputToolsOutCSV1= ttk.Label(self.frameOutputTools)
		self.fileChooserOutputToolsOutCSV1.configure(text='CSV Output:')
		self.fileChooserOutputToolsOutCSV1.grid(column='0',row='1',sticky="w")

		self.fileChooserOutputToolsOutCSV = FileChooser(master=self.frameOutputTools,labelText='',changeCallback=False,mode='create',title='',buttonText='Choose File')
		self.fileChooserOutputToolsOutCSV.grid(column='1',row='1',sticky="e")

		self.fileChooserOutputToolsOutCSVseparator = ttk.Separator(self.frameOutputTools)
		self.fileChooserOutputToolsOutCSVseparator.configure(orient='horizontal')
		self.fileChooserOutputToolsOutCSVseparator.grid(column='0', columnspan='2', row='2',sticky="ew")

		self.checkbuttonOutputMeshs = ttk.Checkbutton(self.frameOutputTools)
		self.checkbuttonOutputMeshs.configure(text='Meshs')
		self.checkbuttonOutputMeshs.grid(column='0',columnspan='2',row='3')

		self.checkbuttonOutputPointClouds = ttk.Checkbutton(self.frameOutputTools)
		self.checkbuttonOutputPointClouds.configure(text='Point Clouds')
		self.checkbuttonOutputPointClouds.grid(column='0',columnspan='2',row='4')

		self.buttonOutputMakeGeometries = ttk.Button(self.frameOutputTools)
		self.buttonOutputMakeGeometries.configure(text='Make Geometries')
		self.buttonOutputMakeGeometries.grid(column='0', columnspan='2', row='5')
		self.buttonOutputMakeGeometries.configure(command=self.OutputToolsMakeGeometriesButtonPress)

		self.labelDownscaleGeometry = ttk.Label(self.frameOutputTools)
		self.labelDownscaleGeometry.configure(text='Downscaling Factor:')
		self.labelDownscaleGeometry.grid(column='0',row='6',sticky="w")

		self.entryDownscaleGeometry = ttk.Entry(self.frameOutputTools)
		self.entryDownscaleGeometry.insert('0',"1")
		self.entryDownscaleGeometry.grid(column='1',row='6',sticky="e")

		self.buttonOutputMakeGeometriesseparator = ttk.Separator(self.frameOutputTools)
		self.buttonOutputMakeGeometriesseparator.configure(orient='horizontal')
		self.buttonOutputMakeGeometriesseparator.grid(column='0', columnspan='2', row='7',sticky="ew")

		self.buttonOutputGetStats = ttk.Button(self.frameOutputTools)
		self.buttonOutputGetStats.configure(text='Get Model Output Stats')
		self.buttonOutputGetStats.grid(column='0', columnspan='2', row='8')
		self.buttonOutputGetStats.configure(command=self.OutputToolsModelOutputStatsButtonPress)

		self.textOutputOutput = tk.Text(self.frameOutputTools,relief='solid',height=15)
		_text_ = '''Analysis Output Will Be Here'''
		self.textOutputOutput.insert('0.0', _text_)
		self.textOutputOutput.grid(column='0',columnspan='2',row='9',padx=5,pady=5,sticky='nsew')

		self.tabHolder.add(self.frameOutputTools, text='Output Tools')


		self.frameNeuroGlancer = ttk.Frame(self.tabHolder)

		self.labelNeuroImages = ttk.Label(self.frameNeuroGlancer)
		self.labelNeuroImages.configure(text='Raw Images(.tif): ')
		self.labelNeuroImages.grid(column='0', row='0',sticky='ew')

		self.pathchooserinputNeuroImages = PathChooserInput(self.frameNeuroGlancer)
		self.pathchooserinputNeuroImages.configure(type='file')
		self.pathchooserinputNeuroImages.grid(column='1', row='0',sticky='ew')

		self.labelNeuroLabel = ttk.Label(self.frameNeuroGlancer)
		self.labelNeuroLabel.configure(text='Model Output(.h5): ')
		self.labelNeuroLabel.grid(column='0', row='1',sticky='ew')

		self.pathchooserinputNeuroLabel = PathChooserInput(self.frameNeuroGlancer)
		self.pathchooserinputNeuroLabel.configure(type='file')
		self.pathchooserinputNeuroLabel.grid(column='1', row='1',sticky='ew')

		self.labelNeuroLabelX = ttk.Label(self.frameNeuroGlancer)
		self.labelNeuroLabelX.configure(text='X Scale(Horizontal along one image)(nm): ')
		self.labelNeuroLabelX.grid(column='0', row='2',sticky='ew')

		self.entryNeuroX = ttk.Entry(self.frameNeuroGlancer)
		self.entryNeuroX.configure()
		self.entryNeuroX.grid(column='1', row='2',sticky='ew')

		self.labelNeuroLabelY = ttk.Label(self.frameNeuroGlancer)
		self.labelNeuroLabelY.configure(text='Y Scale(Vertical along one image)(nm): ')
		self.labelNeuroLabelY.grid(column='0', row='3',sticky='ew')

		self.entryNeuroY = ttk.Entry(self.frameNeuroGlancer)
		self.entryNeuroY.configure()
		self.entryNeuroY.grid(column='1', row='3',sticky='ew')

		self.labelNeuroLabelZ = ttk.Label(self.frameNeuroGlancer)
		self.labelNeuroLabelZ.configure(text='Z Scale(Each image slice moves along the Z axis)(nm): ')
		self.labelNeuroLabelZ.grid(column='0', row='4',sticky='ew')

		self.entryNeuroZ = ttk.Entry(self.frameNeuroGlancer)
		self.entryNeuroZ.configure()
		self.entryNeuroZ.grid(column='1', row='4',sticky='ew')

		self.labelNeuroglancerURL = ttk.Label(self.frameNeuroGlancer, foreground="blue", cursor="hand2")
		self.labelNeuroglancerURL.configure(text="")
		self.labelNeuroglancerURL.grid(column="0", row="6", columnspan="2")

		self.buttonNeuroOpen = ttk.Button(self.frameNeuroGlancer)
		self.buttonNeuroOpen.configure(text='Launch Neuroglancer')
		self.buttonNeuroOpen.grid(column='0', row='7', columnspan="2")
		self.buttonNeuroOpen.configure(command=self.openNeuroGlancer2)

		self.frameNeuroGlancer.grid_columnconfigure(0, weight=1)
		self.frameNeuroGlancer.grid_columnconfigure(1, weight=1)

		self.tabHolder.add(self.frameNeuroGlancer, text="Neuroglancer")

		print("Width x Height", master.winfo_width(), 'x', master.winfo_height())

	def openNeuroGlancer1(self):
		imagefilepath=self.pathchooserinputDataImages.entry.get()
		modelOutputFilePath=self.pathchooserinputDataLabel.entry.get()
		self.neuroglancerThread = threading.Thread(target=openNeuroGlancerThread, args=(imagefilepath, modelOutputFilePath, self.labelDataglancerURL,(1,1,1),'pre'))
		self.neuroglancerThread.setDaemon(True)
		self.neuroglancerThread.start()

	def openNeuroGlancer2(self):
		imagefilepath=self.pathchooserinputNeuroImages.entry.get()
		modelOutputFilePath=self.pathchooserinputNeuroLabel.entry.get()
		x = int(self.entryNeuroX.get())
		y = int(self.entryNeuroY.get())
		z = int(self.entryNeuroZ.get())

		self.neuroglancerThread = threading.Thread(target=openNeuroGlancerThread, args=(imagefilepath, modelOutputFilePath, self.labelNeuroglancerURL,(z,y,x),'post'))
		self.neuroglancerThread.setDaemon(True)
		self.neuroglancerThread.start()

	def semantic2dProcessor(self):
		modelOutputFilePath=self.pathChooserUseOutputFile.entry.get()
		f = h5py.File(modelOutputFilePath, "r")
		post_arr=np.array(f['vol0'])
		f.close()
		del f
		print('\n',post_arr.shape)
		post_arr=np.invert(post_arr)

		Recombine=[]
		for layer in post_arr[0]:
			new_layer=np.expand_dims(layer, axis=0)
			new_layer=binary_watershed(new_layer,thres1=0.8,thres2=0.85, thres_small=1024,seed_thres=35)
			Recombine.append(new_layer)
		
		post_arr=np.stack(Recombine, axis=0)
		del Recombine
		print('after combine',post_arr.shape)
		post_arr=np.expand_dims(post_arr, axis=0)
		print(post_arr.shape)
		writeH5(modelOutputFilePath+'_s2D_out',np.array(post_arr))
		del post_arr
		print("Finished Semantic2D Process! Please find the 'Model Output' with its original name + _s2D_out")

	def semantic3dProcessor(self):  
		modelOutputFilePath=self.pathChooserUseOutputFile.entry.get()
		f = h5py.File(modelOutputFilePath, "r")
		post_arr=np.array(f['vol0'][:2])
		f.close()
		del f
		print('\n',post_arr.shape)

		post_arr=bc_watershed(post_arr,thres1=0.9,thres2=0.8,thres3=0.8,thres_small=1024,seed_thres=35)
		post_arr=np.expand_dims(post_arr, axis=0)
		print(post_arr.shape)

		writeH5(modelOutputFilePath+'_s3D_out',np.array(post_arr))
		del post_arr
		print("Finished Semantic3D Process! Please find the 'Model Output' with its original name + _s3D_out")

	def instance2dProcessor(self):
		modelOutputFilePath=self.pathChooserUseOutputFile.entry.get()
		
		f = h5py.File(modelOutputFilePath, "r")
		post_arr=np.array(f['vol0'])
		f.close()
		del f
		print('\n',post_arr.shape)
		Recombine=[]
		for layer in post_arr[0]:
			new_layer=np.expand_dims(layer, axis=0)
			new_layer=bc_watershed(new_layer,thres1=0.9,thres2=0.8,thres3=0.8,thres_small=1024,seed_thres=35)
			Recombine.append(new_layer)
		
		post_arr=np.stack(Recombine, axis=0)
		del Recombine
		print('after combine',post_arr.shape)
		post_arr=np.expand_dims(post_arr, axis=0)
		print(post_arr.shape)
		writeH5(modelOutputFilePath+'_i2D_out',np.array(post_arr))
		del post_arr
		print("Finished Instance2D Process! Please find the 'Model Output' with its original name + _i2D_out")

	def instance3dProcessor(self):  
		modelOutputFilePath=self.pathChooserUseOutputFile.entry.get()
		
		f = h5py.File(modelOutputFilePath, "r")
		post_arr=np.array(f['vol0'][:2])
		f.close()
		del f
		print('\n',post_arr.shape)
		post_arr = bc_watershed(post_arr, thres1=0.85, thres2=0.6, thres3=0.8, thres_small=1024)
		print(post_arr.shape)
		writeH5(modelOutputFilePath+'_i3D_out',np.expand_dims(post_arr, axis=0))
		del post_arr
		print("Finished Instance Process! Please find the 'Model Output' with its original name + _i3D_out")

	def closeNeuroGlancer(self):
		self.labelNeuroglancerURL.configure(text="")
		closeNeuroglancerThread()

	def visualizeOpenButtonPress(self):
		from visualizationGUI import runVisualizationWindow
		runVisualizationWindow()

	def longButtonPressHandler(self, thread, memStream, textBox, listToReEnable, refreshTime=1000):
		textBox.delete(1.0,"end")
		textBox.insert("end", memStream.text)
		textBox.see('end')

		if thread.is_alive():
			self.root.after(refreshTime, lambda: self.longButtonPressHandler(thread, memStream, textBox, listToReEnable, refreshTime))
		else:
			for element in listToReEnable:
				element['state'] = 'normal'
			self.RefreshVariables()
			
	def trainTrainButtonPress(self):
		self.buttonTrainTrain['state'] = 'disabled'
		try:
			image = self.pathChooserTrainImageStack.entry.get()
			labels = self.pathChooserTrainLabels.entry.get()

			configToUse = self.configChooserVariable.get() 	

			sizex = int(self.entryTrainX.get())
			sizey = int(self.entryTrainY.get())
			sizez = int(self.entryTrainZ.get())

			gpuNum = int(self.numBoxTrainGPU.get())
			cpuNum = int(self.numBoxTrainCPU.get())
			lr = float(self.numBoxTrainBaseLR.get())
			itStep = int(self.numBoxTrainIterationStep.get())
			itSave = int(self.numBoxTrainIterationSave.get())
			itTotal = int(self.numBoxTrainIterationTotal.get())
			samples = int(self.numBoxTrainSamplesPerBatch.get())
			windowSize = self.entryWindowSize.get()

			name = self.entryTrainModelName.get()
			cluster=0

			with open('Data' + sep + 'configs' + sep + configToUse,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)
				file.close()

			config['SYSTEM']['NUM_GPUS'] = gpuNum
			config['SYSTEM']['NUM_CPUS'] = cpuNum

			config['MODEL']['INPUT_SIZE'] = [int(s) for s in windowSize.split(',')]
			config['MODEL']['OUTPUT_SIZE'] = [int(s) for s in windowSize.split(',')]
			config['INFERENCE']['INPUT_SIZE'] = [int(s) for s in windowSize.split(',')]
			config['INFERENCE']['OUTPUT_SIZE'] = [int(s) for s in windowSize.split(',')]

			config['DATASET']['IMAGE_NAME'] = image
			config['DATASET']['LABEL_NAME'] = labels
			config['DATASET']['OUTPUT_PATH'] = getcwd() + sep + 'Data' + sep + 'models' + sep + name + sep
			config['SOLVER']['BASE_LR'] = lr
			config['SOLVER']['ITERATION_STEP'] = itStep
			config['SOLVER']['ITERATION_SAVE'] = itSave
			config['SOLVER']['ITERATION_TOTAL'] = int(itTotal) + 1
			config['SOLVER']['SAMPLES_PER_BATCH'] = samples

			if not isdir('Data' + sep + 'models' + sep + name):
				mkdir('Data' + sep + 'models' + sep + name)

			with open("Data" + sep + "models" + sep + name + sep + "config.yaml", 'w') as file:
				yaml.dump(config, file)
				file.close()

			metaDictionary = {}
			metaDictionary['configType'] = configToUse
			metaDictionary['x_scale'] = sizex
			metaDictionary['y_scale'] = sizey
			metaDictionary['z_scale'] = sizez
			with open("Data" + sep + "models" + sep + name + sep + "metadata.yaml", 'w') as file:
				yaml.dump(metaDictionary, file)
				file.close()

			if cluster==0:
				memStream = MemoryStream()
				t = threading.Thread(target=trainThreadWorker, args=("Data" + sep + "models" + sep + name + sep + "config.yaml", memStream))
				t.setDaemon(True)
				t.start()
				self.longButtonPressHandler(t, memStream, self.textTrainOutput, [self.buttonTrainTrain])
		except:
			self.buttonTrainTrain['state'] = 'normal'
			traceback.print_exc()

	def getConfigForModel(self, model): 
		return "Data" + sep + "models" + sep + model + sep + "config.yaml" 

	def getLastCheckpointForModel(self, model):
		checkpointFiles = os.listdir('Data' + sep + 'models' + sep + model)
		biggestCheckpoint = 0

		for subFile in checkpointFiles:
			if not subFile[-8:] == '.pth.tar':
				continue
			try:
				checkpointNumber = int(subFile.split('_')[1][:-8])
			except:
				raise Exception(subFile, 'file unable to be parsed in getLastCheckpointForModel')
			if checkpointNumber > biggestCheckpoint:
				biggestCheckpoint = checkpointNumber
		biggestCheckpoint = 'Data' + sep + 'models' + sep + model + sep + 'checkpoint_' + str(biggestCheckpoint).zfill(5) + '.pth.tar'
		return biggestCheckpoint

	def getMetadataForModel(self, model):
		with open('Data' + sep + 'models' + sep + model + sep + 'metadata.yaml','r') as file:
			metaData = yaml.load(file, Loader=yaml.FullLoader)
			file.close()
		metaDataStr = str(metaData)
		print('Metadata For Model:', model, '|', metaDataStr)
		return metaDataStr

	def UseModelLabelButtonPress(self):
		self.buttonUseLabel['state'] = 'disabled'
		recombine = False
		try:
			cluster = 0
			model = self.modelChooserVariable.get()
			gpuNum = int(self.numBoxTrainGPU.get())
			cpuNum = int(self.numBoxTrainCPU.get())
			samples = int(self.numBoxUseSamplesPerBatch.get())
			image = self.pathChooserUseImageStack.entry.get()
			outputFile = self.pathChooserUseOutputFile.entry.get()

			padSize = self.entryUsePadSize.get()
			padSize = [int(s) for s in padSize.split(',')]
			augMode = self.entryUseAugMode.get()
			augNum = 'None'
			stride = self.entryUseStride.get()
			stride = [int(s) for s in stride.split(',')]

			configToUse = self.getConfigForModel(model)
			with open(configToUse,'r') as file:
				config = yaml.load(file, Loader=yaml.FullLoader)
				file.close()

			config['SYSTEM']['NUM_GPUS'] = gpuNum
			config['SYSTEM']['NUM_CPUS'] = cpuNum

			outputPath, outputName = os.path.split(outputFile)

			if not outputName.split('.')[-1] == 'h5':
				outputName += '.h5'

			config['INFERENCE']['OUTPUT_PATH'] = outputPath
			config['INFERENCE']['OUTPUT_NAME'] = outputName
			config['INFERENCE']['IMAGE_NAME'] = image
			config['INFERENCE']['SAMPLES_PER_BATCH'] = samples
			config['INFERENCE']['STRIDE'] = stride
			config['INFERENCE']['AUG_MODE'] = augMode
			config['INFERENCE']['AUG_NUM'] = augNum
			config['INFERENCE']['PAD_SIZE'] = padSize

			with open('temp.yaml','w') as file:
				yaml.dump(config, file)
				file.close()

			if cluster==0:
				biggestCheckpoint = self.getLastCheckpointForModel(model)
				metaData = self.getMetadataForModel(model)
				memStream = MemoryStream()
				t = threading.Thread(target=useThreadWorker, args=('temp.yaml', memStream, biggestCheckpoint, metaData, recombine))
				t.setDaemon(True)
				t.start()
				self.longButtonPressHandler(t, memStream, self.textUseOutput, [self.buttonUseLabel])
		except:
			traceback.print_exc()
			self.buttonUseLabel['state'] = 'normal'

	def ImageToolsCombineImageButtonPressTif(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineTif['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-4:] == '.tif':
				outputFile = outputFile + '.tif'

			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineTif])
		except:
			traceback.print_exc()
			self.buttonImageCombineTif['state'] = 'normal'

	def ImageToolsCombineImageButtonPressTxt(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineTxt['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-4:] == '.txt':
				outputFile = outputFile + '.txt'
				
			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineTxt])
		except:
			traceback.print_exc()
			self.buttonImageCombineTxt['state'] = 'normal'

	def ImageToolsCombineImageButtonPressJson(self):
		try:
			memStream = MemoryStream()
			self.buttonImageCombineJson['state'] = 'disabled'
			filesToCombine = self.fileChooserImageToolsInput.getFilepath()
			outputFile = self.fileChooserImageToolsOutput.getFilepath()
			if not outputFile[-5:] == '.json':
				outputFile = outputFile + '.json'
				
			t = threading.Thread(target=ImageToolsCombineImageThreadWorker, args=(filesToCombine, outputFile, memStream))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textImageTools, [self.buttonImageCombineJson])
		except:
			traceback.print_exc()
			self.buttonImageCombineJson['state'] = 'normal'

	def OutputToolsModelOutputStatsButtonPress(self):
		try:
			memStream = MemoryStream()
			self.buttonOutputGetStats['state'] = 'disabled'
			filename = self.fileChooserOutputStats.getFilepath() 
			csvfilename = self.fileChooserOutputToolsOutCSV.getFilepath()
			if not csvfilename[-4:] == '.csv' and len(csvfilename) > 0:
				csvfilename += '.csv'
			t = threading.Thread(target=OutputToolsGetStatsThreadWorker, args=(filename, memStream, csvfilename))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textOutputOutput, [self.buttonOutputGetStats])
		except:
			traceback.print_exc()
			self.buttonOutputGetStats['state'] = 'normal'

	def OutputToolsMakeGeometriesButtonPress(self):
		try:
			memStream = MemoryStream()
			self.buttonOutputMakeGeometries['state'] = 'disabled'
			h5Path = self.fileChooserOutputStats.getFilepath()
			makeMeshs = self.checkbuttonOutputMeshs.instate(['selected'])
			makePoints = self.checkbuttonOutputPointClouds.instate(['selected'])
			downScaleFactor = self.entryDownscaleGeometry.get()
			if len(downScaleFactor.strip()) ==len(''):
				downScaleFactor = 1
			else:
				downScaleFactor = int(downScaleFactor)
			t = threading.Thread(target=OutputToolsMakeGeometriesThreadWorker, args=(h5Path, makeMeshs, makePoints, memStream, downScaleFactor))
			t.setDaemon(True)
			t.start()
			self.longButtonPressHandler(t, memStream, self.textOutputOutput, [self.buttonOutputMakeGeometries])
		except:
			traceback.print_exc()
			self.buttonOutputMakeGeometries['state'] = 'normal'

	def RefreshVariables(self, firstTime=False):
		configs = sorted(listdir('Data' + sep + 'configs'))
		for file in configs:
			if not file[-5:] == '.yaml':
				configs.remove(file)
		configs = list(sorted(configs))
		self.configs = configs

		modelList = []
		if not os.path.isdir('Data' + sep + 'models'):
			os.makedirs('Data' + sep + 'models')
		models = listdir('Data' + sep + 'models')
		for model in models:
			if os.path.isdir('Data' + sep + 'models' + sep + model):
				modelList.append(model)
		if len(modelList) == 0:
			modelList.append('No Models Yet')
		modelList = list(sorted(modelList))
		self.models = modelList

		if not firstTime:
			self.modelChooserSelect.set_menu('None', *self.models)
			self.configChooserSelect.set_menu('None', *self.configs)

	def run(self):
		self.root.mainloop()


from tkinter import font

if __name__ == '__main__':
	mpl.use('Agg')
	root = ThemedTk(theme='breeze')

	root.option_add("*font", "sans-serif 11")
	root.title("Anatomics MLT 2024")
	root.minsize(width=750, height=700)
	root.maxsize(width=750, height=700)

	sp = os.getcwd()
	imgicon = tk.PhotoImage(file=os.path.join(sp,'icon.png'))
	root.tk.call('wm', 'iconphoto', root._w, imgicon)
	root.geometry('750x700')

	try:
		from torch.cuda import is_available,get_device_name
		cuda_check=is_available()
		if cuda_check==True:
			print("CUDA is available:", cuda_check)
			print("Using Cuda Device:",get_device_name(0))
		else:
			print("\nWARNING! GPU is NOT in use")
	except:
		pass

	app = TabguiApp(root)
	app.run()
