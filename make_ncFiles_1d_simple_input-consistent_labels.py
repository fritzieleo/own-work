#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys,argparse,re,ocrolib,codecs
#from pylab import imread, imshow,gray,show
from numpy import array, where, zeros, ones,sort,shape,amax
import scipy.misc
import Image
from PIL import Image,ImageOps
import netcdf_helpers
from ocrolib import lineest
import urdu_dict_1d

#Read Dictionary
urdu_dict = urdu_dict_1d.urdu_dict


##Make class labels and print 
labels = []
for index in range(len(urdu_dict.keys())):
  c = urdu_dict.keys()[index]
  labels.append(c.encode('utf-8',"backslashreplace"))
print "# Total Labels:",len(labels)


inputs = []
seqLengths = []
seqDims = []
targetStrings = []
seqTags = []

def readFeatures(filename):
  """Read features from the image files"""
  #Open Image
  #image = Image.open(filename).transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270).convert('L') 
  #print image.size
  image = Image.open(filename).convert('L') 
  #Normalize using OCRopus normalizer
  imagea = ocrolib.pil2array(image)
  lnorm.measure(amax(imagea)-imagea)
  try:    
    imagea = lnorm.normalize(imagea,cval=amax(imagea))
  except (ZeroDivisionError,ValueError): 
    print 'Bad image, removing'
  
  #Read Image frames (1-column width window of heigh equal to that of the image)
  for w in range(imagea.shape[1]):
    v = imagea[:,w]
    inputs.append(v)
  
  #seqLengths is the width of the image and seqDims is the dimensions of the frames
  seqLengths.append(imagea.shape[1])
  seqDims.append([seqLengths[-1]])
  
      
def readTranscript(txt_file_name):
  """Read transcription from the ground-truth file"""
  
  transcriptFile=codecs.open(txt_file_name,'r','utf-8')
  line= transcriptFile.readline().rstrip()
  line = line.replace(' ','|') # Encoded space with |
  line = line[::-1] #Urdu is read from right to left
  transcript = ''
  if any((c.encode('utf-8','backslashreplace') not in labels) for c in line):
    print '=== ',txt_file_name, ' contains foreign symbols'
    print "the ground truth is ===========", line
    return ''
  else:  
    for c in line:
	  transcript += c.encode('utf-8','backslashreplace')+ ' '
    return transcript.rstrip()
  
  

parser = argparse.ArgumentParser("Generate NC File for Urdu-1D-NoPos experiments")
parser.add_argument("-l","--height",default=48,type=int)
parser.add_argument("-t","--notargets",action="store_true")
parser.add_argument("-o","--ncFileName",default=None)
parser.add_argument("files",nargs="*")
args = parser.parse_args()

#Read all file names
images = ocrolib.glob_all(args.files)
if len(images)==0:
  parser.print_help()
  sys.exit(0)
  
print "# Input images:",len(images)


# load the OCRopus line normalizer (center)
print '# Loading OCRopus line normalizer'
lnorm = lineest.CenterNormalizer()
lnorm.setHeight(args.height)

counter = 0
print '# Reading %d Text-line images and corresponding transcription'%len(images)

for fname in images: 
  if len(images) >= 1000 and counter == int(0.5*len(images)): print 'Hang on, I have read half of the images'
  
  base,_ = ocrolib.allsplitext(fname)
  
  targetString = readTranscript(base+'.gt.txt')
  #print 'targetString:',targetString
  if targetString == '':
    print 'removing the image'
    continue
  
  targetStrings.append(targetString)  
  
  
  seqTags.append(base)
  readFeatures(fname)
  
  counter += 1
  


#create a new .nc file
print '# Creating NetCDFFile:',args.ncFileName

ncfile =netcdf_helpers.NetCDFFile(args.ncFileName, 'w')

#create the dimensions
netcdf_helpers.createNcDim(ncfile,'numSeqs',len(seqLengths))
netcdf_helpers.createNcDim(ncfile,'numTimesteps',len(inputs))
netcdf_helpers.createNcDim(ncfile,'inputPattSize',len(inputs[0]))
netcdf_helpers.createNcDim(ncfile,'numDims',1)
netcdf_helpers.createNcDim(ncfile,'numLabels',len(labels))

#create the variables
netcdf_helpers.createNcStrings(ncfile,'labels',labels,('numLabels','maxLabelLength'),'labels')
netcdf_helpers.createNcStrings(ncfile,'targetStrings',targetStrings,('numSeqs','maxTargStringLength'),'target strings')
netcdf_helpers.createNcStrings(ncfile,'seqTags',seqTags,('numSeqs','maxSeqTagLength'),'sequence tags')
netcdf_helpers.createNcVar(ncfile,'seqLengths',seqLengths,'i',('numSeqs',),'sequence lengths')
netcdf_helpers.createNcVar(ncfile,'seqDims',seqDims,'i',('numSeqs','numDims'),'sequence dimensions')
netcdf_helpers.createNcVar(ncfile,'inputs',inputs,'f',('numTimesteps','inputPattSize'),'input patterns')
  
#write the data to disk
print "writing data to", args.ncFileName
ncfile.close()
