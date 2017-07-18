import os, glob, sys
import editdistance
import argparse

parser = argparse.ArgumentParser(description = """
Compute the edit distances between ground truth and recognizer output.
Run with the ground truth files as arguments, and it will find the
corresponnding recognizer output files using the given extension (-x).
Missing output files are handled as empty strings.
""")
#parser.add_argument("files",default=[],nargs='*',help="input lines")
parser.add_argument("-p","--path", default=".", help="path for the ground truth and output files")
parser.add_argument("-g","--groundtruth_extension",default=".gt.txt",help="extension for ground truth, default: %(default)s")
parser.add_argument("-x","--extension",default=".txt",help="extension for recognizer output, default: %(default)s")
args = parser.parse_args()


gt_path = os.path.join(args.path, '*{0}'.format(args.groundtruth_extension))
gt_files = glob.glob(gt_path)

if len(gt_files) < 1 and args.path is not '.':
	sys.stderr.write("No ground truth file found with the extension...\n")
	sys.stderr.write("Make sure that the given path contains your ground truth, as well as output files with the correct extensions.\n")

my_dict = {}
for gt_file in sorted(gt_files):
	gtFile = open(gt_file)
	for gt in file(gt_file).readlines():
		gt = gt.decode("utf-8")
		#prediction = prediction[::-1] #reverse the labels 
		break
	
	out_file = gt_file.replace(args.groundtruth_extension, args.extension)

	if not os.path.isfile(out_file):
		sys.stderr.write("{0} output file not found!!\n".format(out_file))
		ed = len(gt)
	else:
		outFile = open(out_file)	
		for out in file(out_file).readlines():
			out = out.decode("utf-8")
			#actual = actual[::-1] #reverse the labels
			break
		ed = editdistance.eval(gt, out)

	#my_dict ----- key: filename, value:(no of character errors, ground truth length, output length, % character errors)
	my_dict[os.path.basename(out_file)] = ed, len(gt), len(out), float(ed)*100/float(len(gt))
	#print os.path.basename(out_file), float(ed)*100/float(len(gt))



with open('errors.txt', 'w') as errors:
	for key in sorted(my_dict.keys()):
		errors.write('Filename = {0}, Percentage Character Errors = {4}%\r\n'.format(key, my_dict[key][3]))
		
	
	



