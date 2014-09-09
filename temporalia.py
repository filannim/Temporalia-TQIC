#!/usr/bin/python

from collections import namedtuple
import itertools
import os
import pickle
import re
import subprocess
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET
import multiprocessing

from feature_extractor import FeatureExtractor

#Sample = namedtuple('Sample', ['id', 'sentence', 'utterance', 'features', 'CLASS'])
#fe = FeatureExtractor()

def analyse_a_query(query):
	id = query.find('id').text.strip()
	sentence = query.find('query_string').text.strip()
	utterance = query.find('query_issue_time').text.strip()
	#print 'Analysing query: "%s" [%s]' % (sentence,utterance)
	features = fe.extract(sentence, utterance)
	temporal_class = query.find('temporal_class').text
	if type(temporal_class)== str:
		temporal_class = temporal_class.strip()
	else:
		temporal_class = None
	sample = Sample(id, sentence, utterance, features, temporal_class)
	#print 'Analysed query: "%s" [%s]' % (sentence,utterance)
	return sample

def get_data(source, save_to=None):
	dryrun = ET.parse(source)
	root = dryrun.getroot()
	pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
	samples = pool.map(analyse_a_query, root.findall('query'))
	
	#samples = []
	#for query in root.findall('query'):
	#	samples.append(analyse_a_query(query))

	if save_to:
		pickle.dump(samples, open(save_to, 'w'))

	return samples

def export_tsv(data, save_to='./data/output.csv'):
	result = open(save_to, 'w')
	# header
	header = ['', 'CLASS']#sample_arguments
	header[0] = data[0].features.keys()[0]
	for id, feature in enumerate(data[0].features.keys()[1:], start=1):
		header.insert(id, feature)
	result.write('\t'.join(header) + '\n')

	#samples
	for sample in data:
		l = []#[sample.id, sample.sentence, sample.utterance]
		for feature in sample.features.values():
			l.append('"'+feature+'"')
		if sample.CLASS:
			l.append('"'+sample.CLASS+'"')
		else:
			l.append('""')
		result.write('\t'.join(l) + '\n')

	#for sample in data:
		#print sample.id

	result.close()

# def annotate_data(data, feature_list):
# 	#build temporary arff file
# 	with NamedTemporaryFile('w+t', suffix='.arff', delete=False) as f:
# 		filename = f.name
# 		f.write("@relation ''\n\n")
# 		f.write("@attribute tenses {VBN,'',VBG,VBP-VBD,VB,VBZ,VBN-VBD,VBD,VBP,VBP-VBG,VBG-VBD,VBZ-VBG-VBN,VBZ-VB,VB-VBD}\n")
# 		f.write("@attribute part_of_the_year {M,B,E}\n")
# 		f.write("@attribute timing {present,past,none,future}\n")
# 		f.write("@attribute w_type {none,how,what,who,when}\n")
# 		f.write("@attribute trigger_classes {future-present,'',future,present,past-present,present-future,past,future-past,atemporal-present,past-future}\n")
# 		f.write("@attribute CLASS {Recent,Past,Future,Atemporal}\n\n")
# 		f.write("@data\n")
# 		for sample in data:
# 			feature_values = []
# 			for feature_name in feature_list:
# 				value = sample.features[feature_name]
# 				if value:
# 					feature_values.append(value)
# 				else:
# 					feature_values.append("''")
# 			feature_values.append('?')
# 			f.write(','.join(feature_values) + "\n")
	
# 	#run weka through java
# 	process = subprocess.Popen(['java', '-cp', '/usr/share/java/weka.jar', 'weka.classifiers.functions.SMO', '-T', filename, '-l', './weka_experiments/SMO_NormalizedPolyKernel.model', '-p', '0'], stdout=subprocess.PIPE)
# 	output, _ = process.communicate()
# 	#parse output
# 	os.remove(filename)
# 	classes = {'1':'Recent','2':'Past','3':'Future','4':'Atemporal'}
# 	regex_prediction_line = re.compile(r'^ +\d+ +\d:[\?A-Za-z]+ +(?P<predicted_class>\d):')
# 	predictions = []
# 	for line in output.split('\n'):
# 		matching = regex_prediction_line.match(line)
# 		if matching:
# 			predictions.append(classes[matching.group('predicted_class')])

# 	#return classes
# 	return predictions

# def export_in_submission_format(data, predictions, run_id, group_id="UniMAN"):
# 	print '\t'.join(['id', 'class', 'group_id', 'run_id'])
# 	for sample, prediction in zip(data,predictions):
# 		print '\t'.join([sample.id, prediction.lower(), group_id, run_id])

def from_WEKA_text_to_submission_format(weka_txt_file_path, run_id, group_id="UniMAN"):
	from collections import namedtuple
	import csv

	classes = {'1':'Recent', '2':'Past', '3':'Future', '4':'Atemporal'}
	with open(weka_txt_file_path, 'r') as weka_output:
		reader = csv.reader(weka_output, delimiter='\t')
		Row = namedtuple('Row',  ['id','gold','prediction','error'])
		print '\t'.join(['id','class','group_id','run_id'])
		for r in reader:
			r = Row(*r)
			print '\t'.join([r.id, classes[r.prediction[0]].lower(), group_id, run_id])			


def main():
	#os.chdir('/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/temporalia')
	#filenames = ('dryrun_annotated', 'formalrun_annotated')
	#for f in filenames:
	#	data = get_data('../data/'+f+'.xml', save_to='./data/'+f+'.pickled')
		#data = pickle.load(open('./data/data.pickled','r'))
	#	export_tsv(data, save_to='./data/'+f+'.csv')

	from_WEKA_text_to_submission_format('ml_output/formal_run_annotated_full.csv', 'full')

	#X = [sample.features.values() for sample in data if sample.CLASS]
	#y = [sample.CLASS for sample in data if sample.CLASS]
	#test_set = [sample for sample in data if not sample.CLASS]
	#feature_list = ['tenses', 'part_of_the_year', 'timing', 'w_type', 'trigger_classes']
	#export_in_submission_format(test_set, annotate_data(test_set, feature_list), 'piccolino'	)


if __name__ == '__main__':
	main()