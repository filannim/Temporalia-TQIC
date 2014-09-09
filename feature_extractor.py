#!/usr/bin/python

from __future__ import division

import pprint
import re
import sys
import time
import os
from collections import Counter, namedtuple, defaultdict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

sys.path.append('/home/filannim/Dropbox/Workspace/ManTIME')
from annotate_sentence import annotate as ManTIME
from external.general_timex_normaliser import normalise as TemporallyNormalise

TempoSynset = namedtuple('TempoSynset', ['past','present','future','atemporal'])

def top(l):
	"""
	Return the first element of an iterable object if it is not empty.
	"""
	return l[0] if l else ''

def get_temporal_expressions(sentence, utt):
	"""
	This method is a wrapper for ManTIME. It analyses ManTIME's output 
	and returns two lists: years and months mentions.
	"""
	start = time.time()
	r_timex3_vals = re.compile(r'value=\"([^\"]+)\"')
	r_timex3_years = re.compile(r'^[12][0-9]{3}$')
	r_timex3_months = re.compile(r'^[12][0-9]{3}\-[01][0-9]')
	timex3_vals = r_timex3_vals.findall(ManTIME(sentence, utterance=utt)[0])
	timex3_years = [int(v[:4]) for v in timex3_vals if r_timex3_years.match(v)]
	timex3_months = [v for v in timex3_vals if r_timex3_months.match(v)]
	#if ManTIME doesn't find anything, use a simple DDDD regular expression
	if not(timex3_years) and not(timex3_months):
		timex3_years = [int(v) for v in re.findall(r'[12][0-9]{3}', sentence)]
	print "ManTIME (" + str(round(time.time()-start,2)) + " s.)" 
	return (timex3_years, timex3_months)

class TempoWordNet(object):

	def __init__(self):
		self._path = '/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/data/tempowordnet_1.0.tab_separated.txt'

		self.offsets = defaultdict(TempoSynset)
		with open(self._path, 'r') as source:
			for line in source:
				line = line.strip()
				if line:
					fields = line.split('\t')
					id = int(fields[0])
					self.offsets[id] = TempoSynset(float(fields[4]), 
						float(fields[5]), float(fields[6]), float(fields[7]))

	def past(self, id): return self.offsets[id].past

	def present(self, id): return self.offsets[id].present

	def future(self, id): return self.offsets[id].future

	def atemporal(self, id): return self.offsets[id].atemporal

	def scores(self, id): return self.offsets[id]

class FeatureExtractor(object):

	def __init__(self):
		print 'Feature extractor:',
		start = time.time()
		self.tempowordnet = TempoWordNet()
		self.wordnetlemmatizer = WordNetLemmatizer()
		self.wikipedia_titles = set(open('/opt/DBpedia/sorted_uniq_lowered_super_stopworded_nocategories_labels.txt','r').read().splitlines())
		self.atemporal_triggers = set(open('/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/temporalia/data/atemporal.txt', 'r').read().splitlines())
		self.past_triggers = set(open('/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/temporalia/data/past.txt', 'r').read().splitlines())
		self.present_triggers = set(open('/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/temporalia/data/recent.txt', 'r').read().splitlines())
		self.future_triggers = set(open('/home/filannim/Dropbox/Workspace/NTCIR-11_Temporalia/temporalia/data/future.txt', 'r').read().splitlines())
		print "initialised (" + str(round(time.time()-start,2)) + " s.)" 

	def __w_type(self, sentence):
		"""
		It returns the kind of W type of the query (what, where, when, how,
		why, who), by checking at the beginning of the sentence
		"""
		assert type(sentence) == Sentence
		s = sentence.normalised.lower()
		if s.startswith("what"):
			return "what"
		elif s.startswith("where"):
			return "where"
		elif s.startswith("when"):
			return "when"
		elif s.startswith("how"):
			return "how"
		elif s.startswith("why"):
			return "why"
		elif s.startswith("who"):
			return "who"
		else:
			return ""

	def __timing(self, sentence, utterance):
		"""
		It returns:
		1) a boolean value, found/not_found at least a TIMEX
		2) numerical difference between query and utterance
		3) the temporal delta according to the utterance time:
		'', past, present, future.
		"""
		assert type(sentence) == Sentence
		assert type(utterance) == Utterance
		years, months = get_temporal_expressions(sentence.raw_text, utterance.ISO8601)

		if not(years) and not(months):
			if 'VBD' in sentence.postags:
				return '0', '', 'past'
			return '0', '', ''
		if years and not months:					# compute year-distance
			mean_year = sum(years)/len(years)
			if utterance.year > mean_year:
				return '1', str((utterance.year-mean_year)*12), 'past'
			elif utterance.year == mean_year:
				delta_month = utterance.month-6
				if delta_month > 0:
					return '1', str(delta_month), 'past'
				elif delta_month < 0:
					return '1', str(delta_month), 'future'
				else:
					return '1', str(delta_month), 'present'
			else:
				return '1', str((utterance.year-mean_year)*12), 'future'
		elif months and not years:					# compute month-distance
			mean_month = 0
			for month in months:
				year, month = map(int, month[:7].split('-'))
				mean_month += (year*12) + month
			mean_month = mean_month / 12
			mean_utterance = ((utterance.year*12) + utterance.month) / 12
			if mean_utterance > mean_month:
				return '1', str(mean_utterance-mean_month), 'past'
			elif mean_utterance == mean_month:
				return '1', '0', 'present'
			else:
				return '1', str(mean_utterance-mean_month), 'future'
		else:
			return '1', '', ''

	def __part_of_the_year_in_3(self, utterance):
		"""
		It returns the part of the year in which the query has been submitted:
		B = Beginning 	(from January to April)
		M = Middle 		(from May to August)
		E = End 		(from September to December)
		"""
		assert type(utterance) == Utterance
		if utterance.month <= 4:
			return 'B'
		elif utterance.month >= 9:
			return 'E'
		else:
			return 'M'

	def __part_of_the_year_in_4(self, utterance):
		"""
		It returns the part of the year in which the query has been submitted:
		B = Beginning 	(from January to March)
		M1 = Middle 1	(from April to June)
		M2 = Middle 2   (from July to September)
		E = End 		(from October to December)
		"""
		assert type(utterance) == Utterance
		if utterance.month in range(1,4):
			return 'B'
		elif utterance.month in range(4,7):
			return 'M1'
		elif utterance.month in range(7,10):
			return 'M2'
		else:
			return 'E'

	def __get_trigger_classes(self, sentence):
		"""
		It returns an ordered list of temporal classes according to the
		external vocabularies.
		"""
		assert type(sentence) == Sentence
		c = Counter()
		for token in sentence.tokens:
			token = token.lower()
			if token in self.atemporal_triggers: c['atemporal'] += 1
			if token in self.past_triggers: c['past'] += 1
			if token in self.present_triggers: c['present'] += 1
			if token in self.future_triggers: c['future'] += 1
		if c:
			sorted_classes = zip(*c.most_common(len(c)))[0]
			return top(sorted_classes), '-'.join(sorted_classes)
		else:
			return '', ''
		
	def __TempoWordNet_classes(self, sentence, tempowordnet, lemmatizer):
		"""
		It returns an ordered list of temporal classes according to 
		TempoWordNet.
		"""
		assert type(sentence) == Sentence
		def get_WN_pos(p):
			if p in ['JJ','JJR','JJS']: return wn.ADJ
			elif p in ['RB','RBR','RBS']: return wn.ADV
			elif p in ['NN','NNS']: return wn.NOUN
			elif p in ['VB','VBD','VBG','VBN','VBP','VBZ']: return wn.VERB
			elif p in ['NNP','NNPS']: return 'skip'
			else: return None
		
		scores = defaultdict(float)
		for token, penn_pos in zip(sentence.tokens, sentence.postags):
			pos = get_WN_pos(penn_pos)
			if pos == wn.NOUN:
				continue
			if penn_pos == 'VBD':
				scores['past'] += 1.
			elif token == 'will':
				scores['future'] += 1.
			elif pos!='skip':
				if pos:
					lemma = lemmatizer.lemmatize(token, pos=pos)
				else:
					lemma = lemmatizer.lemmatize(token)
				for rank, offset in enumerate(map(int, [s.offset for s in wn.synsets(lemma, pos=pos)][:3]), start=1):
					try:
						ss_offset = int()		# the list can be empty
						scores['past'] += tempowordnet.scores(offset).past/rank
						scores['present'] += tempowordnet.scores(offset).present/rank
						scores['future'] += tempowordnet.scores(offset).future/rank
						scores['atemporal'] += tempowordnet.scores(offset).atemporal/rank
					except:
						continue
		sorted_scores = sorted(scores, key=scores.get, reverse=True)
		return top(sorted_scores), '-'.join(sorted_scores)

	def __tenses(self, sentence):
		assert type(sentence) == Sentence
		filtered_postags = Counter([p for t, p in zip(sentence.tokens, sentence.postags) if p.startswith('V') or t=='will'])
		sorted_postags = sorted(filtered_postags, key=filtered_postags.get, reverse=True)
		return top(sorted_postags), '-'.join(sorted_postags)

	def __pos_footprint(self, sentence):
		assert type(sentence) == Sentence
		return '-'.join(sentence.postags)

	def __pos_ordered_footprint(self, sentence):
		assert type(sentence) == Sentence
		pos_tags = Counter(sentence.postags)
		sorted_postags = sorted(pos_tags, key=pos_tags.get, reverse=True)
		return top(sorted_postags), '-'.join(sorted_postags)

	def __pos_simplified_footprint(self, sentence):
		assert type(sentence) == Sentence
		return '-'.join(map(lambda p: p[0], sentence.postags))

	def __pos_simplified_ordered_footprint(self, sentence):
		assert type(sentence) == Sentence
		pos_tags = Counter(map(lambda p: p[0], sentence.postags))
		sorted_postags = sorted(pos_tags, key=pos_tags.get, reverse=True)
		return top(sorted_postags), '-'.join(sorted_postags)

	def __is_wikipedia_title(self, sentence):
		assert type(sentence) == Sentence
		if sentence.lower in self.wikipedia_titles:
			return 'T'
		else:
			return 'F'

	def extract(self, sentence, utterance):
		sentence = Sentence(sentence)
		utterance = Utterance(utterance)
		first_tempowordnet_class, tempowordnet_classes = self.__TempoWordNet_classes(sentence, self.tempowordnet, self.wordnetlemmatizer)
		first_trigger_class, trigger_classes = self.__get_trigger_classes(sentence)
		first_ordered_pos, ordered_pos_footprint = self.__pos_ordered_footprint(sentence)
		first_simplified_ordered_pos, simplified_ordered_pos_footprint = self.__pos_simplified_ordered_footprint(sentence)
		first_tense, tenses = self.__tenses(sentence)
		timex3_in, timing_num, timing = self.__timing(sentence, utterance)
		return {'QUERY':sentence.raw_text,
				'QUERY_normalised': sentence.normalised,
				'w_type': self.__w_type(sentence),
				'timex3_in': timex3_in,
				'timing_num': timing_num,
				'timing': timing,
				'tempowordnet_first_class': first_tempowordnet_class,
				'tempowordnet_classes': tempowordnet_classes,
				'part_of_the_year_in_3': self.__part_of_the_year_in_3(utterance),
				'part_of_the_year_in_4': self.__part_of_the_year_in_4(utterance),
				'trigger_first_class': first_trigger_class,
				'trigger_classes': trigger_classes,
				'tenses': tenses,
				'first_tense': first_tense,
				'pos_footprint': self.__pos_footprint(sentence),
				'pos_simplified_footprint': self.__pos_simplified_footprint(sentence),
				'pos_first_ordered_footprint': first_ordered_pos,
				'pos_ordered_footprint': ordered_pos_footprint,
				'pos_first_simplified_ordered_footprint': first_simplified_ordered_pos,
				'pos_simplified_ordered_footprint': simplified_ordered_pos_footprint,
				'is_wikipedia_title': self.__is_wikipedia_title(sentence)}

class Utterance(object):

	def __init__(self, utterance):
		assert type(utterance) == str
		self.raw_text = utterance
		self.ISO8601 = TemporallyNormalise(utterance, '20140331')[2]
		self.year, self.month, self.day = map(int, self.ISO8601.split('-'))

class Sentence(object):

	def __init__(self, sentence):
		assert type(sentence) == str
		assert len(sentence) > 0
		self.raw_text = sentence.strip()
		self.lower = sentence.lower()
		self.normalised = sentence.replace('"', '``').strip()
		self.tokens, self.postags = zip(*pos_tag(word_tokenize(sentence)))

def main():
	_, sentence, utterance = sys.argv
	fe = FeatureExtractor()
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(fe.extract(sentence, utterance))

if __name__ == "__main__":
	main()