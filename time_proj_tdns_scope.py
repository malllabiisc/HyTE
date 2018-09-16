from models import *
from helper import *
from random import *
from pprint import pprint
import pandas as pd
# import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt, uuid, sys, os, time, argparse
import pickle, pdb, operator, random, sys
import tensorflow as tf
from collections import defaultdict as ddict
#from pymongo import MongoClient
from sklearn.metrics import precision_recall_fscore_support
YEARMIN = -50
YEARMAX = 3000
class HyTE(Model):
	def read_valid(self,filename):
		valid_triples = []
		with open(filename,'r') as filein:
			temp = []
			for line in filein:
				temp = [int(x.strip()) for x in line.split()[0:3]]
				temp.append([line.split()[3],line.split()[4]])
				valid_triples.append(temp)

		return valid_triples

	def getOneHot(self, start_data, end_data, num_class):
		temp = np.zeros((len(start_data), num_class), np.float32)
		for i, ele in enumerate(start_data):
			if end_data[i] >= start_data[i]:
				temp[i,start_data[i]:end_data[i]+1] = 1/(end_data[i]+1-start_data[i]) 
			else:
				pdb.set_trace()
		return temp	



	def getBatches(self, data, shuffle = True):
		if shuffle: random.shuffle(data)
		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]


	def create_year2id(self,triple_time):
		year2id = dict()
		freq = ddict(int)
		count = 0
		year_list = []

		for k,v in triple_time.items():
			try:
				start = v[0].split('-')[0]
				end = v[1].split('-')[0]
			except:
				pdb.set_trace()

			if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
			if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))

		# for k,v in entity_time.items():
		# 	start = v[0].split('-')[0]
		# 	end = v[1].split('-')[0]
			
		# 	if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
		# 	if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))
			# if int(start) > int(end):
			# 	pdb.set_trace()
		
		year_list.sort()
		for year in year_list:
			freq[year] = freq[year] + 1

		year_class =[]
		count = 0
		for key in sorted(freq.keys()):
			count += freq[key]
			if count > 300:
				year_class.append(key)
				count = 0
		prev_year = 0
		i=0
		for i,yr in enumerate(year_class):
			year2id[(prev_year,yr)] = i
			prev_year = yr+1
		year2id[(prev_year, max(year_list))] = i + 1
		self.year_list =year_list
		# pdb.set_trace()

		# pdb.set_trace()
		# for k,v in entity_time.items():
		# 	if v[0] == '####-##-##' or v[1] == '####-##-##':
		# 		continue
		# 	if len(v[0].split('-')[0])!=4 or len(v[1].split('-')[0])!=4:
		# 		continue
		# 	start = v[0].split('-')[0]
		# 	end = v[1].split('-')[0]
		# for start in start_list:
		# 	if start not in start_year2id:
		# 		start_year2id[start] = count_start
		# 		count_start+=1

		# for end in end_list:
		# 	if end not in end_year2id:
		# 		end_year2id[end] = count_end
		# 		count_end+=1
		
		return year2id
	def get_span_ids(self, start, end):
		start =int(start)
		end=int(end)
		if start > end:
			end = YEARMAX

		if start == YEARMIN:
			start_lbl = 0
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if start >= key[0] and start <= key[1]:
					start_lbl = lbl
		
		if end == YEARMAX:
			end_lbl = len(self.year2id.keys())-1
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if end >= key[0] and end <= key[1]:
					end_lbl = lbl
		return start_lbl, end_lbl

	def create_id_labels(self,triple_time,dtype):
		YEARMAX = 3000
		YEARMIN =  -50
		
		inp_idx, start_idx, end_idx =[], [], []
		
		for k,v in triple_time.items():
			start = v[0].split('-')[0]
			end = v[1].split('-')[0]
			if start == '####':
				start = YEARMIN
			elif start.find('#') != -1 or len(start)!=4:
				continue

			if end == '####':
				end = YEARMAX
			elif end.find('#')!= -1 or len(end)!=4:
				continue
			
			start =int(start)
			end=int(end)
			
			if start > end:
				end = YEARMAX
			inp_idx.append(k)
			if start == YEARMIN:
				start_idx.append(0)
			else:
				for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
					if start >= key[0] and start <= key[1]:
						start_idx.append(lbl)
			
			if end == YEARMAX:
				end_idx.append(len(self.year2id.keys())-1)
			else:
				for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
					if end >= key[0] and end <= key[1]:
						end_idx.append(lbl)

		return inp_idx, start_idx, end_idx

	def load_data(self):
		triple_set = []
		with open(self.p.triple2id,'r') as filein:
			for line in filein:
				tup = (int(line.split()[0].strip()) , int(line.split()[1].strip()), int(line.split()[2].strip()))
				triple_set.append(tup)
		triple_set=set(triple_set)

		train_triples = []
		self.start_time , self.end_time, self.num_class  = ddict(dict), ddict(dict), ddict(dict)
		triple_time, entity_time = dict(), dict()
		self.inp_idx, self.start_idx, self.end_idx ,self.labels = ddict(list), ddict(list), ddict(list), ddict(list)
		max_ent, max_rel, count = 0, 0, 0

		with open(self.p.dataset,'r') as filein:
			for line in filein:
				train_triples.append([int(x.strip()) for x in line.split()[0:3]])
				triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]]
				count+=1


		# self.start_time['triple'], self.end_time['triple'] = self.create_year2id(triple_time,'triple')

		with open(self.p.entity2id,'r') as filein2:
			for line in filein2:
				# entity_time[int(line.split('\t')[1])]=[x.split()[0] for x in line.split()[2:4]]
				max_ent = max_ent+1

		self.year2id = self.create_year2id(triple_time)
		# self.start_time['entity'], self.end_time['entity'] = self.create_year2id(entity_time,'entiy')
		# self.inp_idx['entity'],self.start_idx['entity'], self.end_idx['entity'] = self.create_id_labels(entity_time,'entity')
		self.inp_idx['triple'], self.start_idx['triple'], self.end_idx['triple'] = self.create_id_labels(triple_time,'triple')
		#pdb.set_trace()	
		for i,ele in enumerate(self.inp_idx['entity']):
			if self.start_idx['entity'][i] > self.end_idx['entity'][i]:
				print(self.inp_idx['entity'][i],self.start_idx['entity'][i],self.end_idx['entity'][i])
		self.num_class = len(self.year2id.keys())
		
		# for dtype in ['entity','triple']:
		# 	self.labels[dtype] = self.getOneHot(self.start_idx[dtype],self.end_idx[dtype], self.num_class)# Representing labels by one hot notation

		keep_idx = set(self.inp_idx['triple'])
		for i in range (len(train_triples)-1,-1,-1):
			if i not in keep_idx:
				del train_triples[i]

		with open(self.p.relation2id, 'r') as filein3:
			for line in filein3:
				max_rel = max_rel +1
		index = randint(1,len(train_triples))-1
		
		posh, rela, post = zip(*train_triples)
		head, rel, tail = zip(*train_triples)

		posh = list(posh) 
		post = list(post)
		rela = list(rela)

		head  =  list(head) 
		tail  =  list(tail)
		rel   =  list(rel)

		for i in range(len(posh)):
			if self.start_idx['triple'][i] < self.end_idx['triple'][i]:
				for j in range(self.start_idx['triple'][i] + 1,self.end_idx['triple'][i] + 1):
					head.append(posh[i])
					rel.append(rela[i])
					tail.append(post[i])
					self.start_idx['triple'].append(j)
		keep_tail = ddict(set)
		keep_head = ddict(set)
		for z in range(len(head)):
			tup = (int(head[z]), int(rel[z]), int(self.start_idx['triple'][z]))
			keep_tail[tup].add(tail[z])
			tup = (int(tail[z]), int(rel[z]), int(self.start_idx['triple'][z]))
			keep_head[tup].add(head[z])

		max_time_class = len(self.year2id.keys())
		self.ph, self.pt, self.r,self.nh, self.nt , self.triple_time  = [], [], [], [], [], []
		for triple in range(len(head)):
			neg_set = set()
			neg_time_set = set()
			neg_time_set.add((tail[triple], rel[triple], self.start_idx['triple'][triple], head[triple]))
			sample_time = np.arange(max_time_class)
			random.shuffle(sample_time)
			check = 0
			head_corrupt = []
			check = 0
			random.shuffle(sample_time)
			possible_head = randint(0,max_ent-1)
			for z, time in enumerate(sample_time):
				if time == self.start_idx['triple'][triple]: continue
				if (tail[triple], rel[triple], time) not in keep_head: continue
				for s, value in enumerate(keep_head[(tail[triple], rel[triple], time)]):
					if value != head[triple] and (value, rel[triple], tail[triple]) not in neg_set:
						if (tail[triple], rel[triple], self.start_idx['triple'][triple]) in keep_tail  and value in keep_tail[(tail[triple], rel[triple], self.start_idx['triple'][triple])]:
							continue
						head_corrupt.append(value)
			head_corrupt = list(set(head_corrupt))
			random.shuffle(head_corrupt)
			for k in range(self.p.M):
				if k < len(head_corrupt):
					self.nh.append(head_corrupt[k])
					self.nt.append(tail[triple])
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx['triple'][triple])
					neg_set.add((head_corrupt[k], rel[triple],tail[triple]))
				else:
					# break
					while (possible_head, rel[triple], tail[triple]) in triple_set or (possible_head, rel[triple],tail[triple]) in neg_set:
						possible_head = randint(0,max_ent-1)
					self.nh.append(possible_head)
					self.nt.append(tail[triple])
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx['triple'][triple])
					neg_set.add((possible_head, rel[triple],tail[triple]))

				
				# if check == 1:
				# 	# while (possible_head, rel[triple], tail[triple]) in triple_set or (possible_head, rel[triple],tail[triple]) in neg_set:
				# 	# 	possible_head = randint(0,max_ent-1)


		
		for triple in range(len(tail)):
			neg_set = set()
			neg_time_set = set()
			neg_time_set.add((head[triple], rel[triple], self.start_idx['triple'][triple], tail[triple]))
			sample_time = np.arange(max_time_class)
			random.shuffle(sample_time)
			check = 0
			tail_corrupt = []
			
			possible_tail = randint(0,max_ent-1)
			check = 0
			for z, time in enumerate(sample_time):
				if time == self.start_idx['triple'][triple]: continue
				if (head[triple], rel[triple], time) not in keep_tail: continue
				for s, value in enumerate(keep_tail[(head[triple], rel[triple], time)]):
					if value != tail[triple] and (head[triple], rel[triple], value) not in neg_set:
						if (head[triple], rel[triple], self.start_idx['triple'][triple]) in keep_head  and value in keep_head[(head[triple], rel[triple], self.start_idx['triple'][triple])]:
							continue
						tail_corrupt.append(value)
			tail_corrupt = list(set(tail_corrupt))
			random.shuffle(tail_corrupt)
			for k in range(self.p.M):
				if k < len(tail_corrupt):
				# while (head[triple], rel[triple],possible_tail) in triple_set or (head[triple], rel[triple],possible_tail) in neg_set:
				# 	possible_tail = randint(0,max_ent-1)
					self.nh.append(head[triple])
					self.nt.append(tail_corrupt[k])
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx['triple'][triple])
					neg_set.add((head[triple], rel[triple],tail_corrupt[k]))
				else:
					while (head[triple], rel[triple],possible_tail) in triple_set or (head[triple], rel[triple],possible_tail) in neg_set:
							possible_tail = randint(0,max_ent-1)
					self.nh.append(head[triple])
					self.nt.append(possible_tail)
					self.r.append(rel[triple])
					self.ph.append(head[triple])
					self.pt.append(tail[triple])
					self.triple_time.append(self.start_idx['triple'][triple])
					neg_set.add((head[triple], rel[triple],possible_tail))



		# self.triple_time = triple_time
		# self.entity_time = entity_time
		self.max_rel = max_rel
		self.max_ent = max_ent
		self.max_time = len(self.year2id.keys())
		self.time_steps = sorted(self.year2id.values())
		self.data = list(zip(self.ph, self.pt, self.r , self.nh, self.nt, self.triple_time))
		self.data = self.data + self.data[0:self.p.batch_size]
		for i in range(len(self.ph)):
			if self.ph[i] == self.nh[i] and self.pt[i] == self.nt[i]:
				print("False")
		# self.max_trip = len(train_triples)
		# Data content: X(0), Y(1), Edges(2), ETEdges(3), Cats(4), Tense(5), ET(6), ETIdx(7)
		
		# # self.num_class, self.lbl2id, data = self.update_label(data)			# Maps labels to an unique id
		# print('Number of classes {}'.format(len(np.unique(data['train']['Y']))))
		# self.num_class = self.p.num_class

		# self.logger.info('Removing Train:{}, Test:{}, Valid:{}'.format(len(rm_idx['train']), len(rm_idx['test']), len(rm_idx['valid'])))

		# # Get Word List
		# self.wrd_list 	= list(self.voc2id.items())					# Get vocabulary
		# self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		# self.wrd_list, _ = zip(*self.wrd_list)
		
		# self.data_list = {}
		# key_list =  ['X', 'Y', 'ETIdx', 'ETEdges', 'DepEdges', 'Fname']

		# for dtype in ['train', 'test', 'valid']:

		# 	if self.p.use_et_labels == False:
		# 		for i, edges in enumerate(data[dtype]['ETEdges']):												# if you want to ignore level information in event time graph
		# 			for j, edge in enumerate(edges): data[dtype]['ETEdges'][i][j] = (edge[0], edge[1], 0)   
		# 		self.num_etLabel = 1

		# 	if self.p.use_de_labels == False:
		# 		for i, edges in enumerate(data[dtype]['DepEdges']):												# if you want to ignore level information in dependency graph
		# 			for j, edge in enumerate(edges): data[dtype]['DepEdges'][i][j] = (edge[0], edge[1], 0)
		# 		self.num_deLabel = 1


		# 	self.data_list[dtype] = []
		# 	for i in range(len(data[dtype]['X'])):
		# 		if i in rm_idx[dtype]: continue
		# 		self.data_list[dtype].append([data[dtype][key][i] for key in key_list])          # data_list contains all the fields for train test and valid documents
		# 	self.logger.info('Document count [{}]: {}'.format(dtype, len(self.data_list[dtype])))


	def get_adj(self,facts, max_ent, max_rel):
		adj_in, adj_out = {}, {}
		in_ind, in_data   = ddict(list), ddict(list)
		out_ind, out_data = ddict(list), ddict(list)

		for row in facts:
			row=row.split('\t')
			if row[0].isdigit() and row[1].isdigit() and row[2].split('\n')[0].isdigit():
				lbl  = int(row[1])
				src  = int(row[0])
				dest = int(row[2])
				out_ind [lbl].append((src, dest))
				out_data[lbl].append(1.0)

				in_ind  [lbl].append((dest, src))
				in_data [lbl].append(1.0)
		try:
			for lbl in range(max_rel):
				if lbl not in out_ind and lbl not in in_ind:
					adj_in [lbl] = sp.coo_matrix((max_ent, max_ent))
					adj_out[lbl] = sp.coo_matrix((max_ent, max_ent))
				else:
					adj_in [lbl] = sp.coo_matrix((in_data[lbl],  zip(*in_ind[lbl])),  shape=(max_ent, max_ent))
					adj_out[lbl] = sp.coo_matrix((out_data[lbl], zip(*out_ind[lbl])), shape=(max_ent, max_ent))
		except Exception as e:
			pdb.set_trace()
		return adj_in,  adj_out
	

	def add_placeholders(self):
		# self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')
		#self.year  = dict([(lbl,tf.placeholder(tf.float32 ,shape=[None, self.num_class],name = 'start_time_{}'.format(lbl))) for lbl in ['triple','entity']])
		self.start_year = tf.placeholder(tf.int32 ,shape=[None], name = 'start_time')
		self.end_year   = tf.placeholder(tf.int32 ,shape=[None],name = 'end_time')
		self.pos_head 	= tf.placeholder(tf.int32, [None,1])
		self.pos_tail 	= tf.placeholder(tf.int32, [None,1])
		self.rel      	= tf.placeholder(tf.int32, [None,1])
		self.neg_head 	= tf.placeholder(tf.int32, [None,1])
		self.neg_tail 	= tf.placeholder(tf.int32, [None,1])
		self.mode 	  	= tf.placeholder(tf.int32,shape = ())
		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')

	def create_feed_dict(self, batch, wLabels=True,dtype='train'):
		ph, pt, r, nh, nt, start_idx = zip(*batch)
		feed_dict = {}
		feed_dict[self.pos_head] = np.array(ph).reshape(-1,1)
		feed_dict[self.pos_tail] = np.array(pt).reshape(-1,1)
		feed_dict[self.rel] = np.array(r).reshape(-1,1)
		feed_dict[self.start_year] = np.array(start_idx)
		# feed_dict[self.end_year]   = np.array(end_idx)
		if dtype == 'train':
			feed_dict[self.neg_head] = np.array(nh).reshape(-1,1)
			feed_dict[self.neg_tail] = np.array(nt).reshape(-1,1)
			feed_dict[self.mode]   	 = 1
		else: 
			feed_dict[self.mode] = -1
		if dtype != 'train':
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0

		return feed_dict


	def time_projection(self,data,t):
		inner_prod  = tf.tile(tf.expand_dims(tf.reduce_sum(data*t,axis=1),axis=1),[1,self.p.inp_dim])
		prod 		= t*inner_prod
		data = data - prod
		return data

	def add_model(self):
			#nn_in = self.input_x
		with tf.name_scope("embedding"):
			self.ent_embeddings =  tf.get_variable(name = "ent_embedding",  shape = [self.max_ent, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.rel_embeddings =  tf.get_variable(name = "rel_embedding",  shape = [self.max_rel, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.time_embeddings = tf.get_variable(name = "time_embedding",shape = [self.max_time, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform =False))
		
			transE_in_dim = self.p.inp_dim
			transE_in     = self.ent_embeddings
		####################------------------------ time aware GCN MODEL ---------------------------##############


	
		## Some transE style model ####
		neutral = tf.constant(0)
		test_type = tf.constant(0)

		def f_train():
			pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
			t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
			return pos_h_e, pos_t_e, pos_r_e, t_1
		def f_test():
			e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_h_e = tf.reshape(tf.tile(e1,[self.max_time]),(self.max_time, transE_in_dim))
			pos_t_e = tf.reshape(tf.tile(e2,[self.max_time]),(self.max_time, transE_in_dim))
			r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
			pos_r_e = tf.reshape(tf.tile(r,[self.max_time]),(self.max_time,transE_in_dim))
			t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
			return pos_h_e, pos_t_e, pos_r_e, t_1

		pos_h_e, pos_t_e, pos_r_e, t_1 = tf.cond(self.mode > neutral, f_train, f_test)
		neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
		neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

		#### ----- time -----###
		
		pos_h_e_t_1 = self.time_projection(pos_h_e,t_1)
		neg_h_e_t_1 = self.time_projection(neg_h_e,t_1)
		pos_t_e_t_1 = self.time_projection(pos_t_e,t_1)
		neg_t_e_t_1 = self.time_projection(neg_t_e,t_1)
		pos_r_e_t_1 = self.time_projection(pos_r_e,t_1)
		
		if self.p.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims = True) 
			neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims = True) 
		else:
			pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims = True) 
			neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims = True)
		return pos, neg

	def add_loss(self, pos, neg):
		with tf.name_scope('Loss_op'):
			loss     = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			return loss

	def add_optimizer(self, loss):
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		# ent_normalizer = tf.assign(self.ent_embeddings, tf.nn.l2_normalize(self.ent_embeddings,dim = 1))
		# rel_normalizer = tf.assign(self.rel_embeddings, tf.nn.l2_normalize(self.rel_embeddings,dim = 1))
		time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings,dim = 1))
		return train_op

	def __init__(self, params):
		self.p  = params
		self.p.batch_size = self.p.batch_size
		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
		self.load_data()
		self.nbatches = len(self.data) // self.p.batch_size
		self.add_placeholders()
		self.pos, neg = self.add_model()
		self.loss      	= self.add_loss(self.pos, neg)
		self.train_op  	= self.add_optimizer(self.loss)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None
		print('model done')

	def run_epoch(self, sess,data,epoch):
		losses = []

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			l, a = sess.run([self.loss, self.train_op],feed_dict = feed)
			# print(l,step)
			losses.append(l)
			# pdb.set_trace()
		return np.mean(losses)

	def fit(self, sess):
		#self.best_val_acc, self.best_train_acc = 0.0, 0.0
		saver = tf.train.Saver(max_to_keep=None)
		save_dir = './checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_dir_results = './results/'+ self.p.name + '/'
		if not os.path.exists(save_dir_results): os.makedirs(save_dir_results)
		if self.p.restore:
			save_path = os.path.join(save_dir, 'epoch_{}'.format(self.p.restore_epoch))
			saver.restore(sess, save_path)
		print('start fitting')
		self.best_prf = None
		validation_data = self.read_valid(self.p.test_data)
		if not self.p.onlyTest:
			for epoch in range(self.p.max_epochs):
				l = self.run_epoch(sess,self.data,epoch)
				
				if epoch%(self.p.test_freq/5) == 0:
					print('Epoch {}\tLoss {}\t model {}'.format(epoch,l,self.p.name))
				if epoch % self.p.test_freq == 0 and epoch != 0:
					save_path = os.path.join(save_dir, 'epoch_{}'.format(epoch))   ## -- check pointing -- ##
					saver.save(sess=sess, save_path=save_path)
					save_dir_results = 'temp_scope/'+ self.p.name + '/'
					if not os.path.exists(save_dir_results): os.makedirs(save_dir_results) 
					
					if epoch == self.p.test_freq:
						f_valid = open(save_dir_results+'/valid.txt','w')
					f_time = open(save_dir_results+'/valid_time_pred_{}.txt'.format(epoch),'w')
					
					for i,t in enumerate(validation_data):
						loss =np.zeros(self.max_ent)
						start_trip 	= t[3][0].split('-')[0]
						end_trip 	= t[3][1].split('-')[0]
						if start_trip == '####':
							start_trip = YEARMIN
						elif start_trip.find('#') != -1 or len(start_trip)!=4:
							continue

						if end_trip == '####':
							end_trip = YEARMAX
						elif end_trip.find('#')!= -1 or len(end_trip)!=4:
							continue
							
						start_lbl, end_lbl = self.get_span_ids(start_trip, end_trip)
						if epoch == self.p.test_freq:
							f_valid.write(str(start_lbl)+'\t'+str(end_lbl)+'\n')
						pos_time = sess.run(self.pos ,feed_dict = { self.pos_head:  	np.array([t[0]]).reshape(-1,1), 
																   	self.rel:       	np.array([t[1]]).reshape(-1,1), 
																   	self.pos_tail:	    np.array([t[2]]).reshape(-1,1),
																   	self.start_year:    np.array(self.time_steps),
																   	self.end_year:      np.array([end_lbl]*self.max_ent),
																   	self.mode: 			-1,
																   }
										    )
		

						pos_time = np.squeeze(pos_time)
						f_time.write(' '.join([str(x) for x in pos_time]) + '\n')
						
						if i%1000 == 0:
							print('{}. no of valid_triples complete'.format(i))
						
					
					if epoch == self.p.test_freq:
						f_valid.close()
					f_time.close()
					print("Validation Ended")
		else: 
			time_mat = sess.run(self.time_embeddings)
			df = pd.DataFrame(data=time_mat.astype(float)) 
			df.to_csv(self.p.name + str(self.p.restore_epoch) + '.tsv', sep='\t', header=False, float_format='%.6f', index=False)

if __name__== "__main__":
	print('here in main')
	parser = argparse.ArgumentParser(description='KG temporal inference using GCN')

	parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose')
	parser.add_argument('-version',dest = 'version', default = 'large', help = 'data version to choose')
	parser.add_argument('-test_freq', 	 dest="test_freq", 	default = 25,   	type=int, 	help='Batch size')
	parser.add_argument('-neg_sample', 	 dest="M", 		default = 1,   	type=int, 	help='Batch size')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_300',	 	help='Embedding for initialization')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
	parser.add_argument('-margin', 	 dest="margin", 	default= 10 ,   	type=float, 	help='margin')
	parser.add_argument('-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer')
	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model on test')
	parser.add_argument('-onlytransE', dest="onlytransE", 	action='store_true', 		help='Evaluate model on only transE loss')
	parser.add_argument('-restore',	 		 dest="restore", 	    action='store_true', 		help='Restore from the previous best saved model')
	parser.add_argument('-res_epoch',	     dest="restore_epoch", 	default=200,   type =int,		help='Restore from the previous best saved model')
	
	args = parser.parse_args()
	args.dataset = 'data/'+ args.data_type +'/'+ args.version+'/train.txt'
	args.entity2id = 'data/'+ args.data_type +'/'+ args.version+'/entity2id.txt'
	args.relation2id = 'data/'+ args.data_type +'/'+ args.version+'/relation2id.txt'
	args.test_data  =  'data/'+ args.data_type +'/'+ args.version+'/valid.txt'
	args.triple2id  =   'data/'+ args.data_type +'/'+ args.version+'/triple2id.txt'
	args.embed_dim = int(args.embed_init.split('_')[1])

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)
	model  = HyTE(args)
	print('model object created')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print('enter fitting')
		model.fit(sess)
