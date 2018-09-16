from pymongo import MongoClient
import numpy as np, sys, unicodedata, requests, os, random, pdb, requests, json
from random import randint
from pprint import pprint
import logging, logging.config, itertools, pathlib
from sklearn.metrics import precision_recall_fscore_support

np.set_printoptions(precision=4)
# c_dosa 	 = MongoClient('mongodb://10.24.28.104:27017/')
# db_word2vec 	= c_dosa['word2vec']['google_news_300']

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def getWord2vec(wrd_list):
	dim = 300
	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_word2vec.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds



# def getPhr2vec(phr_list, embed_type):
# 	dim = int(embed_type.split('_')[1])
# 	db_glove = c_dosa['glove'][embed_type]
	
# 	wrd_list = []

# 	embeds = np.zeros((len(phr_list), dim), np.float32)
# 	embed_map = {}

# 	for phr in phr_list:
# 		wrd_list += phr.split('_')

# 	wrd_list = list(set(wrd_list))

# 	res = db_glove.find({"_id": {"$in": wrd_list}})
# 	for ele in res:
# 		embed_map[ele['_id']] = ele['vec']
	
# 	count = 0
# 	for phr in phr_list:
# 		wrds = phr.split('_')
# 		vec  = np.zeros((dim,), np.float32)
# 		for wrd in wrds:
# 			if wrd in embed_map: 	vec += np.float32(embed_map[wrd])
# 			else: 			vec += np.float32(np.random.randn(dim))
# 		vec = vec / len(wrds)
# 		embeds[count, :] = vec
# 	return embeds

# def signal(message):
# 	requests.post( 'http://10.24.28.210:9999/jobComplete', data=message)

# def len_key(tp):
# 	return len(tp[1])

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def shape(tensor):
	s = tensor.get_shape()
	return tuple([s[i].value for i in range(0, len(s))])


# coreNLP_url = [ 'http://10.24.28.106:9006/', 'http://10.24.28.106:9007/', 'http://10.24.28.106:9008/', 'http://10.24.28.106:9009/', 'http://10.24.28.106:9010/', 'http://10.24.28.106:9011/', 
# 		'http://10.24.28.106:9012/', 'http://10.24.28.106:9013/', 'http://10.24.28.106:9014/', 'http://10.24.28.106:9015/', 'http://10.24.28.106:9016/']

# def callnlpServer(text):
#         params = {
#         	'properties': 	'{"annotators":"tokenize"}',
#         	'outputFormat': 'json'
#         }

#         res = requests.post(	coreNLP_url[randint(0, len(coreNLP_url)-1)],
#         			params=params, data=text, 
#         			headers={'Content-type': 'text/plain'})

#         if res.status_code == 200: 	return res.json()
#         else: 				print("CoreNLP Error, status code:{}".format(res.status_codet))


def debug_nn(res_list,feed_dict):
	import tensorflow as tf
	# ph = np.zeros(self.p.batch_size, dtype = np.int32)
	# pt = np.zeros(self.p.batch_size, dtype = np.int32)
	# r = np.zeros(self.p.batch_size, dtype = np.int32)
	# nh = np.zeros(self.p.batch_size, dtype = np.int32)
	# nt = np.zeros(self.p.batch_size, dtype = np.int32)
		 		
	# ph_addr = ph.__array_interface__['data'][0]
	# pt_addr = pt.__array_interface__['data'][0]
	# r_addr = r.__array_interface__['data'][0]
	# nh_addr = nh.__array_interface__['data'][0]
	# nt_addr = nt.__array_interface__['data'][0]
	# lib.init(self.max_ent, self.max_rel, 483142, self.p.batch_size)
	# lib.getBatch(ph_addr, pt_addr, r_addr, nh_addr, nt_addr, batch_size,1)
	# feed_dict = {pos_head : ph,
	# 			 pos_tail : pt, 
	# 			 rel 	  : r,
	# 			 neg_head : nh,
	# 		 	 neg_tail : nt}
	# facts = open('../data/train.txt','wb')
	# kg_adj_in, kg_adj_out = self.get_adj(facts, self.max_ent, self.max_rel)  # max_et + 1(DCT)		
	# for lbl in range(self.max_rel):
	# 	feed_dict[self.kg_adj_mat_in[i][lbl]] = tf.SparseTensorValue( 	indices 	= np.array([kg_adj_in[i][lbl].row, kg_adj_in[i][lbl].col]).T,
	# 								      								values  	= kg_adj_in[i][lbl].data,
	# 																	dense_shape	= kg_adj_in[i][lbl].shape)

	# 	feed_dict[self.kg_adj_mat_out[i][lbl]] = tf.SparseTensorValue(  indices 	= np.array([kg_adj_out[i][lbl].row, kg_adj_out[i][lbl].col]).T,
	# 	    								values  	= kg_adj_out[i][lbl].data,
 # 										dense_shape	= kg_adj_out[i][lbl].shape)
	# if dtype != 'train':
	# 	feed_dict[self.dropout]     = 1.0
	# 	feed_dict[self.rec_dropout] = 1.0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	pdb.set_trace()

def stanford_tokenize(text):
	res = callnlpServer(text)
	toks = [ele['word'] for ele in res['tokens']]
	return toks


def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
 
	return False

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def get_logger(name):
	config_dict = json.load(open('/scratchd/home/shikhar/gcn/config/log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = '/scratchd/home/shikhar/gcn/main/log/' + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

# doc = 'Delhi is the capital of India. Mumbai is not the capital of India.'
# pprint(callnlpServer(doc))