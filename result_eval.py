import numpy as np
import argparse,pdb, sys
parser = argparse.ArgumentParser(description='Eval model outputs')
parser.add_argument('-model', 	 	dest = "model", required=True,				help='Dataset to use')
parser.add_argument('-eval_mode', 	 	dest = "eval_mode", required=True,		help='To evaluate test or validation')
parser.add_argument('-test_freq', 	dest = "freq", 	required=True,	type =int,  help='what is to be predicted')

#parser.add_argument('-entity2id'  , dest="entity2id", 		required=True,			help='Entity 2 id')
#parser.add_argument('-relation2id', dest="relation2id", 	required=True,			help=' relation to id')
args = parser.parse_args()

best_rank = sys.maxsize	
print(args.model)
for k in range(args.freq,30000,args.freq):
	valid_output = open('results/'+args.model+'/'+args.eval_mode+'.txt')
	model_output_head = open('results/'+args.model+'/'+args.eval_mode+'_head_pred_{}.txt'.format(k))
	model_output_tail = open('results/'+args.model+'/'+args.eval_mode+'_tail_pred_{}.txt'.format(k))
	model_out_head = []
	model_out_tail = []
	count = 0
	for line in model_output_head:
	    count = 0
	    temp_out = []
	    for ele in line.split():
	        tup  = (float(ele),count)
	        temp_out.append(tup)
	        count = count+1
	    model_out_head.append(temp_out)

	for line in model_output_tail:
	    count = 0
	    temp_out = []
	    for ele in line.split():
	        tup  = (float(ele),count)
	        temp_out.append(tup)
	        count = count+1
	    model_out_tail.append(temp_out)
	
	for row in model_out_head:
	    row.sort(key=lambda x:x[0])

	for row in model_out_tail:
	    row.sort(key=lambda x:x[0])
	
	final_out_head , final_out_tail= [], []
	for row in model_out_head:
	    temp_dict =dict()
	    count = 0
	    for ele in row:
	        temp_dict[ele[1]] = count
	        count += 1
	    final_out_head.append(temp_dict)

	for row in model_out_tail:
	    temp_dict =dict()
	    count = 0
	    for ele in row:
	        temp_dict[ele[1]] = count
	        count += 1
	    final_out_tail.append(temp_dict)
	
	ranks_head = []
	ranks_tail = []

	for i,row in enumerate(valid_output):
		ranks_head.append(final_out_head[i][int(row.split()[0])])
		ranks_tail.append(final_out_tail[i][int(row.split()[2])])

	print('Epoch {} : {}_tail rank {}\t {}_head rank {}'.format(k, args.eval_mode, np.mean(np.array(ranks_tail))+1, args.eval_mode, np.mean(np.array(ranks_head))+1))

	tail_array = np.array(ranks_tail)
	head_array = np.array(ranks_head)

	hit_at_10_tail = tail_array[np.where(tail_array < 10)]
	hit_at_10_head = head_array[np.where(head_array < 10)]

	print('Epoch {} : {}_tail HIT@10 {}\t {}_head HIT@!) {}'.format(k, args.eval_mode, len(hit_at_10_tail)/float(len(tail_array))*100, args.eval_mode, len(hit_at_10_head)/float(len(head_array))*100))
	
	if args.eval_mode == 'valid':
		if (np.mean(np.array(ranks_tail))+1 + np.mean(np.array(ranks_head))+1)/2 < best_rank:
			best_rank = (np.mean(np.array(ranks_tail))+1 + np.mean(np.array(ranks_head))+1)/2
			best_epoch = k
			best_tail_rank = np.mean(np.array(ranks_tail))+1
			best_head_rank = np.mean(np.array(ranks_head))+1
		print('------------------------------------------')
		print('Best Validation Epoch till now Epoch {}, tail rank: {}, head rank: {}'. format(best_epoch, best_tail_rank, best_head_rank))
		print('------------------------------------------')
