import numpy as np
import argparse,pdb
parser = argparse.ArgumentParser(description='Eval model outputs')
parser.add_argument('-model', 	 	dest = "model", required=True,				help='Dataset to use')
parser.add_argument('-test_freq', 	dest = "freq", 	required=True,	type =int,  help='what is to be predicted')

#parser.add_argument('-entity2id'  , dest="entity2id", 		required=True,			help='Entity 2 id')
#parser.add_argument('-relation2id', dest="relation2id", 	required=True,			help=' relation to id')
args = parser.parse_args()

	
print(args.model)
for k in range(args.freq,30000,args.freq):
	valid_output = open('results/'+args.model+'/valid.txt')
	model_output_head = open('results/'+args.model+'/valid_head_pred_{}.txt'.format(k))
	model_output_tail = open('results/'+args.model+'/valid_tail_pred_{}.txt'.format(k))
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
	# pdb.set_trace()
	for i,row in enumerate(valid_output):
		ranks_head.append(final_out_head[i][int(row.split()[0])])
		ranks_tail.append(final_out_tail[i][int(row.split()[2])])
	print('Epoch {} :  test_tail rank {}\t test_head rank {}'.format(k ,np.mean(np.array(ranks_tail))+1, np.mean(np.array(ranks_head))+1))
