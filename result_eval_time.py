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
	valid_output = open('results/temp_scope/'+args.model+'/valid.txt')
	model_time = open('results/temp_scope/'+args.model+'/valid_time_pred_{}.txt'.format(k))
	model_out_time = []
	count = 0
	for line in model_time:
	    count = 0
	    temp_out = []
	    for ele in line.split():
	        tup  = (float(ele),count)
	        temp_out.append(tup)
	        count = count+1
	    model_out_time.append(temp_out)
	
	for row in model_out_time:
	    row.sort(key=lambda x:x[0])
	
	final_out_time = []
	for row in model_out_time:
	    temp_dict =dict()
	    count = 0
	    for ele in row:
	        temp_dict[ele[1]] = count
	        # temp_dict[count] = ele[1]
	        count += 1
	    final_out_time.append(temp_dict)
	
	ranks_time = []
	
	for i,row in enumerate(valid_output):
		avg_rank = []
		top_time = final_out_time[i][0]
		start_time = int(row.split()[0])
		end_time   = int(row.split()[1])
		for e in range(start_time,end_time+1,1):
			avg_rank.append(final_out_time[i][e])
		ranks_time.append(np.min(np.array(avg_rank)))
		# if top_time <= end_time and top_time >= start_time:
		# 	ranks_time.append(1)
		# else:
		# 	ranks_time.append(0)
	# pdb.set_trace()
	print('Epoch {} :  time_rank {}'.format(k ,np.mean(np.array(ranks_time))))
