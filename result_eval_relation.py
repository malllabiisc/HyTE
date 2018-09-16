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
	model_output_rel = open('results/'+args.model+'/valid_rel_pred_{}.txt'.format(k))
	model_out_rel = []
	for line in model_output_rel:
	    count = 0
	    temp_out = []
	    for ele in line.split():
	        tup  = (float(ele),count)
	        temp_out.append(tup)
	        count = count+1
	    model_out_rel.append(temp_out)

	for row in model_out_rel:
	    row.sort(key=lambda x:x[0])
	
	final_out_rel= []

	for row in model_out_rel:
	    temp_dict =dict()
	    count = 0
	    for ele in row:
	        temp_dict[ele[1]] = count
	        count += 1
	    final_out_rel.append(temp_dict)
	
	ranks_rel = []
	# pdb.set_trace()
	for i,row in enumerate(valid_output):
 		ranks_rel.append(final_out_rel[i][int(row.split()[1])])
	print('Epoch {} :  test_rel rank {}'.format(k ,np.mean(ranks_rel)+1)) ## as we calculate rank from zero