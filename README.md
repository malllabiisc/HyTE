# HyTE
## HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding

Source code and dataset for [EMNLP 2018](http://emnlp2018.org) paper: [HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding](http://malllabiisc.github.io/publications/).

![](https://raw.githubusercontent.com/malllabiisc/NeuralDater/master/overview.png)
*Overview of HyTE (proposed method). a temporally aware
KG embedding method which explicitly incorporates time in the entity-relation space by
stitching each timestamp with a corresponding hyperplane. HyTE not only performs KG
inference using temporal guidance, but also predicts temporal scopes for relational facts with missing time annotations. Please refer paper for more details.*
### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.


### Dataset:

* Download the processed version (includes dependency and temporal graphs of each document) of [WikiData](To be shared) and [YAGO](https://drive.google.com/open?id=1tll04ZBooB3Mohm6It-v8MBcjMCC3Y1w) datasets.
* Unzip the `.pkl` file in `data` directory.
* Documents are originally taken from YAGO(share yago's web address) and Wikidata(share wiki data website).


### Usage:

* After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

* `time_proj.py` contains TensorFlow (1.x) based implementation of HyTE (proposed method). 
* To start training: 
  ```shell
  python time_proj.py -data data/nyt_processed_data.pkl -class 10 -name test_run -<other_optins> ...
  ```
*  Some of the important Available options include:
  ```shell
  	'-data_type' default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose'
	'-version',  default = 'large', choices = ['large','small'], help = 'data version to choose'
	'-test_freq', 	 default = 25,   	type=int, 	help='testing frequency'
	'-neg_sample', 	 default = 5,   	type=int, 	help='negative samples for training'
	'-gpu', 	 dest="gpu", 		default='1',			help='GPU to use'
	'-name', 	 dest="name", 		help='Name of the run'
	'-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate'
	'-margin', 	 dest="margin", 	default=1,   	type=float, 	help='margin'
	'-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size'
	'-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs'
	'-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization'
	'-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization'
	'-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='')
	'-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer'
   ```
* After trainig start validation/test using--
 ```shell
 ```


### Citing:

```tex
@InProceedings{
}
```
