import os
import pandas as pd
import csv
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# path to dataset
data_path = "./benchmarks/FB15K237/"
n_dimensions = 200

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = data_path,
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(data_path, "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = n_dimensions,
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

# retrieve embedding vectors
vectors = pd.DataFrame(transe.get_parameters('list')['ent_embeddings.weight'])
path_to_entities = os.path.join(data_path, 'entity2id.txt')
entities = pd.read_csv(path_to_entities, sep='\t', skiprows=1, header=None, names=['entity'], usecols=[0])
entities_with_vectors = pd.merge(left=entities, right=vectors, left_index=True, right_index=True)

# map entities to DBpedia
dbpedia_mapping = pd.read_csv(os.path.join(data_path, 'dbpedia_mapping.csv.bz2'), sep=' ', compression='bz2')
entities_with_vectors = pd.merge(left=entities_with_vectors, right=dbpedia_mapping, how='left', left_on='entity', right_on='source')[['target'] + list(range(n_dimensions))]

# persist the model
if not os.path.exists('results'):
	os.makedirs('results')
result_path = 'results/transe_FB15K237.embeddings.txt'
entities_with_vectors.to_csv(result_path, sep=' ', header=False, index=False, quoting=csv.QUOTE_NONE)
