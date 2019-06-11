# Import packages
# download nltk only once
# import nltk
# nltk.download('punkt')
import numpy
import torch
from models import InferSent
import csv
import sys
# import createCSV
from itertools import islice

csv.field_size_limit(sys.maxsize)

# Set up infersent
V = 2 # 1: GloVe    2: fastText
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

# Select word to vector path file
if V == 1:
    W2V_PATH = 'dataset/GloVe/glove.840B.300d.txt' # GloVe
else:
    W2V_PATH = 'dataset/fastText/crawl-300d-2M.vec' # fastText
infersent.set_w2v_path(W2V_PATH)

# Choose number of words to add to vocabulary
# infersent.build_vocab_k_words(K=500000)

startIndex = 0

for iternum in range(14):
    # Import at most 10,000 sentences
    sentences = []
    labels = []
    sentence_fileName = "sentences.csv"
    path = "dataset/TrainingData/"

    with open(path + sentence_fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        lineNum = 0

        if iternum == 13: # go all the way on the last iter since we have less than 10,000 sentences
            endIdx = None
        else:
            endIdx = startIndex + 10000

        for line in islice(csv_reader, startIndex, endIdx):
            sent = line[0]
            sentences.append(sent)

    startIndex = endIdx + 1 # update new start index

    # For test data
    # sentences = createCSV.getSentences()

    print('{0} sentences will be encoded to embeddings.'.format(len(sentences)))

    # Extract vocabulary from sentences
    infersent.build_vocab(sentences, tokenize=True)

    # Convert to word embeddings
    # embeddings are an ndarray of size: [len(sentences), 4096]
    embeddings = infersent.encode(sentences, tokenize=True, verbose=True)
    print('nb sentences encoded : {0}.'.format(len(embeddings)))

    # Write word embeddings array to csv
    # path = "../Classifiers/Data/Test/"
    vector_fileName = "X_fastText_" + str(iternum)
    # numpy.savetxt(path+vector_fileName, embeddings, delimiter=",")
    numpy.savetxt(vector_fileName, embeddings, delimiter=",")

    # Update vocabulary
    infersent.update_vocab(sentences, tokenize=True)