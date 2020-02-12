from torch.utils.data import Dataset 
from preprocessing import * 
from tokenizer import * 


class BertDataset(Dataset):
  def __init__(self, source , target):  
    source_sentences, target_sentences = load_files(source,target)
    self.tokenizer = construct_vocabulary(source_sentences, target_sentences)
    # X is formed of the ids of the tokenised source sentences 
    # Y is formed of the ids of the tokenised target sentences 
    # data is the concatenation of X and masked_Y 
    # token_type_ids and masked_lm_labels are variables "indicators" that will be needed for pre-training and fine tunning BERT. 
      
    self.X, self.Y, self.data, self.token_type_ids, self.masked_lm_labels = bert_format(source_sentences, target_sentences, self.tokenizer,0.15) 

  
  def constract_batches(self) : 
      # This function returns batches of sequences that have the same length 
      # In order to switch to tensors, our batches should have the same length. 
      # We would think of padding or tuncating them, but in the paper that we are refering to, 
      # they have choosed to form batches of sequences that have the same length. 
      lengths = {}
      for i in range (len(self.data)): 
        if len(self.data[i]) in lengths.keys(): 
          lengths[len(self.data[i])].append(i) 
        else: 
          lengths[len(self.data[i])] = [i]

      batches_data = []
      batches_token_type_ids = []
      baches_masked_lm_labels =[]
      for length in lengths.keys(): 
        batches_data.append(self.data[np.array(lengths[length])])
        batches_token_type_ids.append( self.token_type_ids [np.array(lengths[length])])
        baches_masked_lm_labels.append( self.masked_lm_labels [np.array(lengths[length])])

      return convert_to_tensors(np.array(batches_data)), convert_to_tensors(np.array(batches_token_type_ids)), convert_to_tensors(np.array(baches_masked_lm_labels))
      


class TextDataset(Dataset):
  def __init__(self, source , target , batch_size ):  
    source_sentences, target_sentences = load_files(source,target)
    self.tokenizer = construct_vocabulary(source_sentences, target_sentences)
    X_train = ["[CLS] " + x + " [SEP]"  for x in source_sentences]
    Y_train = ["[CLS] " + y + " [SEP]"  for y in target_sentences]
    
    tokenized_X_train = [self.tokenizer.tokenize(sent) for sent in X_train]
    tokenized_Y_train = [self.tokenizer.tokenize(sent) for sent in Y_train]
    #self.X = np.array([self.tokenizer.convert_tokens_to_indexes(txt) for txt in tokenized_X_train])
    #self.Y = np.array([self.tokenizer.convert_tokens_to_indexes(txt) for txt in tokenized_Y_train])

    self.X = np.array([self.tokenizer.convert_tokens_to_bertids(txt) for txt in tokenized_X_train])
    self.Y = np.array([self.tokenizer.convert_tokens_to_bertids(txt) for txt in tokenized_Y_train])

    self.batch_size = batch_size
    
  def getMaxLength(self, tokenized_texts):
    max_len = 0 
    for x in tokenized_texts: 
      if len(x)>max_len  : 
        max_len = len(x)
    return max_len 
  
  def pad_sequences(self, batches): 
    def _pad_sequences_(batch):
        max_l = self.getMaxLength(batch)
        for i in range(len(batch)):
          while (len(batch[i])<max_l): 
            batch[i].append(0) 
        return batch 
    batches_padded_tensors = []
    for batch in batches : 
      padded_batch = _pad_sequences_(batch) 
      batches_padded_tensors.append(torch.stack([ torch.tensor(padded_batch[j]) for j in range(len(padded_batch)) ]))
    return batches_padded_tensors
    
  def constract_batches(self) : 
    batches_X = []
    batches_Y = []
    for _ in range(len(self.Y)//self.batch_size):
      ind = np.random.randint(len(self.X)-self.batch_size)
      batches_X.append(self.X[ind:ind+self.batch_size])
      batches_Y.append(self.Y[ind:ind+self.batch_size])

    batches_X_tensors = self.pad_sequences(batches_X)
    batches_Y_tensors = self.pad_sequences(batches_Y)
    return (batches_X_tensors, batches_Y_tensors)



def load_files(src,target):
  # load the text sentences and the target sentences 
  # target could be a summary or a translation of text.  
  with open(src, 'r') as f:
    src_sentences = f.read().split('\n')
  with open(target, 'r') as f:
    target_sentences = f.read().split('\n')
  return src_sentences, target_sentences

 
