import numpy as np
import random 
import torch


# This File contains the function used to adapt the data to the bert format and to the description of the paper. 

def mask(percentage, Y):
  # Masks Percentage % of tokens from Y 
  # Returns an y with some masked tokens, and a variable masked_lm_labels which has the same shape as Y 
  # It contains -1 where Y hadn't been modified, the id of the masked token elsewhere
  masked_data = []
  masked_lm_labels = []
  for sentence in Y: 
    masked_lm_label = [-1] * len(sentence)
    indexes = random.sample(range(len(sentence)), int(percentage*len(sentence)))
    for i in indexes: 
      masked_lm_label[i] = sentence[i] #Contient les lettres cachées
      sentence[i] = 2 
      
    masked_data.append(sentence)
    masked_lm_labels.append(masked_lm_label)
  return masked_data, masked_lm_labels


def bert_format(src_sentences, target_sentences, myTokenizer, mask_percentage): 
  # This function processes the inputs to be convinient to Bert's inputs. 
  # It adds CLS at the begining of the sentences and SEP at their ends. 
  # It tokenizes the sentences and convert their tokens to ids 
  # It masks mask_percentage % of the target sentence (as it has been described in the paper that we are implementing)
  # It concatenates the source tokens with the target tokens of whome we have masked some in the variable data 
  # It constructs the token_type_ids witch defines A and B indices associated to first and second sentence (see the paper of BERT)
  # It constructs masked_lm_labels variable which is formed of the real id token that have been masked, -1 elsewhere. 

  X_train = ["[CLS] " + x + " [SEP]"  for x in src_sentences]
  Y_train = ["[CLS] " + y + " [SEP]"  for y in target_sentences]
  tokenized_X_train = [myTokenizer.tokenize(sent) for sent in X_train]
  tokenized_Y_train = [myTokenizer.tokenize(sent) for sent in Y_train]

  #X = [myTokenizer.convert_tokens_to_indexes(txt) for txt in tokenized_X_train]
  #Y = [myTokenizer.convert_tokens_to_indexes(txt) for txt in tokenized_Y_train]
 
  X = [myTokenizer.convert_tokens_to_bertids(txt) for txt in tokenized_X_train]
  Y = [myTokenizer.convert_tokens_to_bertids(txt) for txt in tokenized_Y_train]
 
  masked_Y, masked_lm_labels  = mask(mask_percentage,Y)
  
  # concatenation 
  data = np.array([np.array(X[i]+masked_Y[i][1:], dtype=np.int64) for i in range(len(X))])

  token_type_ids = np.array([np.concatenate((np.zeros(len(X[i]), dtype=np.int64),np.ones(len(Y[i])-1,dtype=np.int64))) for i in range(len(X))])
 
  masked_lm_labels = np.array([np.concatenate((-1* np.ones(len(X[i]),dtype=np.int64) ,np.array(masked_lm_labels[i][1:])) ) for i in range(len(X))])

  return X, Y, data, token_type_ids, masked_lm_labels 



def convert_to_tensors (batches):
  batches_tensors = []
  for batch in batches : 
    batches_tensors.append(torch.stack([ torch.tensor(batch[i]) for i in range(len(batch)) ]))
  return batches_tensors 


