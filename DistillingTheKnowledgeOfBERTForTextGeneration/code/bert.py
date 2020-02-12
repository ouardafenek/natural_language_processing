from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert import BertAdam 
from preprocessing import * 

def bert_train(inputs, token_type_ids, masked_lm_labels):
	#Pre_training & Fine tuning Bert for text Generation 
	model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	# hack to remove pooler, which is not used
	# thus it produce None grad that break apex
	param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	        ]
	optimizer = BertAdam(optimizer_grouped_parameters,
	                             lr=5e-5,
	                             warmup=0.1,
	                             t_total=300)
	model.train()
	n_steps = 10 
	n_batches = len(inputs)
	for epoch in range(0,n_steps): #(0,2)
	  eveloss = 0
	  for i in range (n_batches) :  # (1):
	    loss = model(inputs[i], token_type_ids = token_type_ids[i], masked_lm_labels =masked_lm_labels[i] )
	    eveloss += loss.mean().item()
	    loss.backward()
	    optimizer.step()
	    optimizer.zero_grad()
	  print("step "+ str(epoch) + " : " + str(eveloss))
	return model 

def bert_predict(model, vocab_size, batch_X, batch_Y, mask_percentage): 

  batch_Y_masked , _ = mask(mask_percentage,batch_Y.clone())
  batch_Y_masked_tensor = torch.stack([t for t in batch_Y_masked])
 
  input_for_bert = torch.cat([batch_X.clone(),batch_Y_masked_tensor],1)
  
  x_length = batch_X.shape[1]
  model.eval() 
  with torch.no_grad():
    predictions = model(input_for_bert)
    # predictions are of shape: Batch x  Length ((X)concatenated to (Y)) x Vocab Size
    pred_bert = predictions[:,x_length:,:]
    #print("Now we have to get the word predicted by taking the one whose probability is the highest")
    y_pred_bert = torch.argmax(pred_bert.clone(),dim=2)
    return y_pred_bert