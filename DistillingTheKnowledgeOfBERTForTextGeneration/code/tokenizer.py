from pytorch_pretrained_bert import BertTokenizer
class MyTokenizer:
    def __init__(self, name):
        self.name = name
        #we start from 1 to keep the 0 for padding 
        self.word2index = {'[CLS]': 1, '[SEP]':2, '[MASK]':3}
        self.index2word = {1: "[CLS]", 2: "[SEP]", 3:"[MASK]"}

        self.n_words = 3  # Count CLS and SEP and MASK 

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bertid2word = {101: "[CLS]", 102: "[SEP]", 103:"[MASK]"} 
        self.word2bertid = {'[CLS]': 101, '[SEP]':102, '[MASK]':103} 

    def addSentence(self, sentence):
        words = self.tokenizer.tokenize(sentence) 
        for word in words :
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
          
            self.word2bertid[word] = self.tokenizer.convert_tokens_to_ids([word])[0]
            self.bertid2word[self.tokenizer.convert_tokens_to_ids([word])[0]] = word

            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        
    def tokenize(self,sentence):
      return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_bertids(self,words): 
      return [self.word2bertid[word] for word in words]
    
    def convert_tokens_to_indexes(self,words):
      return [self.word2index[word] for word in words]
    
    def from_bert_id_to_index_id(self,bert_id): 
      return word2index[bertid2word[bert_id]] 

    def convert_bertids_to_tokens(self,bertids):
      return self.tokenizer.convert_ids_to_tokens(bertids)

def construct_vocabulary(src_sentences,target_sentences): 
  # Bert uses WordPiece which is a huge corpus, 
  # due to lack of memory, we will reduce the vocab size to 
  # the vocabulary of our corpus 
  # That's why we built our own tokenizer rather than directly using Bert's tokenizer
  # But if you have enough memory, we would rather advice you to keep the initial vocab_size
  myTokenizer = MyTokenizer("myTokenizer")
  for i,x in enumerate(src_sentences): 
    myTokenizer.addSentence(x)
    myTokenizer.addSentence(target_sentences[i])
  return myTokenizer
