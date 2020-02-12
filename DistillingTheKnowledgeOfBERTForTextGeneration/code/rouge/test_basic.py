from unittest import TestCase

import rouge
import json


class BasicTest(TestCase):
    def setUp(self):
        self.hyp_path = 'predicted_sentences_task1_ref0_2.txt'
        self.ref_path = 'task1_ref0_2.txt'

        self.data_path = 'data2.json'
        with open(self.data_path) as f:
            self.data = json.load(f)

        self.rouge = rouge.Rouge()
        self.files_rouge = rouge.FilesRouge()

    def test_one_sentence(self):


        for d in self.data[:1]:
            hyp = d["hyp"]
            ref = d["ref"]
            score = self.rouge.get_scores(hyp, ref)[0]
            print("score",score)
            

    def test_multi_sentence(self):
        data = self.data
        hyps, refs = map(list, zip(*[[d['hyp'], d['ref']] for d in data]))
       
        scores = self.rouge.get_scores(hyps, refs)
        print("scores: ",scores)
       

    def test_files_scores(self):
        data = self.data
        hyps, refs = map(list, zip(*[[d['hyp'], d['ref']] for d in data]))
        scores = self.files_rouge.get_scores(self.hyp_path, self.ref_path)
        #print("scores: ",scores)
        for i, score in enumerate (scores): 
            self.data[i]["scores"] = score 

        with open("data2.json", "w") as f_write: 
            json.dump(self.data,f_write, indent =4)

     

def create_data(hyp, ref):
    datas = []

    f_hyp = open(hyp,"r")
    f_ref = open(ref,"r")
    hyp_lines = f_hyp.readlines()
    ref_lines = f_ref.readlines() 
    for i in range(len(ref_lines)): 
        data = dict() 
        data["hyp"] = hyp_lines[i]
        data["ref"] = ref_lines[i]
        data["scores"] = dict() 
        datas.append(data)

    with open("data2.json", "w") as f_write: 
        json.dump(datas,f_write)

create_data("predicted_sentences_task1_ref0_2.txt","task1_ref0_2.txt")

basicTest = BasicTest()
basicTest.setUp() 
basicTest.test_files_scores()