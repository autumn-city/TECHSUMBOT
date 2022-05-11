from locale import delocalize
import os
from pickle import TRUE
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel,BertForSequenceClassification, BertTokenizer, BertModel,RobertaForSequenceClassification
import pickle
from collections import OrderedDict  
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from lexrank import STOPWORDS, LexRank
import _3_module.summa.summarizer as summarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity



def read_correspond_answers(file_path):

    corr_answers = {}
    path = file_path
    with open(path, 'rb')as f:
        new_dict = pickle.load(f)
    corr_answer = []
    for order, item in enumerate(new_dict):
        SO_AnswerUnit_tmp = [item[0], item[3:], item[1], item[2]]
        corr_answer.append(SO_AnswerUnit_tmp)
    # corr_ans .wers[q_id] = corr_answer
    # print(corr_answer)
    return corr_answer


def answer_score(query, answer):

    encoding = tokenizer(
    query,
    answer,
    padding="max_length", 
    truncation=True,
    max_length = max_length,
    return_tensors='pt',
    )
    # print(tokenizer.decode(encoding["input_ids"]))
    
    outputs = model(**encoding)

    outputs = outputs[0]
    n_cls = outputs.size()[-1]
    if n_cls == 2:
        outputs = F.softmax(outputs, dim=-1)[:, 1]

    rel_scores = outputs.cpu().detach().numpy()  # d_batch,
    return rel_scores

class SO_Ans:
    __slots__ = 'id', 'body', 'score', 'parent_id'

    def __init__(self, id, body, score, parent_id):
        self.id = id
        self.body = body
        self.score = score
        self.parent_id = parent_id


def sentence_similarity_score(sent1, sent2):

    cosine_sim_0_1 = 1 - cosine(sent1, sent2)

    return cosine_sim_0_1


def sentence_similarity(sent1, sent2):
    texts = []
    texts.append(sent1)
    texts.append(sent2)

    # texts = [
    #     "There's a kid on a skateboard.",
    #     "A kid is skateboarding.",
    # ]
    
    inputs = sim_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = sim_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])

    # print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
    return cosine_sim_0_1

def get_sentence_simcsse_similarity(sent_list):
    input = []

    for item in sent_list:
        input.append(item[0])

    inputs = sim_tokenizer(input, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = sim_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings


def get_counts(sents):
    count_vec = CountVectorizer()
    counts = count_vec.fit_transform(sents)
    return counts

def get_sentence_tfidf_similarity(sent_list):
    input = []
    for item in sent_list:
        input.append(item[0])
    counts = get_counts(input)
    # print(counts);exit()
    tf_transformer = TfidfTransformer()
    tf_mat = tf_transformer.fit_transform(counts)
    return tf_mat

def lexrank(input):

    summary = lxr.get_summary(input, summary_size=5, threshold=0.1)

    return summary




def summarize(candidate, removed, filename, topk, algorithm):
    input = []
    for item in candidate:
        input.append(item[0])

    if len(candidate)>topk:

        input = input[:topk]

    # select the summarize algorithm between textrank and lexrank
    if algorithm == 'lexrarnk':
        result = lexrank(input)
        if not os.path.exists('result/lexrank_'+str(threshold)+'_top'+str(topk)+'/'):
            os.mkdir('result/lexrank_'+str(threshold)+'_top'+str(topk)+'/')
        with open('result/lexrank_'+str(threshold)+'_top'+str(topk)+'/'+filename[:-4]+'.txt','w') as f:
            for sent in result:
                f.write(sent+'\n')
    if algorithm == 'textrank':
        result = summarizer.summarize(input,sent_length=5)

        if not os.path.exists('result/texrank_'+str(sim_algorithm)+'_'+str(threshold)+'_top'+str(topk)+'/'):
            os.mkdir('result/texrank_'+str(sim_algorithm)+'_'+str(threshold)+'_top'+str(topk)+'/')
        with open('result/texrank_'+str(sim_algorithm)+'_'+str(threshold)+'_top'+str(topk)+'/'+filename[:-4]+'.txt','w') as f:
            for sent in result:
                f.write(sent+'\n')
            # f.writelines(result)    


if __name__ == "__main__":

    # module 1 model setting 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    max_length = 64
    bert_model = "bert-base-uncased"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('_1_module/model/aspn_fine_tuned/models/tanda_roberta_base_asnq/update_model')
    model.eval()

    # module 2 model setting
    sim_tokenizer = AutoTokenizer.from_pretrained("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")
    sim_model = AutoModel.from_pretrained("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")
    # print(sentence_similarity(text1, text2));exit(0)

    # threshold for the hyper-parameter
    threshold = 1
    topk = 15
    #lexrank textrank
    summarize_algorithm='textrank'
    # tfidf;simcse
    sim_algorithm = 'simcse'

    # get the data 
    data_dir = "../dataset/input/json/"
    for candidate in os.listdir(data_dir):


        # candidate = "24_What's the difference between \"2*2\" and \"2**2\" in Python?.pkl"

        # start module_1
        # module_1_result store each sentence in the answer and their score for QA system
        module_1_result = {}
        
        query=candidate.split('_')[1][:-4]
        file_path = data_dir+candidate
        answerlist = read_correspond_answers(file_path)
        '''
        answers list is a list for relevant answers
        each row [answer id, *answer sentences, votes, question id]
        '''
        for answer in answerlist:
            for sent in answer[1]:            
                with torch.no_grad():
                    # query first, then answers
                    sent_score = answer_score(query,sent)
                    module_1_result[sent]=sent_score.astype(float)[0]
        module_1_result = sorted(module_1_result.items(),key = lambda x:x[1],reverse = True)
        # module_1_result = OrderedDict(module_1_result)
        # print(module_1_result)
        print('Dic length %s; top: %s; sim: %s; sim algorithn: %s; summarization algorithm: %s'%(str(len(module_1_result)),topk,threshold,sim_algorithm,summarize_algorithm))

        # start module_2 (we've already get the sorted result for module-1)
        # set the sentence with highest score as the groundtruth, delete all sentences similar with it, then loop for top-10 un-deleted sentences
        
        #
        # test mudule_1 只用前一半
        # 
        module_1_result = list(module_1_result)

        copy = module_1_result
        removed = []
        rest = []

        if sim_algorithm =='simcse':
            embedding = get_sentence_simcsse_similarity(copy).tolist()
        if sim_algorithm =='tfidf':
            embedding = get_sentence_tfidf_similarity(copy)
            del_copy = []
        # print(len(embedding));exit()



        for index_1, ground_truth in enumerate(embedding):
            if sim_algorithm =='tfidf':
                if index_1 in del_copy:
                    continue
            for index_2, sent in enumerate(embedding):
                if index_1!=index_2:
                    if sim_algorithm == 'simcse':
                        sim_score = sentence_similarity_score(ground_truth, sent)
                    if sim_algorithm == 'tfidf':
                        sim_score = cosine_similarity(ground_truth, sent)[0][0]
                    # print(float(sim_score));exit(0)
                    if (float(sim_score) >threshold) and (float(sim_score)<1):
                        removed.append(copy[index_2][0])
                        if sim_algorithm =='simcse':
                            del copy[index_2]   
                            del embedding[index_2]
                            print('del')
                        if sim_algorithm =='tfidf':
                            del_copy.append(copy[index_2])
                            print(float(sim_score))
                            print(copy[index_2])
                            print(copy[index_1])

        if sim_algorithm =='tfidf':
            # for item in del_copy:
            #     for index, item_1 in enumerate(copy):
            #         if item_1==item:
            #             del copy[index]   
            copy = [i for i in copy if i not in del_copy]


        if summarize_algorithm =='lexrank':
            documents_dir = '../dataset/input/json'
            documents = []            
            for file in os.listdir(documents_dir): 
                with open(os.path.join(documents_dir,file),'rb') as f:
                    new_dict = pickle.load(f)
                    for order, item in enumerate(new_dict):
                        SO_AnswerUnit_tmp = SO_Ans(item[0], item[3:], item[1], item[2])
                        documents.append(SO_AnswerUnit_tmp.body)

            lxr = LexRank(documents, stopwords=STOPWORDS['en'])

        summarize(copy,removed,candidate, topk, summarize_algorithm)