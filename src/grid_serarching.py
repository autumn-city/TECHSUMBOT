# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
from tkinter.tix import InputOnly
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel,BertForSequenceClassification, BertTokenizer, BertModel,RobertaForSequenceClassification
import transformers
transformers.logging.set_verbosity_error()
import pickle
from collections import OrderedDict  
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from lexrank import STOPWORDS, LexRank
import _3_module.summa.summarizer as summarizer
from _3_module.summa.preprocessing.textcleaner import clean_text_by_sentences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from math import log10
import numpy 


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

def get_sentence_simcsse_similarity(input):

    inputs = sim_tokenizer(input, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = sim_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def get_sentence_simcsse_similarity_update(input):
    embeddings = []
    # for sent in input:
    #     tokenized_sent = sim_tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
    #     with torch.no_grad():
    #         embedding = sim_model(**tokenized_sent, output_hidden_states=True, return_dict=True).pooler_output
    #     embeddings.append(embedding)
    from simcse import SimCSE
    model = SimCSE("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")
    for sent in input:
        embedding = model.encode(sent)
        # if sent == 'commit() writes the data synchronously (blocking the thread its called from).':
        #     print(embedding)
        #     print(len(embeddings));exit()
        embeddings.append(embedding)
    return embeddings


def get_counts(sents):
    count_vec = CountVectorizer()
    counts = count_vec.fit_transform(sents)
    return counts

def get_sentence_tfidf_similarity(input):
    counts = get_counts(input)
    # print(counts);exit()
    tf_transformer = TfidfTransformer()
    tf_mat = tf_transformer.fit_transform(counts)
    return tf_mat

def lexrank(input):

    summary = lxr.get_summary(input, summary_size=5, threshold=0.1)

    return summary

def centralize_topk(candidate, filename, topk, algorithm,embedding_algorithm):

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


    input = []
    for item in candidate:
        input.append(item[0])

    if len(candidate)>topk:

        input = input[:topk]

    # select the summarize algorithm between textrank and lexrank
    if algorithm == 'lexrarnk':
        result = lexrank(input)
        write_summarize(result, filename,algorithm)

    if algorithm == 'textrank':
        # return the centrlized sentence list, add 'sent_length' to limit the number of sentence
        # embedding: simcse / tfidf
        result = summarizer.summarize(input,embedding=embedding_algorithm)
        return result

def write_summarize(summary,filename,redundancy_threshold,topk,sim_algorithm,embedding_algorithm):
    summarize_path = 'result/redundancy_'+str(redundancy_threshold)+'_topk_'+str(topk)+'_redundancyalgorithm_'+str(sim_algorithm)+'_summarizationalgorithm_'+str(embedding_algorithm)+'/'
    if not os.path.exists(summarize_path):
        os.mkdir(summarize_path)
    with open(summarize_path+filename[:-4]+'.txt','w') as f:
        for sent in summary:
            f.write(sent+'\n')


def reduce_redundancy(input,redundancy_threshold):
    # get the sentences from <sentence, score>
    sents = []

    for item in input:
        sents.append(item[0])

    # get the semantic embedding
    if sim_algorithm =='simcse':
        # embedding = get_sentence_simcsse_similarity(sents).tolist()
        embedding = get_sentence_simcsse_similarity_update(sents)
        
    if sim_algorithm =='tfidf':
        input = [item[0] for item in input]
        embedding = get_sentence_tfidf_similarity(sents)
    
    # iteratively get the non-redundant sentences
    summary = []
    summary_embedding = []

    # print(sentence_similarity_score(embedding[7], embedding[10]));exit()

    for index_1, ground_truth in enumerate(embedding):
        flag = False

        # 第一句是7
        # if index_1 ==10:
        #     from simcse import SimCSE
        #     model = SimCSE("_2_module/SimCSE/result/my-sup-simcse-bert-base-uncased/")
        #     emd = model.encode('\"commit() returns true if the save works, false otherwise.', device=None, return_numpy=False, normalize_to_unit=True, keepdim=False, max_length=128)
        #     print('-'+str(sents[10]+'-'))
        #     print(sentence_similarity_score(emd, embedding[10]));exit()

        for index_2, sent in enumerate(summary_embedding):

            if sim_algorithm =='simcse':
                sim_score = sentence_similarity_score(ground_truth, sent)
            if sim_algorithm =='tfidf':
                sim_score = cosine_similarity(ground_truth, sent)
            if sim_score>redundancy_threshold:
                print('===========================\nThe redundant sentences pair is shown below:')
                print(sents[index_1])
                print(summary[index_2])
                print('the similarity score of both sentences are %s'%(str(sim_score)))
                print('\n\n\n')
                flag = True
        if not flag:                
            summary.append(sents[index_1])
            summary_embedding.append(embedding[index_1])
    return summary[:5]




def summarize(candidate,redundancy_threshold,topk,sim_algorithm,embedding_algorithm):
    # candidate = "24_What's the difference between \"2*2\" and \"2**2\" in Python?.pkl"

    # start module_1
    # module_1_result store each sentence in the answer and their score for QA system
    '''
    answers list is a list for relevant answers
    each row [answer id, *answer sentences, votes, question id]
    '''
    module_1_result = {}
    query=candidate.split('_')[1][:-4]
    file_path = data_dir+candidate
    answerlist = read_correspond_answers(file_path)

    for answer in answerlist:
        for sent in answer[1]:            
            with torch.no_grad():
                # query first, then answers
                sent_score = answer_score(query,sent)
                module_1_result[sent]=sent_score.astype(float)[0]
    module_1_result = sorted(module_1_result.items(),key = lambda x:x[1],reverse = True)
    # print('Dic length %s; top: %s; sim: %s; sim algorithn: %s; summarization algorithm: %s'%(str(len(module_1_result)),topk,threshold,sim_algorithm,summarize_algorithm))

    # start module_2 (we've already get the sorted result for module-1)
    # set the sentence with highest score as the groundtruth, delete all sentences similar with it, then loop for top-10 un-deleted sentences
    module_1_result = list(module_1_result)
    copy = module_1_result

    # module for centralize: return the list ranking with centralize score
    centralized_list = centralize_topk(copy,candidate, topk, summarize_algorithm,embedding_algorithm)

    # module for reducing redundancy
    summary = reduce_redundancy(centralized_list,redundancy_threshold)
    write_summarize(summary, candidate,redundancy_threshold,topk,sim_algorithm,embedding_algorithm)

def main():
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
    # test();exit(0)

    # threshold for the hyper-parameter
    threshold = 1
    topks = [x for x in range(15,31,5)]
    #lexrank textrank
    summarize_algorithm='textrank'
    # sim algorithm for removing redundancy : tfidf;simcse
    # sim_algorithms = ['simcse','tfidf']
    sim_algorithms = ['simcse']
    # embedding algorithm for input of textrank : simcse/tfidf
    # embedding_algorithms = ['tfidf','simcse']
    embedding_algorithms = ['simcse']

    # redundancy_threshold
    redundancy_thresholds = [round(x,2) for x in numpy.arange(0.6,1.01,0.05) ]
    # get the data 
    data_dir = "../dataset/input/json/"
    
    
    for candidate in os.listdir(data_dir):
        for topk in topks:
            for sim_algorithm in sim_algorithms:
                for  embedding_algorithm in embedding_algorithms:
                    for redundancy_threshold in redundancy_thresholds:
                        summarize(candidate,redundancy_threshold,topk,sim_algorithm,embedding_algorithm)



if __name__ == "__main__": 
    main()
