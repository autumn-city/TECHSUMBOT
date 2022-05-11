# -*- coding: utf-8 -*-
from pandas.io import clipboards
import pymysql 
import pandas as pd 
import pickle
import signal
import time
# -- start mysql use: mysql --local-infile=1 -u root -p
# -- password: 123456
def sql_connection():
   db = pymysql.connect(host='localhost',
                        user='root',
                        password='123456',
                        database='SO')

   cursor = db.cursor(pymysql.cursors.DictCursor)
   return cursor

def random_select_one_data(attribute):

   cursor = sql_connection()
   sql = '''
SELECT '''+str(attribute)+''',id FROM posts 
WHERE id >= FLOOR(71282374 * RAND()) 
and PostTypeId=1
LIMIT 1;
   '''
   # sql = '''
   # # SELECT title, id FROM posts ORDER BY RAND() LIMIT 1
   # '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

def set_timeout(num, callback):
  def wrap(func):
    def handle(signum, frame): # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
      raise RuntimeError
    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle) # 设置信号和回调函数
        signal.alarm(num) # 设置 num 秒的闹钟
      #   print('start alarm signal.')
        r = func(*args, **kwargs)
      #   print('close alarm signal.')
        signal.alarm(0) # 关闭闹钟
        return r
      except RuntimeError as e:
        callback()
    return to_do
  return wrap

def after_timeout(): # 超时后的处理函数
  print("Time out!")

@set_timeout(3, after_timeout) # 限时 2 秒超时
def get_query_name(Id):
   cursor = sql_connection()
   sql = '''
   select Title from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_parientid(Id):
   cursor = sql_connection()
   sql = '''
   select ParentId from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()
   if results:
      return results
   else: 
      return 'Null'

@set_timeout(3, after_timeout) # 限时 2 秒超时
def get_votes(Id):
   cursor = sql_connection()
   sql = '''
   select Score from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_query_body(Id):
   cursor = sql_connection()
   sql = '''
   select Body from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results



@set_timeout(3, after_timeout) # 限时 2 秒超时
def get_answer_body(Id):
   cursor = sql_connection()
   sql = '''
   select Body from posts
   where Id= '''+Id+''' 
   and PostTypeId  = 2'''
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_answer_votes(Id):
   cursor = sql_connection()
   sql = '''
   select Title from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_all_java_answers():
   cursor = sql_connection()

   sql = '''
   select id,title from posts where 
   PostTypeId = 1
   and tags like '%<java>%'
   ''' 

   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   # for row in results:
   #     fname = row[0]
   #     lname = row[1]
   #     # 打印结果
   #     print ("fname=%s,lname=%s" % \
   #             (fname, lname))
   # except:
   #    print ("Error: unable to fetch data")
   return results
def get_all_python_answers():
   cursor = sql_connection()

   sql = '''
   select id,title from posts where 
   PostTypeId = 1
   and tags like '%<python>%'
   ''' 

   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()

   return results

def get_relatedID(id,pl):
   cursor = sql_connection()
   sql = '''
   select b.PostId
   from posts as a, post_links as b
   where b.RelatedPostId='''+str(id)+'''
   and b.PostId=a.Id
   and a.Tags like '%'''+str(pl)+'''%'
   and b.LinkTypeId = 3
   group by b.PostId
   '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results
   # data = pd.DataFrame(list(results),columns=['Original post ID','number of related answers'])
   # data.to_csv('data.csv',index=False)
   # print(data)

def get_related_posts(pl):
   cursor = sql_connection()
   sql = '''select b.RelatedPostId,
   (sum(a.AnswerCount)+(select a.AnswerCount from posts as a where a.Id=b.RelatedPostId))TotalAnswerCount
   from posts as a, post_links as b 
   where b.PostId=a.Id 
   and a.Tags like '%<'''+str(pl)+'''>%' 
   and b.LinkTypeId = 3  
   group by b.RelatedPostId 
   order by (sum(a.AnswerCount)+(select a.AnswerCount from posts as a where a.Id=b.RelatedPostId))  desc;'''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results


def get_data(id):
   cursor = sql_connection()
   sql = '''select * from posts
   where ParentId ='''+str(id)
   cursor.execute(sql)
   result = cursor.fetchall()
   return result

def get_id(file):
   java_candidate = file[(file['TotalAnswerCount']>=10) & (file['TotalAnswerCount']<=18)]
   # candidate 
   # [RelatedPostId, TotalAnswerCount]
   return java_candidate

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_duplicate_pairs(pl):
   cursor = sql_connection()
   sql = '''
   select  a.RelatedPostId,a.postid  from post_links as a, posts as b  
   where a.LinkTypeId = 3
   and a.PostId=b.Id 
   and b.Tags like '%<'''+pl+'''>%'
   '''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

@set_timeout(3, after_timeout) # 限时 2 秒超时
def get_post_tags(id):

   cursor = sql_connection()
   sql = '''
   select tags from posts
   where id ='''+id+'''
   and PostTypeId = 1
   '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

def get_tag_posts(tag):
   cursor = sql_connection()
   sql = '''
   select id, AcceptedAnswerId from posts where posttypeid=1 
   and tags like '%<'''+str(tag)+'''>%' 
   and AcceptedAnswerId is not null 
   '''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

@set_timeout(3, after_timeout) # 限时 2 秒超时
def get_neg_answer(parentid):
   cursor = sql_connection()
   sql = '''
   select body, id,score from posts where ParentId = '''+parentid+''' 
   and PostTypeId  = 2 
   and Score<0
   '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results