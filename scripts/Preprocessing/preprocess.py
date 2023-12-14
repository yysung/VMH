# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:10:11 2021

@author: YY
"""

import csv
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def chunks(df, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]
        
def invite_nonacc_df_process(url):
    df = pd.read_csv(url)
    df_ = df[['HITId','WorkerId','WorkTimeInSeconds','Input.headline1','Input.video1','Input.headline2','Input.video2','Answer.taskAnswers']]
    final_dff2 = df_[['HITId','WorkerId','Input.headline2','Input.video2','Answer.taskAnswers']]
    print('length of dataframe: ',len(df_))
    print('length of time spent', df_[['WorkTimeInSeconds']].mean()/60)
    return df, final_dff2

def invite_nonacc_process_label(df_):
    
    final_df = pd.DataFrame()
    #first video
    k = 2
    headlines = []
    answer_lists = []
    properties = []
    video_urls = []
    q4, q5, q6_input,q7_input, q9, q10, q11,q12 = ([] for i in range(8))

    for i in range(len(df_)):

        headline = df_[['Input.headline{}'.format(k)]].iloc[i].values.tolist()[0]
        video_url = df_[['Input.video{}'.format(k)]].iloc[i].values.tolist()[0]
        #print(headline)
        answer = json.loads(df_[['Answer.taskAnswers']].iloc[i].values.tolist()[0])[0]
        q1 = [key for key,value in answer['p{}_question1'.format(k)].items() if value==True]
        #print(q1)
        if q1[0]=='Statement':
            q2 = [key for key,value in answer['p{}_question2'.format(k)].items() if value==True]
            if len(q2) > 0:
                prop = q2[0]+' '+q1[0]
            else:
                prop= 'not a statement'
        elif q1[0]=='Question': 
            q3 = [key for key,value in answer['p{}_question3'.format(k)].items() if value==True]
            prop = q3[0]+' '+q1[0]
        #print(prop)
        #print(video_url)


        if prop == 'Factual Statement':
            #before watching the video
            q4 = [key for key,value in answer['p{}_question4'.format(k)].items() if value==True]
            if len(q4)==0:
                q4=['na']
            #after watching the video
            q9 = [key for key,value in answer['p{}_question9'.format(k)].items() if value==True]
            if len(q9)==0:
                q9=['na']
        elif prop == 'Opinionated Statement':
            q5 = [key for key,value in answer['p{}_question5'.format(k)].items() if value==True]
            if len(q5)==0:
                q5=['na']
            q10 = [key for key,value in answer['p{}_question10'.format(k)].items() if value==True]
            if len(q10)==0:
                q10=['na']
        elif prop == 'Neither Statement':
            q6_input = [answer['p{}_question6_input'.format(k)]]
            if len(q6_input)==0:
                q6_input=['na']
            q11 = [key for key,value in answer['p{}_question11'.format(k)].items() if value==True]
            if len(q11)==0:
                q11=['na']

        elif prop == 'Factual Question':
            q7_input = [answer['p{}_question7_input'.format(k)]]
            if len(q7_input)==0:
                q7_input=['na']
            q12 = [key for key,value in answer['p{}_question12'.format(k)].items() if value==True]
            if len(q12)==0:
                q12=['na']
        elif prop == 'Opinionated Question':
            q7_input = [answer['p{}_question7_input'.format(k)]]
            if len(q7_input)==0:
                q7_input=['na']
            q12 = [key for key,value in answer['p{}_question12'.format(k)].items() if value==True]
            if len(q12)==0:
                q12=['na']
        else:
            print('error')

        if len(q4)==0:
            q4=['na']
        if len(q9)==0:
            q9=['na']
        if len(q5)==0:
            q5=['na']
        if len(q10)==0:
            q10=['na']
        if len(q6_input)==0:
            q6_input=['na']
        if len(q11)==0:
            q11=['na']
        if len(q7_input)==0:
            q7_input=['na']
        if len(q12)==0:
            q12=['na']

        answer_list = q4+q5+q6_input+q7_input+q9+q10+q11+q12
        #print(answer_list)
        answer_lists.append(answer_list)
        answer_list = []
        q4, q5, q6_input,q7_input, q9, q10, q11,q12 = ([] for i in range(8))

        headlines.append(headline)
        properties.append(prop)
        video_urls.append(video_url)
        headline_df = pd.DataFrame(headlines)
        pro_df = pd.DataFrame(properties)
        url_df = pd.DataFrame(video_urls)
        answer_lists_df = pd.DataFrame(answer_lists)

        sfinal_df = pd.concat([url_df,headline_df, pro_df, answer_lists_df],axis=1)
        sfinal_df.columns = ['Video_urls','Headlines','Properties',
                             '(F.S) Based on your own knowledge, how would you rate the statement? If you do not know, select I don’t know.',
                             '(O.S) Do you have prior knowledge about the statement in the headline to make a judgment (e.g., agree/disagree) on the statement?',
                             '(N.S) Write down what you expect to see in a video',
                             '(Q) Write down what you expect to see in a video',
                             '(F.S) Based on the information provided in the video, how would you rate the statement? If you do not know, select I don’t know. Please rate the following statement solely based on the knowledge from the video.',
                             '(O.S) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video.  The video justifies the opinion in the headline.',
                             '(N.S) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video. The video talks about the statement',
                             '(Q) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video. The information provided by the video helps you answer the question in the headline.']
    final_df = sfinal_df
    print('length of df: ',len(final_df))

    part_df = final_df.iloc[:,7:11]
    answer = []
    for idx, row in part_df.iterrows():
        #print(row.values[np.where(row!='na')[0]][0])
        try:
            answer.append(row.values[np.where(row!='na')[0]][0])
        except:
            answer.append('Agree')
    answer_df = pd.DataFrame(answer)
    answer_df.columns = ['label']

    final_df['label'] = answer_df
    final_df['label'] = final_df['label'].replace(['False', 'Mostly False', 'Half True', 'Disagree','Somewhat Disagree','I do not know'],'misleading')
    final_df['label'] = final_df['label'].replace(['True','Mostly True','Agree','Somewhat Agree'],'leading')
    
    part_df = final_df.iloc[:,7:11]
    answer = []
    for idx, row in part_df.iterrows():
        #print(row.values[np.where(row!='na')[0]][0])
        try:
            answer.append(row.values[np.where(row!='na')[0]][0])
        except:
            answer.append('Agree')
    answer_df = pd.DataFrame(answer)
    answer_df.columns = ['label']
    
    final_df['label'] = answer_df
    final_df['label'] = final_df['label'].replace(['False', 'Mostly False', 'Half True', 'Disagree','Somewhat Disagree','I do not know'],'misleading')
    final_df['label'] = final_df['label'].replace(['True','Mostly True','Agree','Somewhat Agree'],'leading')
  
    return final_df
  
def process_label(df_):
    final_df = pd.DataFrame()
    #first video
    for k in range(1,3):
        headlines = []
        answer_lists = []
        properties = []
        video_urls = []
        q4, q5, q6_input,q7_input, q9, q10, q11,q12 = ([] for i in range(8))
        for i in range(len(df_)):

            headline = df_[['Input.headline{}'.format(k)]].iloc[i].values.tolist()[0]
            video_url = df_[['Input.video{}'.format(k)]].iloc[i].values.tolist()[0]
            print(headline)
            answer = json.loads(df_[['Answer.taskAnswers']].iloc[i].values.tolist()[0])[0]
            q1 = [key for key,value in answer['p{}_question1'.format(k)].items() if value==True]
            print(q1)
            if q1[0]=='Statement':
                q2 = [key for key,value in answer['p{}_question2'.format(k)].items() if value==True]
                if len(q2) > 0:
                    prop = q2[0]+' '+q1[0]
                else:
                    prop= 'not a statement'
            elif q1[0]=='Question': 
                q3 = [key for key,value in answer['p{}_question3'.format(k)].items() if value==True]
                prop = q3[0]+' '+q1[0]
            print(prop)
            print(video_url)


            if prop == 'Factual Statement':
                #before watching the video
                q4 = [key for key,value in answer['p{}_question4'.format(k)].items() if value==True]
                if len(q4)==0:
                    q4=['na']
                #after watching the video
                q9 = [key for key,value in answer['p{}_question9'.format(k)].items() if value==True]
                if len(q9)==0:
                    q9=['na']
            elif prop == 'Opinionated Statement':
                q5 = [key for key,value in answer['p{}_question5'.format(k)].items() if value==True]
                if len(q5)==0:
                    q5=['na']
                q10 = [key for key,value in answer['p{}_question10'.format(k)].items() if value==True]
                if len(q10)==0:
                    q10=['na']
            elif prop == 'Neither Statement':
                q6_input = [answer['p{}_question6_input'.format(k)]]
                if len(q6_input)==0:
                    q6_input=['na']
                q11 = [key for key,value in answer['p{}_question11'.format(k)].items() if value==True]
                if len(q11)==0:
                    q11=['na']

            elif prop == 'Factual Question':
                q7_input = [answer['p{}_question7_input'.format(k)]]
                if len(q7_input)==0:
                    q7_input=['na']
                q12 = [key for key,value in answer['p{}_question12'.format(k)].items() if value==True]
                if len(q12)==0:
                    q12=['na']
            elif prop == 'Opinionated Question':
                q7_input = [answer['p{}_question7_input'.format(k)]]
                if len(q7_input)==0:
                    q7_input=['na']
                q12 = [key for key,value in answer['p{}_question12'.format(k)].items() if value==True]
                if len(q12)==0:
                    q12=['na']
            else:
                print('error')

            if len(q4)==0:
                q4=['na']
            if len(q9)==0:
                q9=['na']
            if len(q5)==0:
                q5=['na']
            if len(q10)==0:
                q10=['na']
            if len(q6_input)==0:
                q6_input=['na']
            if len(q11)==0:
                q11=['na']
            if len(q7_input)==0:
                q7_input=['na']
            if len(q12)==0:
                q12=['na']

            answer_list = q4+q5+q6_input+q7_input+q9+q10+q11+q12
            print(answer_list)
            answer_lists.append(answer_list)
            answer_list = []
            q4, q5, q6_input,q7_input, q9, q10, q11,q12 = ([] for i in range(8))

            headlines.append(headline)
            properties.append(prop)
            video_urls.append(video_url)
        headline_df = pd.DataFrame(headlines)
        pro_df = pd.DataFrame(properties)
        url_df = pd.DataFrame(video_urls)
        answer_lists_df = pd.DataFrame(answer_lists)

        sfinal_df = pd.concat([url_df,headline_df, pro_df, answer_lists_df],axis=1)
        sfinal_df.columns = ['Video_urls','Headlines','Properties',
                             '(F.S) Based on your own knowledge, how would you rate the statement? If you do not know, select I don’t know.',
                             '(O.S) Do you have prior knowledge about the statement in the headline to make a judgment (e.g., agree/disagree) on the statement?',
                             '(N.S) Write down what you expect to see in a video',
                             '(Q) Write down what you expect to see in a video',
                             '(F.S) Based on the information provided in the video, how would you rate the statement? If you do not know, select I don’t know. Please rate the following statement solely based on the knowledge from the video.',
                             '(O.S) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video.  The video justifies the opinion in the headline.',
                             '(N.S) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video. The video talks about the statement',
                             '(Q) Assuming that the information provided by the video is correct, how would you rate the following statement? Please rate the following statement solely based on the knowledge from the video. The information provided by the video helps you answer the question in the headline.']
        final_df = pd.concat([final_df,sfinal_df],axis=0,ignore_index=True)
    print('length of df: ',len(final_df))
    
    part_df = final_df.iloc[:,7:11]
    answer = []
    for idx, row in part_df.iterrows():
        #print(row.values[np.where(row!='na')[0]][0])
        try:
            answer.append(row.values[np.where(row!='na')[0]][0])
        except:
            answer.append('Agree')
    answer_df = pd.DataFrame(answer)
    answer_df.columns = ['label']
    
    final_df['label'] = answer_df
    final_df['label'] = final_df['label'].replace(['False', 'Mostly False', 'Half True', 'Disagree','Somewhat Disagree','I do not know'],'misleading')
    final_df['label'] = final_df['label'].replace(['True','Mostly True','Agree','Somewhat Agree'],'leading')

    return part_df, final_df

def invite_nonacc_process_rationales(original_df, final_df):
    #cut = int(len(final_df['label'])/2)
    #df_label_1 =final_df['label'].iloc[:cut]
    #df_label_2 = final_df['label'].iloc[cut:]
    #df_label_2 = df_label_2.reset_index(drop=True)
    #label_12 = pd.concat([df_label_1,df_label_2],axis=1)
    #label_12.columns = ['label_1','label_2']
    #dff = pd.concat([original_df, label_12],axis=1)
    
    dff.columns = ['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward',
           'CreationTime', 'MaxAssignments', 'RequesterAnnotation',
           'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds',
           'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds',
           'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
           'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
           'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
           'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.headline1',
           'Input.video1', 'Input.headline2', 'Input.video2', 'Answer.taskAnswers',
           'Approve', 'Reject', 'label_1', 'label_2']
           
    label_2 = pd.DataFrame(final_df['label'])
    label_2.columns = ['label_2']
    
    dff = pd.concat([original_df, label_2],axis=1)
    rationale_lists = []
    q13, q15, q16, q14, q17, q17_input, q18, q18_input, q19, q19_input = ([] for i in range(10))
    k=2
    for i in range(len(dff)):
        answer = json.loads(dff[['Answer.taskAnswers']].iloc[i].values.tolist()[0])[0]
        label = dff['label_{}'.format(k)].iloc[i]

        if label == 'leading':
            q13 = [key for key,value in answer['p{}_question13'.format(k)].items() if value==True]
            if len(q13)==0:
                q13=['na']
            if q13[0] =='Yes':
                q15 = [key for key,value in answer['p{}_question15'.format(k)].items() if value==True]
                if len(q15)==0:
                    q15=['na']
            elif q13[0] == 'No':
                q16 = [key for key,value in answer['p{}_question16'.format(k)].items() if value==True]
                if len(q16)==0:
                    q16=['na']
        elif label == 'misleading':
            q14 = [key for key,value in answer['p{}_question14'.format(k)].items() if value==True]
            if len(q14)==0:
                q14=['na']

            q17 = [key for key,value in answer['p{}_question17'.format(k)].items() if value==True]
            if len(q17)==0:
                q17=['na']
            if 'p{}_question17_input'.format(k) in answer.keys():
                q17_input = [answer['p{}_question17_input'.format(k)]]
            else:
                q17_input=['na']                    

            q18 = [key for key,value in answer['p{}_question18'.format(k)].items() if value==True]
            if len(q18)==0:
                q18=['na']
            if 'p{}_question18_input'.format(k) in answer.keys():
                q18_input = [answer['p{}_question18_input'.format(k)]]
            else:
                q18_input=['na']

            q19 = [key for key,value in answer['p{}_question19'.format(k)].items() if value==True]
            if len(q19)==0:
                q19=['na']
            if 'p{}_question19_input'.format(k) in answer.keys():
                q19_input = [answer['p{}_question19_input'.format(k)]]
            else:
                q19_input=['na']

        if len(q13)==0:
            q13=['na']
        if len(q15)==0:
            q15=['na']
        if len(q16)==0:
            q16=['na']
        if len(q14)==0:
            q14=['na']

        if len(q17)==0:
            q17=['na']
        if len(q17_input)==0:
            q17_input=['na']

        if len(q18)==0:
            q18=['na']
        if len(q18_input)==0:
            q18_input=['na']

        if len(q19)==0:
            q19=['na']
        if len(q19_input)==0:
            q19_input=['na']


        rationale_list = q13+q15+q16+q14+q17+q17_input+q18+q18_input+q19+q19_input
        #print(rationale_list)
        rationale_lists.append(rationale_list)
        q13,q15,q16,q14,q17,q17_input, q18, q18_input, q19,q19_input = ([] for i in range(10))

    rationale_lists_df = pd.DataFrame(rationale_lists)
    rationale_lists_df.columns = ['(Leading)Was there anything other than what you expected in the video?','(Leading-Y)What would make the headline misleading? Can you revise/rephrase the headline to become misleading?',
                                      '(Leading-N)Write anything in the video that you would have liked to be mentioned in the headline.','(Misleading)Choose which option is correct about the video and the headline.',
                                      'Select why you thought this headline does not cover all the content of the video. If you have other reasons, please write them down.','Other reason1','Select why you thought the headline implies more than what is introduced in the video. If you have other reasons, please write them down. ','Other reason2',
                                      'Select why you thought the headline provides contradictory information of the video. If you have other reasons, please write them down. ','Other reason3']
    return rationale_lists_df

def process_survey(df_):
    survey_lists = []
    df_ = df1
    for k in range(1,3):
        for i in range(len(df_)):
            worker_id = [df_[['WorkerId']].iloc[i].values.tolist()[0]]
            #print(worker_id)
            answer = json.loads(df_[['Answer.taskAnswers']].iloc[i].values.tolist()[0])[0]

            q20 = [key for key,value in answer['p{}_question20'.format(k)].items() if value==True]
            if len(q20)==0:
                q20=['na'] 

            if 'p{}_question20_input'.format(k) in answer.keys():
                    q20_input = [answer['p{}_question20_input'.format(k)]]
            else:
                q20_input=['na']

            q21 = [key for key,value in answer['p{}_question21'.format(k)].items() if value==True]
            if len(q21)==0:
                q21=['na'] 
            q22 = [key for key,value in answer['p{}_question22'.format(k)].items() if value==True]
            if len(q22)==0:
                q22=['na'] 

            if 'p{}_question22_input'.format(k) in answer.keys():
                    q22_input = [answer['p{}_question22_input'.format(k)]]
            else:
                q22_input=['na']

            q23 = [key for key,value in answer['p{}_question23'.format(k)].items() if value==True]
            if len(q23)==0:
                q23=['na'] 

            survey_list = worker_id+q20+q20_input+q21+q22+q22_input+[q23]
            #print(survey_list)
            survey_lists.append(survey_list)
            worker_id,q20,q20_input,q21,q22,q22_input,q23 = ([] for i in range(7))

    survey1 = survey_lists[:int(len(survey_lists)/2)]
    survey1_df = pd.DataFrame(survey1)
    survey2 = survey_lists[int(len(survey_lists)/2):]
    survey2_df = pd.DataFrame(survey2)
    for i in range(len(survey1_df)):
        if survey1_df.loc[i][1] == 'na':
            assert survey1_df.loc[i][0] == survey2_df.loc[i][0], 'ID mismatch'
            survey1_df.loc[i] = survey2_df.loc[i]
        elif survey2_df.loc[i][1] == 'na':
            assert survey2_df.loc[i][0] == survey1_df.loc[i][0], 'ID mismatch'
            survey2_df.loc[i] = survey1_df.loc[i]
    return survey2_df 

def tryout_labels(acc_df, test_df):
    #labels in acc data 
    print('accuracy check: ',acc_df['label'].value_counts().to_dict())

    #labels in non-acc data
    acc_idx = [i for i,value in enumerate(acc_df['label'].values.tolist()) if value=='misleading'] 
    print(test_df.iloc[acc_idx]['label'].value_counts().to_dict())

def invite_labels(df):
    #labels in data 
    print('non-accuracy check:', df['label'].value_counts().to_dict())
