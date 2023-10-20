def majority_vote(final_dff):
    maj_answers = []
    headlines = []
    #properties = []
    video_urls = []
    worker_ids = []
    dict_rows = []
    for row in chunks(final_dff,3):
        dict_row = row['label'].value_counts().to_dict()
        #print(dict_row)
        majority_answer = max(dict_row,key=dict_row.get)
        maj_answers.append(majority_answer)
        video_urls.append(row['Video_urls'].iloc[0])
        headlines.append(row['Headlines'].iloc[0])
        #properties.append(row['Properties'].iloc[0])
        worker_ids.append(row['workerid'].tolist())
        dict_rows.append(dict_row)
    dff_1 = pd.concat([pd.Series(worker_ids),pd.DataFrame(video_urls),pd.DataFrame(headlines),pd.Series(dict_rows),pd.DataFrame(maj_answers)],axis=1)
    dff_1.columns = ['worker_ids','video_urls','headline','labels','majority_answer']
    #dff_1.to_csv('for_alpha.csv')
    leading_labels = dff_1[dff_1['majority_answer']=='leading']
    misleading_labels = dff_1[dff_1['majority_answer']=='misleading']
    result = dff_1['majority_answer'].value_counts().to_dict()
    #print('majority_vote_result: ', dff_1['majority_answer'].value_counts().to_dict())
    return dff_1, result, leading_labels, misleading_labels
