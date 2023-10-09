import pickle

# read
f_save = open('./history/dict_file_1687015893.6667292.pkl','rb')
data_dict = pickle.load(f_save)
for k in data_dict:
    print('\n### Dataset: ', k)
    for kk in data_dict[k]:
        print(kk, data_dict[k][kk][0])
# total_curv_dict = {}
# for k in data_dict:
#     curv_dict_list = data_dict[k]['hyper_dict_list']
#     auc_list = data_dict[k]['avg_auc_array']
#     for i, curv_dict in enumerate(curv_dict_list):
#         if curv_dict['curv_mode'] not in total_curv_dict: total_curv_dict[curv_dict['curv_mode']] = []
#         total_curv_dict[curv_dict['curv_mode']].append('{}_{:.5}'.format(k, auc_list[i]))
# for k in total_curv_dict: 
#     print(k)
#     print(total_curv_dict[k])



    