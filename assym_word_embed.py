from collections import Counter
import csv
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import sys
import pickle
np.random.seed(123)
print('start')
filepath = '/n/data1/hsph/biostat/celehs/lab/junwen/assymWordEmbed/biobank/phecode_drugs_indi_side_CUIs/'
#PPMI_file_T='T_PPMI_biobank_asym_coocur2_win_15_codes_noLab_CUI_thold_30_removed.npz'
prefix = filepath
embed_dim = 300
epsilon = 1e-3
mu = 0.97
method = 'agd'
num_updates=150000
dirr_save='/n/data1/hsph/biostat/celehs/lab/junwen/assymWordEmbed/biobank/phecode_drugs_indi_side_CUIs/embedding/'

k_negative=5
PPMI_file='RM_self_PPMI_Drug_disease30_NOunique'+"K_negative_"+str(k_negative)+'.npz'

index_2_token='RM_self_indx2tok_Drug_disease30_NOunique'+'K_negative_'+str(k_negative)+'.csv'
token_2_index='RM_self_tok2indx_Drug_disease30_NOunique'+'K_negative_'+str(k_negative)+'.csv'
comments_savename="_dim"+str(embed_dim)+'_RM_self_K_negative_'+str(k_negative)+"_win30_Drug_disease30_unique_"
differ_flag=False
# if differ_flag==True:
#     comments_savename="_dim300_win15_codes_noLab_CUI_thold_30_removed_NOdiffer_"
# else:
#     comments_savename = "_dim300_win15_codes_noLab_CUI_thold_30_removed_differ_"
#

if not os.path.exists(dirr_save):
    os.mkdir(dirr_save)
print ("-------------PPMI_file: ",PPMI_file)
#print ("-------------PPMI_file_T: ",PPMI_file_T)
print ("--------------comments_savename: ",comments_savename)
print('start reading dictionaries')

df=pd.read_csv(filepath+index_2_token,header=None)
df.columns=["index","codes"]
tokens_all=list(df["codes"])
reader = csv.reader(open(filepath+index_2_token, 'r'))
indx2tok = {}
for row in reader:
    indx, tok = row
    indx2tok[int(indx)] = tok
reader = csv.reader(open(filepath+token_2_index, 'r'))
tok2indx = {}
for row in reader:
    tok, indx = row
    tok2indx[tok] = int(indx)

print('done')
def ww_sim(word, mat, topn=10, medcode=False):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    if medcode:
        for sindx in sindxs[0:topn]:
            try:
                print("code: %s"%indx2tok[sindx].upper())
                #print("descrption: %s"%code2english[indx2tok[sindx].upper()])
            except:
                continue
    return sim_word_scores

def update_W(W, W_old, V_old, C1, C2, P1, P2, parms, method):
    """Update right factor for matrix completion objective."""
    v = W.shape[0]
    C1T = C1.T
    C2T = C2.T
    mu = parms['mu']
    epsilon = parms['epsilon']
    if method == 'agd':
        W_pred = W + mu*V_old
        gradf = 1/v*(W_pred@(C1@C1T)-P1@C1T + W_pred@(C2@C2T)-P2@C2T)#1/v*(W_pred@(C1@C1T)-P1@C1T + W_pred@(C2@C2T)-P2@C2T) # W - eta*(W@C1@C1T-P1@C1T + W@C2@C2T-P2@C2T)
        V_new = mu * V_old - epsilon * gradf
        W_new = W + V_new
        # Y = W-(1/beta)*gradf
        # Z = W_old - eta*gradf
        # W_new = tau*Z + (1-tau)*Y
    if method == 'heavyball':
        gradf = 1/v*(W @ (C1 @ C1T) - P1 @ C1T + W @ (C2 @ C2T) - P2 @ C2T)
        W_new = W - epsilon * gradf + mu * (W - W_old)
        V_new = 0
    gradf_norm = np.linalg.norm(gradf, 'fro')
    # print("norm of grad for f is %s" % gradf_norm)
    return W_new, W, V_new, gradf_norm

def update_C(C, C_old, V_old, W, P, parms, method):
    v = W.shape[0]
    WT = W.T
    mu = parms['mu']
    epsilon = parms['epsilon']
    if method == 'agd':
        C_pred = C + mu* V_old
        gradf = 1/v*(-WT@P+(WT@W)@C_pred)#1/v*(-WT@P+(WT@W)@C_pred)
        V_new = mu*V_old - epsilon*gradf
        C_new = C+V_new
        # Y = C - (1/beta)*gradf
        # Z = C_old - eta*gradf
        # C_new = tau*Z + (1-tau)*Y
    if method == 'heavyball':
        mu = parms['mu']
        epsilon = parms['epsilon']
        gradf = 1/v*(-WT @ P + (WT @ W) @ C) - mu * (C - C_old)
        C_new = C - epsilon * gradf
        V_new = 0
    gradf_norm = np.linalg.norm(gradf, 'fro')
    # print("norm of grad for f is %s" % gradf_norm)
    return C_new, C, V_new, gradf_norm

def altmin(P1, P2, parms, method, tol=1e-5,compute_mse=True, early_stop=True):
    num_updates = parms['num_updates']
    d = parms['embed_dim']
    v = P1.shape[0]
    if False: #os.path.exists(prefix + affix + method + str(2000) + '.pkl'):
        with open(prefix + affix + method + str(2000) + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            pre_res = pickle.load(f)
            W = pre_res['W']
            C1 = pre_res['C1']
            C2 = pre_res['C2']
            print("starting from previous checkpoint")
    else:
        W = np.random.normal(0, 1, (v, d))
        C1 = np.random.normal(0, 1, (d, v))
        C2 = np.random.normal(0, 1, (d, v))
    # W = np.random.normal(0, 1, (v, d))
    # C1 = np.random.normal(0, 1, (d, v))
    # C2 = np.random.normal(0, 1, (d, v))
    # u1, s1, C1 = sparse.linalg.svds(P1,d)
    # u2, s2, C2 = sparse.linalg.svds(P2,d)
    # smat1 = np.diag(s1)
    # smat2 = np.diag(s2)
    # W = 1/2*(np.dot(u1, smat1)+np.dot(u2, smat2))
    i = 1
    delta = [1]
    error = [1]
    error1 = [1]
    error2 = [1]
    gradf_Ws_norm = []
    gradf_C1s_norm = []
    gradf_C2s_norm = []
    W_old = W.copy()
    C1_old = C1.copy()
    C2_old = C2.copy()
    WV = np.zeros(W.shape)
    C1V = np.zeros(C1.shape)
    C2V = np.zeros(C2.shape)
    rnd_indx = np.random.choice(v, 6000)
    while i<=num_updates and error[-1]>tol and delta[-1]>tol:
        #parms['mu'] = min(1 - 3 / (i + 5), 0.9)
        print('iteration: %s'%i)
        W, W_old, WV, gradf_W_norm = update_W(W, W_old, WV, C1, C2, P1, P2, parms=parms, method=method)
        C1, C1_old, C1V, gradf_C1_norm = update_C(C1, C1_old, C1V, W, P1, parms=parms, method=method)
        C2, C2_old, C2V, gradf_C2_norm = update_C(C2, C2_old, C2V, W, P2, parms=parms, method=method)
        #error0 = np.mean([np.mean(np.square(W_old-W)),np.mean(np.square(C1_old-C1)),np.mean(np.square(C2_old-C2))])
        #error.append(error0)
        if i % 1000 == 0:
            print(W@(C1@C1.T)-P1@C1.T + W@(C2@C2.T)-P2@C2.T)
        if (i % 2000 ==0 or i==2)  and i>1:
            results = {'W': W, 'C1': C1, 'C2': C2, 'error': error[1:],
                       'error1': error1[1:], 'error2': error2[1:],
                       'delta': delta[1:], 'gradf_Ws_norm': gradf_Ws_norm,
                       'gradf_C1s_norm': gradf_C1s_norm, 'gradf_C2s_norm': gradf_C2s_norm}
            with open(dirr_save + affix + method + str(i) + comments_savename+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(results, f)

            W_value = results['W']
            print("----W_value", np.array(W_value).shape)
            C1_value = results['C1']
            print("----C1_value", np.array(C1_value).shape)
            C2_value = results['C2']
            print("----C2_value", np.array(C2_value).shape)

            df = pd.DataFrame(W_value.T, columns=tokens_all)
            df.to_csv(dirr_save+"W_"+comments_savename+".csv",index=False)
            df = pd.DataFrame(C1_value, columns=tokens_all)
            df.to_csv(dirr_save + "C1_" + comments_savename + ".csv", index=False)
            df = pd.DataFrame(C2_value, columns=tokens_all)
            df.to_csv(dirr_save + "C2_" + comments_savename + ".csv", index=False)



        i += 1
        if compute_mse:
            if v < 9000:
                P1hat = W @ C1
                P2hat = W @ C2
                mse1 = np.linalg.norm(P1hat - P1, 'fro')#mean_squared_error(P1hat, P1)
                mse2 = np.linalg.norm(P2hat - P2, 'fro')#mean_squared_error(P1hat, P1)
            else:
                P1hat = W[rnd_indx, :] @ C1
                P2hat = W[rnd_indx, :] @ C2
                mse1 = np.linalg.norm(P1hat - P1[rnd_indx, :], 'fro')#mean_squared_error(P1hat, P1[rnd_indx, :])
                mse2 = np.linalg.norm(P2hat - P2[rnd_indx, :], 'fro')#mean_squared_error(P2hat, P2[rnd_indx, :])
            mse = np.sqrt(mse1**2 + mse2**2)
            error1.append(mse1)
            error2.append(mse2)
            error.append(mse)
            gradf_Ws_norm.append(gradf_W_norm)
            gradf_C1s_norm.append(gradf_C1_norm)
            gradf_C2s_norm.append(gradf_C2_norm)
            # print('5000 submatrix error %s' % mse)
        if early_stop:
            delta0 = np.linalg.norm(W - W_old, 'fro') / np.linalg.norm(W_old, 'fro') \
                     + np.linalg.norm(C2 - C2_old, 'fro') / np.linalg.norm(C2, 'fro') \
                     + np.linalg.norm(C1 - C1_old, 'fro') / np.linalg.norm(C1, 'fro')
            delta.append(delta0)
            # print('parameter change %s' % delta0)
    results={'W':W, 'C1':C1, 'C2':C2, 'error':error[1:],
             'error1': error1[1:], 'error2': error2[1:],
             'delta': delta[1:], 'gradf_Ws_norm':gradf_Ws_norm,
             'gradf_C1s_norm':gradf_C1s_norm, 'gradf_C2s_norm':gradf_C2s_norm}
    # print('left mse %s' % mse1)
    # print('right mse %s' % mse2)
    return (results)

print('start loading ppmi')
ppmi_mat = sparse.load_npz(filepath+PPMI_file)
# if os.path.exists(filepath+PPMI_file_T):
#     ppmi_mat_T = sparse.load_npz(filepath + PPMI_file_T)
# else:
#     ppmi_mat_T=ppmi_mat.T
v = ppmi_mat.shape[0]
# rnd_id = np.random.choice(v, 1000)
# ixgrid = np.ix_(rnd_id, rnd_id)
# subindx = []
# for tok, indx in tok2indx.items():
#     words =s tok.split(':')
#     if len(words) > 1:
#         subindx.append(indx)
#
# ixgrid = np.ix_(subindx, subindx)
# ppmi_submat = ppmi_mat[ixgrid]
print('finish loading')
# if len(sys.argv) > 1:
#     embed_dim = int(sys.argv[1])
#     epsilon = float(sys.argv[2])
#     mu = float(sys.argv[3])
#     method = str(sys.argv[4])
# else:
#     embed_dim = 300
#     epsilon = 1e-3
#     mu = 0.95
#     method = 'agd'

parms = {'embed_dim': embed_dim, 'epsilon':epsilon, 'mu':mu, 'method':method, 'num_updates':num_updates}
affix = "%sepsilon%smu%siter%s"%(embed_dim,epsilon,mu,num_updates)

#"/home/jl762/haoxue/tuning"
filepath = prefix + affix + method + '.pkl'
print("start agd")
print("-----parameters-----")
for parm,value in parms.items():
    print("%s: %s"%(parm, value))

if differ_flag==True:
    ppmi_mat=ppmi_mat.toarray()
    ppmi_mat=np.maximum(ppmi_mat-ppmi_mat.T,0)

results = altmin(ppmi_mat, ppmi_mat.T, parms, method,
                 tol=1e-10, compute_mse=True, early_stop=False)
results['parms'] = parms
# u1, s1, vh1 = sparse.linalg.svds(ppmi_mat,embed_dim)
# u2, s2, vh2 = sparse.linalg.svds(ppmi_mat.T,embed_dim)
# smat1 = np.diag(s1)
# smat2 = np.diag(s2)
# Xhat_svd = np.dot(u1, np.dot(smat1[:,:embed_dim], vh1[:embed_dim,:]))
# Yhat_svd = np.dot(u2, np.dot(smat2[:,:embed_dim], vh2[:embed_dim,:]))
# error_svd = np.sqrt(np.linalg.norm(Xhat_svd-ppmi_mat, 'fro')**2+np.linalg.norm(Yhat_svd-ppmi_mat.T, 'fro')**2)
# print('SVD error is%s'%error_svd)
# W, C1, C2, i, error_agd, error_agd1, error_agd2, _ = altmin_agd(ppmi_mat, ppmi_mat.T, d=embed_dim, num_updates=500, tol=10e-5,
#                                                     compute_mse=True, early_stop=True,  beta=50000, eta=0.000001, tau=0.001)
# Saving the objects:
with open(filepath, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(results, f)


# Getting back the objects:
# with open(prefix + affix + method + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     results = pickle.load(f)
# for item in ['error','gradf_Ws_norm', 'gradf_C1s_norm', 'gradf_C2s_norm']:
#     plt.figure(figsize=(20,10))
#     plt.plot(np.log(results[item]),alpha=1,label="agd,epsilon:{epsilon}".format(epsilon=results['parms']['epsilon']))
#     plt.ylabel('log forb norm')
#     plt.xlabel('iteration')
#     plt.title(item)
#     plt.legend()
#     plt.savefig(prefix + affix + method + '_' + item + '.png')
#     plt.show()
#
#
#
# with open('/n/data1/hsph/biostat/celehs/lab/jl762/haoxue/newPPMI1000_300epsilon0.001mu0.1iter1000agd.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     results = pickle.load(f)
# for item in ['error','gradf_Ws_norm', 'gradf_C1s_norm', 'gradf_C2s_norm']:
#     plt.figure(figsize=(20,10))
#     plt.plot(np.log(results[item]),alpha=1)
#     plt.ylabel('log forb norm')
#     plt.xlabel('iteration')
#     plt.title(item)
#     plt.show()
#
# for epsilon in [0.01,0.001,0.0001]:
#     for mu in [0.1,0.5,0.9,0.99]:
#         path = '/Users/xuehao/Dropbox/harvard/mlt/Hao_Updates/hao_code/code/wenjun/300epsilon'+str(epsilon)+'mu'+str(mu)+'iter1000agd.pkl'
#         if os.path.exists(path):
#             with open(path,'rb') as f:  # Python 3: open(..., 'rb')
#                 results = pickle.load(f)
#                 print('epsilon'+str(epsilon)+'mu'+str(mu))
#                 print(results['error'][-1])
#                 print('---------')
#
# prefix = '/Users/xuehao/Dropbox/harvard/mlt/Hao_Updates/hao_code/code/wenjun/'
# np.savetxt(prefix+'W_300_agd.txt', results['W'])
# np.savetxt(prefix+'C1_300_agd.txt', results['C1'])
# np.savetxt(prefix+'C2_300_agd.txt', results['C2'])
#
#
# W = results['W']
# v = W.shape[0]
# WT = W.T
# C1 = results['C1']
# C2 = results['C2']
# C1T=C1.T
# C2T=C2.T
# P1 = ppmi_mat
# P2 = ppmi_mat.T
# print(1/v*(-WT @ P1 + (WT @ W) @ C1))
# print(1/v*(-WT @ P2 + (WT @ W) @ C2))
# print(1/v*(W @ (C1 @ C1T) - P1 @ C1T + W @ (C2 @ C2T) - P2 @ C2T))
