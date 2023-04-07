import csv
import numpy as np
import pandas as pd
from scipy import sparse
np.random.seed(123)
print('start')

filepath = '/n/data1/hsph/biostat/celehs/lab/jl762/haoxue/'
filepath="D:\\Tianxi\\partners\\biobank\\rolled data\\result_count_parent_code_9_1\\"
# filepath="/n/data1/hsph/biostat/celehs/lab/junwen/assymWordEmbed/co500_va/"
#cooccurence = pd.read_csv("/n/data1/hsph/biostat/celehs/lab/SHARE/COOCCURRENCE_PMI_EMBEDDING/va/from_vidul/from_chuan/White_NonHispanic-4749_asym_agg/codified/part-00000-191fccd8-7e08-4373-a0a8-0ba28cd23832-c000.csv")
filename_cooccurence="biobank_asym_coocur2_win_15_codes_noLab_CUI_thold_30_removed.csv"
savename='PPMI_biobank_asym_coocur2_win_15_codes_noLab_CUI_thold_30_removed.npz'
indx2tok50_name="indx2tok_biobank_asym_coocur2_win_15_codes_noLab_CUI_thold_30_removed.csv"
tok2indx50_name="tok2indx_biobank_asym_coocur2_win_15_codes_noLab_CUI_thold_30_removed.csv"
PPMI_mode="not T"
print("----PPMI_mode: ",PPMI_mode)
cooccurence = pd.read_csv(filepath+filename_cooccurence).iloc[:,1:4]

print ("filename_cooccurence: ",filename_cooccurence)

if PPMI_mode=='T':
    cooccurence.columns = ['code2', 'code1', 'count']
else:
    cooccurence.columns = ['code1', 'code2', 'count']

cooccurence['count'] = cooccurence['count'].astype(int)
cooccurence['code2'] = cooccurence['code2'].astype(str)
cooccurence['code1'] = cooccurence['code1'].astype(str)

n_pairs = cooccurence.shape[0]
print('finish reading %s pairs'%n_pairs)
num_skipgrams = cooccurence['count'].sum()
list_code1=list(cooccurence['code1'])
print ("len(list_code1) 111: ",len(list_code1))
list_code2=list(cooccurence['code2'])
list_code1.extend(list_code1)
print ("len(list_code1) 222: ",len(list_code1))
vocab = list({}.fromkeys(list_code1).keys())
v = len(vocab)
print ("vocab [0:300]: ",vocab[0:300])
print('%s words in total'%v)
indx2tok = {}
tok2indx = {}

index_token=pd.read_csv(filepath+indx2tok50_name,header=None)
index_token.columns = ['_c0','_c1']
co_mat = np.zeros((v,v))
for k in range(index_token.shape[0]):
    indx2tok[int(index_token._c0[k])] = str(index_token._c1[k])
    tok2indx[str(index_token._c1[k])] = int(index_token._c0[k])

# for indx, tok in enumerate(vocab):
#    indx2tok[indx] = str(tok)
#    tok2indx[str(tok)] = indx

# a_file = open(filepath+indx2tok50_name, "w")
# writer = csv.writer(a_file)
# for key, value in indx2tok.items():
#     writer.writerow([key, value])
# a_file.close()
#
# a_file = open(filepath+tok2indx50_name, "w")
# writer = csv.writer(a_file)
# for key, value in tok2indx.items():
#     writer.writerow([key, value])
# a_file.close()

# for creating sparce matrices
row_indxs = []
col_indxs = []

pmi_dat_values = []  # pointwise mutual information
ppmi_dat_values = []  # positive pointwise mutial information
spmi_dat_values = []  # smoothed pointwise mutual information
sppmi_dat_values = []  # smoothed positive pointwise mutual information

# reusable quantities

# sum_over_rows[ii] = sum_over_words[ii] = wwcnt_mat.getcol(ii).sum()
sum_over_words = cooccurence.groupby('code2').sum()['count']
# sum_over_cols[ii] = sum_over_contexts[ii] = wwcnt_mat.getrow(ii).sum()
sum_over_contexts = cooccurence.groupby('code1').sum()['count']
# sum_over_words=sum_over_words/100
# sum_over_contexts=sum_over_contexts/100
# smoothing
alpha = 0.75
sum_over_words_alpha = sum_over_words ** alpha
nca_denom = np.sum(sum_over_words_alpha)

print ("num_skipgrams: ",num_skipgrams)
print ("nca_denom: ",nca_denom)
# num_skipgrams=num_skipgrams/100


for ii,row in cooccurence.iterrows():
    tok_word = str(row['code1'])
    tok_context = str(row['code2'])
    sg_count = int(row['count'])
    if ii % int(n_pairs/10) == 0:
        #print(f'finished {ii / n_pairs:.2%} of skipgrams')
        print ("finished: ",ii / n_pairs," of skipgrams")

    # here we have the following correspondance with Levy, Goldberg, Dagan
    # ========================================================================
    #   num_skipgrams = |D|
    #   nwc = sg_count = #(w,c)
    #   Pwc = nwc / num_skipgrams = #(w,c) / |D|
    #   nw = sum_over_cols[tok_word]    = sum_over_contexts[tok_word] = #(w)
    #   Pw = nw / num_skipgrams = #(w) / |D|
    #   nc = sum_over_rows[tok_context] = sum_over_words[tok_context] = #(c)
    #   Pc = nc / num_skipgrams = #(c) / |D|
    #
    #   nca = sum_over_rows[tok_context]^alpha = sum_over_words[tok_context]^alpha = #(c)^alpha
    #   nca_denom = sum_{tok_content}( sum_over_words[tok_content]^alpha )

    nwc = sg_count+1
    Pwc = max(nwc / num_skipgrams,0)
    nw = sum_over_contexts[tok_word]+1
    Pw = max(nw / num_skipgrams,0)
    nc = sum_over_words[tok_context]+1
    Pc = max(nc / num_skipgrams,0)  #1e-20
    nca = sum_over_words_alpha[tok_context]+1
    Pca = max(nca / nca_denom,0)
    # note
    # pmi = log {#(w,c) |D| / [#(w) #(c)]}
    #     = log {nwc * num_skipgrams / [nw nc]}
    #     = log {P(w,c) / [P(w) P(c)]}
    #     = log {Pwc / [Pw Pc]}
    pmi = np.log2(Pwc / (Pw * Pc)+1e-10)
    ppmi = max(pmi, 0)
    spmi = np.log2(Pwc / (Pw * Pca)+1e-10)
    sppmi = max(spmi, 0)

    row_indxs.append(tok2indx[tok_word])
    col_indxs.append(tok2indx[tok_context])
    pmi_dat_values.append(pmi)
    ppmi_dat_values.append(ppmi)
    spmi_dat_values.append(spmi)
    sppmi_dat_values.append(sppmi)

pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)), shape=(v, v))
ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)), shape=(v, v))
spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)), shape=(v, v))
sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)), shape=(v, v))
sparse.save_npz(filepath+savename, ppmi_mat)
print('construct ppmi matrix, done and saved')
