import pandas as pd
import numpy as np

class NgramLanguageModel():
    
    k = .5
    
    def __init__(self, TOKENS, n=3):
        self.TOKENS = TOKENS
        self.n = n
        self.OHCO = list(TOKENS.index.names)
        self.widx = [f'w{i}' for i in range(self.n)]
        
    def _add_sentence_markers(self):
        self.S = self.TOKENS.groupby(self.OHCO[:-2]).term_str\
            .apply(lambda x: '<s> ' + ' '.join(x) + ' </s>')\
            .to_frame('sent_str')\
            .reset_index(drop=True)\
            .sent_str.apply(lambda x: pd.Series(x.split()))\
            .stack()\
            .to_frame('w0')
        self.S.index.names = OHCO[-2:]
        
    def generate_main_index(self):
        self._add_sentence_markers()
#         self.I = self.S.rename(columns={'term_str':'w0'})
        self.I = self.S
        for i in range(1, self.n): 
            self.I[f'w{i}'] = self.I.w0.shift(-i)
        
    def get_all_indexes(self):
        self.NG = []
        for i in range(self.n+1):
            self.NG.append(self.I.iloc[:,:i+1])
            
    def get_value_counts(self):
        self.LM = []
        for i in range(self.n):
            self.LM.append(self.NG[i].value_counts().to_frame('n'))
            self.LM[i]['p'] = self.LM[i].n / self.LM[i].n.sum()
            self.LM[i]['log_p'] = np.log2(self.LM[i].p)
        # Hack to remove single value tuple ...
        self.LM[0].index = [i[0] for i in self.LM[0].index] 
        
    def apply_smoothing(self):
        self.B = [len(self.LM[0])**i for i in range(self.n)]
        self.Z = [None for _ in range(self.n)]
        for i in range(1, self.n):
            self.LM[i]['cpl'] = (self.LM[i].n + self.k) / (self.LM[i-1].n.sum() + self.B[i] * self.k)
            self.LM[i]['log_cpl'] = np.log2(self.LM[i].cpl)        
            self.Z[i] = np.log2(1/self.B[i]/2)
            self.LM[i].sort_index(inplace=True)
