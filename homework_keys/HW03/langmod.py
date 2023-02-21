import pandas as pd
import numpy as np

class NgramCounter():
    """A class to generate tables of ngram tokens and types from a list of sentences."""
    
    unk_sign = '<UNK>'
    sent_pad_signs = ['<s>','</s>']
        
    def __init__(self, sents:[], vocab:[], n:int=3):
        self.sents = sents # Expected to be normalized
        self.vocab = vocab # Can be extracted from another corpus
        self.n = n
        self.widx = [f'w{i}' for i in range(self.n)] # Used for cols and index names
        
    def generate(self):
        
        # Convert sentence list to dataframe
        self.S = pd.DataFrame(dict(sent_str=self.sents))
            
        # Pad sentences 
        pad = (self.sent_pad_signs[0] + ' ') *  (self.n - 1)
        self.I = (pad + self.S.sent_str + ' ' + self.sent_pad_signs[1])\
            .str.split(expand=True).stack().to_frame('w0')
        
        # Set index names
        self.I.index.names = ['sent_num', 'token_num']
        
        # Remove OOV terms
        self.I.loc[~self.I.w0.isin(self.vocab + self.sent_pad_signs), 'w0'] = self.unk_sign

        # Get sentence lengths (these will include pads)
        self.S['len'] = self.I.groupby('sent_num').w0.count()
                
        # Add w columns
        for i in range(1, self.n):
            self.I[f'w{i}'] = self.I[f"w{i-1}"].shift(-1)         
        
        # Generate ngrams
        self.NG = []
        for i in range(self.n):
            self.NG.append(self.I.iloc[:, :i+1].copy())

        # Remove spurious rows
        self.NG[i] = self.NG[i].dropna()
                                
        # Generate raw ngram counts and MLEs
        self.LM = []
        for i in range(self.n):
            self.LM.append(self.NG[i].value_counts().to_frame('n'))
            self.LM[i]['mle'] = self.LM[i].n / self.LM[i].n.sum()
            self.LM[i] = self.LM[i].sort_index()

        # Hack to remove single value tuple from unigram table ...
        self.LM[0].index = [i[0] for i in self.LM[0].index]
        self.LM[0].index.name = 'w0'
                
        
class NgramLanguageModel():
    """A class to create ngram language models."""
    
    # Set the Lidstone Smoothing value; LaPlace = 1
    k:float = .5
    
    def __init__(self, ngc:NgramCounter):
        self.S = ngc.S
        self.LM = ngc.LM
        self.NG = ngc.NG
        self.n = ngc.n
        self.widx = ngc.widx

    def apply_smoothing(self):
        """Applies simple smoothing to ngram type counts to estimate the models."""
        
        # Z1 and Z2 will hold info about unseen ngrams
        self.Z1 = [None for _ in range(self.n)] # Unseen N grams, but seen N-1 grams
        self.Z2 = [None for _ in range(self.n)] # Unseen N-1 grams too
        
        # The base vocab size (same as number of unigram types)
        V = len(self.LM[0]) # Inlcides <s> and </s>
        
        # The number of ngram types
        B = [V**(i+1) for i in range(self.n)]

        # Handle unigram case (no need for smoothing)
        self.LM[0]['p'] = self.LM[0].n / self.LM[0].n.sum()
        self.LM[0]['log_p'] = np.log2(self.LM[0].p)

        # Handle higher order ngrams
        for i in range(1, self.n):     
            
            # MLE
            self.LM[i]['mle2'] = self.LM[i].n / self.LM[i-1].n

            # Emplopy smoothing formula
            self.LM[i]['p'] = (self.LM[i].n + self.k) / (self.LM[i-1].n + B[i-1] * self.k)
            self.LM[i]['log_p'] = np.log2(self.LM[i].p)

            # Unseen N grams, but seen N-1 grams
            self.Z1[i] = np.log2(self.k / (self.LM[i-1].n + B[i-1] * self.k))

            # Unsess N-1 grams too
            self.Z2[i] = np.log2(self.k / B[i-1] * self.k)
                
            # Tidy up the index
            self.LM[i].sort_index(inplace=True)
        
    def predict(self, test:NgramCounter):
        """Predicts test sentences with estimated models."""
        self.T = test
        for i in range(self.n):
            ng = i + 1
            if i == 0:
                self.T.S[f'ng_{ng}_ll'] = self.T.NG[0]\
                    .join(self.LM[0].log_p, on=self.widx[:ng])\
                    .groupby('sent_num').log_p.sum()
            else:
                self.T.S[f'ng_{ng}_ll'] = self.T.NG[i]\
                    .join(self.LM[i].log_p, on=self.widx[:ng])\
                    .fillna(self.Z1[i]).fillna(self.Z2[i])\
                    .groupby('sent_num').log_p.sum()
                
            self.T.S[f'pp{ng}'] = 2**( -self.T.S[f'ng_{ng}_ll'] / self.T.S['len'])
            

    def generate_text(self, n_sents=20):
        """Generate texts using Shannon Game method."""
        
        LM = self.LM # For convenience
        i = self.n - 1
    
        # Start with beginning sentence marker
        words = ['<s>' for _ in range(i)]

        # Sentence counter
        sent_count = 0

        # Generate sentences until we've reached our limit
        while sent_count < n_sents:

            # Get ngram context
            ng = tuple(words[-i:])

            # Get next word
            words.append(LM[i].loc[ng].sample(weights='mle').index.values[0])

            # Terminate when end-of-sentence marker found
            if words[-1] == '</s>':
                sent_count += 1                        
                if sent_count < n_sents:
                    words.append('<s>')

        # Create text from words
        text = ' '.join(words)

        sents = pd.DataFrame(dict(sent_str=text.split('<s> <s>')))
        sents['len'] = sents.sent_str.str.len()
        sents = sents[sents.len > 0]
        sents.sent_str = sents.sent_str.str.replace('<s> ', '')
        sents.sent_str = sents.sent_str.str.replace(' </s>', '')
        sents.sent_str = sents.sent_str.str.strip()
        sents.sent_str = sents.sent_str.str.replace(r" s ", "'s ", regex=True)
        
        _ = [print(f"{str(x+1).zfill(2)}. {sent}.\n".upper()) for x, sent in enumerate(sents.sent_str)]

        self.generated_sents = sents