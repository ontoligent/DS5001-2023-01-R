import pandas as pd
import numpy as np
import nltk

class TextParser():
    """
    A class to parse a single Gutenberg-type text files into a TOKENS dataframe with
    an OHCO index. Also has methods to extract a VOCAB table, although vocabulary
    tables out to be generated at the corpus level.
    
    Sample parameter values:

    ohco_pats = [
        ('chapter', r"^\s*(chapter|letter)\s+(\d+)", 'm')    
    ]

    clip_pats = [
        r'START OF GUTENBERG PROJECT', 
        r'^\s*THE END'
    ]

    """

    # TODO: Make these private
    src_imported:bool = False       
    src_clipped:bool = False
    src_col_suffix:str ='_str'

    join_pat:str = r'\n'
    strip_hyphens:bool = False
    strip_whitespace:bool = False
    verbose:bool = False

    stanford_pos_model:str = "english-bidirectional-distsim.tagger"
    stanford_pos_model_path = None
        
    # We assume all OHCOs have sentences and tokens
    # and that there are terminal in the list.
    ohco_pats:[] = [
        ('para', r"\n\n", 'd'),
        ('sent', r"[.?!;:]+", 'd'),
        ('token', r"[\s',-]+", 'd')
    ]
        
    _ohco_type:{} = {
        'd': '_num',
        'm': '_id'
    }
        
    def __init__(self, src_file:str, ohco_pats:[], clip_pats:[], use_nltk=True):
        """Initialize the object and extract config info. If using NLTK, download resources."""
        self.src_file = src_file            
        self.clip_pats = clip_pats # TODO: Validate
        self.ohco_pats = ohco_pats + self.ohco_pats # TODO: Validate
        self.OHCO = [item[0]+self._ohco_type[item[2]] for item in self.ohco_pats]
        self.ohco_names = [item[0] for item in self.ohco_pats]
        self.use_nltk = use_nltk

        if self.use_nltk:
            # Override the last two OHCO items
            self.ohco_pats[-2] = ('sent', None, 'nltk')
            self.ohco_pats[-1] = ('token', None, 'nltk')
            # Make sure you have the NLTK stuff
            for package in [
                'tokenizers/punkt', 
                'taggers/averaged_perceptron_tagger', 
                'corpora/stopwords', 
                'help/tagsets'
            ]:
                if self.verbose: print("Checking", package)
                try:
                    nltk.data.find(package)
                except IndexError:
                    nltk.download(package)
            
    def import_source(self, strip:bool=True, char_encoding:str="utf-8-sig"):
        """Convert a raw text file into a dataframe of lines."""
        if self.verbose: print("Importing ", self.src_file)
        text_lines = open(self.src_file,'r', encoding=char_encoding).readlines()
        self.LINES = pd.DataFrame({'line_str':text_lines})
        self.LINES.index.name = 'line_id'
        if strip:
            self.LINES.line_str = self.LINES.line_str.str.strip()
        self.src_imported = True
        if self.verbose: print("Clipping text")
        self._clip_lines()
        return self        

    def _clip_lines(self):
        """Remove cruft lines from beginning and/or end of file."""
        start_pat = self.clip_pats[0]
        end_pat = self.clip_pats[1]
        start = self.LINES.line_str.str.contains(start_pat, regex=True)
        end = self.LINES.line_str.str.contains(end_pat, regex=True)
        try:
            start_line_num = self.LINES.loc[start].index[0]
        except IndexError:
            raise ValueError("Clip start pattern not found.")            
        try:
            end_line_num = self.LINES.loc[end].index[0]
        except IndexError:
            raise ValueError("Clip end pattern not found.")
        self.LINES = self.LINES.loc[start_line_num + 1 : end_line_num - 2]
        self.src_clipped == True
        
    def parse_tokens(self):
        """Convert lines to tokens based on OHCO."""
        if self.src_imported:

            # Start with the LINES df
            self.TOKENS = self.LINES.copy()

            # Walk through each level of the OHCO to build out TOKENS
            for i, level in enumerate(self.OHCO):

                if self.verbose: print(f"Parsing OHCO level {i} {level}", end=' ')

                # Define level-specific variables
                parse_type = self.ohco_pats[i][2]
                div_name = self.ohco_pats[i][0]
                div_pat = self.ohco_pats[i][1]
                if i == 0:
                    src_div_name = 'line'
                else:
                    src_div_name = self.ohco_names[i - 1] 
                src_col = f"{src_div_name}{self.src_col_suffix}"
                dst_col = f"{div_name}{self.src_col_suffix}"

                # By Milestone
                if parse_type == 'm':
                    if self.verbose: print(f"by milestone {div_pat}")
                    div_lines = self.TOKENS[src_col].str.contains(div_pat, regex=True, case=True) # TODO: Parametize case
                    self.TOKENS.loc[div_lines, div_name] = [i+1 for i in range(self.TOKENS.loc[div_lines].shape[0])]
                    self.TOKENS[div_name] = self.TOKENS[div_name].ffill()
                    self.TOKENS = self.TOKENS.loc[~self.TOKENS[div_name].isna()] 
                    self.TOKENS = self.TOKENS.loc[~div_lines] 
                    self.TOKENS[div_name] = self.TOKENS[div_name].astype('int')
                    self.TOKENS = self.TOKENS.groupby(self.ohco_names[:i+1], group_keys=True)[src_col]\
                        .apply(lambda x: '\n'.join(x)).to_frame(dst_col)

                    # print(self.TOKENS[dst_col].str.count(r'\n\n'))
                    print(src_col, dst_col)
                    print(self.TOKENS.columns)


                # By Delimitter
                elif parse_type == 'd':
                    if self.verbose: print(f"by delimitter {div_pat}")
                    self.TOKENS = self.TOKENS[src_col].str.split(div_pat, expand=True).stack().to_frame(dst_col)
                
                # By NLTK 
                elif parse_type == 'nltk':
                    if self.verbose: print(f"by NLTK model")

                    if level == 'sent_num':
                        self.TOKENS = self.TOKENS.para_str\
                                .apply(lambda x: pd.Series(nltk.sent_tokenize(x), dtype='string'))\
                                .stack()\
                                .to_frame('sent_str')
                    
                    if level == 'token_num':
                        if self.strip_hyphens == True:
                            self.TOKENS.sent_str = self.TOKENS.sent_str.str.replace(r"-", ' ')
                        if self.strip_whitespace == True:
                            self.TOKENS = self.TOKENS.sent_str\
                                    .apply(lambda x: pd.Series(
                                            nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)),
                                            dtype='object'
                                        )
                                    )
                        else:
                            self.TOKENS = self.TOKENS.sent_str\
                                    .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))
                        self.TOKENS = self.TOKENS.stack().to_frame('pos_tuple')
                        self.TOKENS['pos'] = self.TOKENS.pos_tuple.apply(lambda x: x[1])
                        self.TOKENS['token_str'] = self.TOKENS.pos_tuple.apply(lambda x: x[0])
                        self.TOKENS['term_str'] = self.TOKENS.token_str.str.lower()   
        
                else:
                    raise ValueError(f"Invalid parse option: {parse_type}.")

                # After creating the current OHCO level
                self.TOKENS.index.names = self.OHCO[:i+1]

            # After iterating through the OHCO

            # Not sure if needed anymore
            # self.TOKENS[dst_col] = self.TOKENS[dst_col].str.strip()
            # self.TOKENS[dst_col] = self.TOKENS[dst_col].str.replace(self.join_pat, ' ', regex=True)
            # self.TOKENS = self.TOKENS[~self.TOKENS[dst_col].str.contains(r'^\s*$', regex=True)]

            if not self.use_nltk:
                self.TOKENS['term_str'] = self.TOKENS.token_str.str.replace(r'[\W_]+', '', regex=True).str.lower()  
            else:
                punc_pos = ['$', "''", '(', ')', ',', '--', '.', ':', '``']
                self.TOKENS['term_str'] = self.TOKENS[~self.TOKENS.pos.isin(punc_pos)].token_str\
                    .str.replace(r'[\W_]+', '', regex=True).str.lower()  
            
        else:
            raise RuntimeError("Source not imported. Please run .import_source()")

    def extract_vocab(self):
        """This should also be done at the corpus level."""
        self.VOCAB = self.TOKENS.term_str.value_counts().to_frame('n')
        self.VOCAB.index.name = 'term_str'
        self.VOCAB['n_chars'] = self.VOCAB.index.str.len()
        self.VOCAB['p'] = self.VOCAB['n'] / self.VOCAB['n'].sum()
        self.VOCAB['s'] = 1 / self.VOCAB['p']
        self.VOCAB['i'] = np.log2(self.VOCAB['s']) # Same as negative log probability (i.e. log likelihood)
        self.VOCAB['h'] = self.VOCAB['p'] * self.VOCAB['i']
        self.H = self.VOCAB['h'].sum()
        return self

    def annotate_vocab(self):
        """This should be done at the corpus level."""
        # Stopwords
        # Max POS
        # POS variability
        # Porter Stems
        pass

    def extract_pos_data(self):
        # TODO: Create dataframe for POS info, including Penn Treebank info
        pass

    def extract_named_entities(self):
        # TODO: Create dataframe of named entities
        pass

    def gather_tokens(self, level=0, grouping_col='term_str', cat_sep=' '):
        """Gather tokens into strings for arbitrary OHCO level."""
        max_level = len(self.OHCO) - 2 # Can't gather tokens at the token level :)
        if level > max_level:
            raise ValueError(f"Level {level} too high. Try between 0 and {max_level}")
        else:
            level_name = self.OHCO[level].split('_')[0]
            idx = self.TOKENS.index.names[:level+1]
            return self.TOKENS.groupby(idx)[grouping_col].apply(lambda x: x.str.cat(sep=cat_sep))\
                .to_frame(f'{level_name}_str')


if __name__ == '__main__':
    pass