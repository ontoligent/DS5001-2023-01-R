import pandas as pd
import numpy as np

class TextImporter():
    """
    A text importing object designed for use with a single Gutenberg-type text files. Generates TOKENS and VOCAB dataframes.
    
    Sample parameter values:

    ohco_pats = [
        ('chapter', r"^\s*(chapter|letter)\s+(\d+)")    
    ]

    clip_pats = [r'START OF GUTENBERG PROJECT', r'^\s*THE END']

    """

    src_imported:bool = False       
    src_clipped:bool = False
    src_col_suffix:str ='_str'
    join_pat:str = r'\n'
        
    # We assume all OHCOs have sentences and tokens
    ohco_pats:[tuple] = [
        ('para', r"\n\n", 'd'),
        ('sent', r"[.?!;:]+", 'd'),
        ('token', r"[\s',-]+", 'd')
    ]
        
    _ohco_type:{} = {
        'd': '_num',
        'm': '_id'
    }
        
    def __init__(self, src_file:str, ohco_pats:[], clip_pats:[]):
        # TODO: Generalize this to work with strings, too
        self.src_file = src_file            
        self.clip_pats = clip_pats # TODO: Validate
        self.ohco_pats = ohco_pats + self.ohco_pats # TODO: Validate
        self.OHCO = [item[0]+self._ohco_type[item[2]] for item in self.ohco_pats]
        self.ohco_names = [item[0] for item in self.ohco_pats]
        
    def import_source(self, strip:bool=True, char_encoding:str="utf-8-sig"):
        """Convert a raw text file into a dataframe of lines"""
        print("Importing ", self.src_file)
        text_lines = open(self.src_file,'r', encoding=char_encoding).readlines()
        self.LINES = pd.DataFrame({'line_str':text_lines})
        self.LINES.index.name = 'line_id'
        if strip:
            self.LINES.line_str = self.LINES.line_str.str.strip()
        self.src_imported = True
        print("Clipping text")
        self._clip_lines()
        return self        

    def _clip_lines(self):
        """Remove cruft lines from beginning and/or end of file"""
        start_pat = self.clip_pats[0]
        end_pat = self.clip_pats[1]
        start = self.LINES.line_str.str.contains(start_pat, regex=True)
        end = self.LINES.line_str.str.contains(end_pat, regex=True)
        start_line_num = self.LINES.loc[start].index[0]
        end_line_num = self.LINES.loc[end].index[0]
        self.LINES = self.LINES.loc[start_line_num + 1 : end_line_num - 2]
        self.src_clipped == True
        
    def parse_tokens(self):
        """Convert lines to tokens with arbitrary OHCO"""
        if self.src_imported:
            self.TOKENS = self.LINES.copy()
            for i, level in enumerate(self.OHCO):
                print(f"Parsing OHCO level {i} {level}", end=' ')
                parse_type = self.ohco_pats[i][2]
                if parse_type == 'd':
                    self.TOKENS = self._split_by_delimitter(self.TOKENS, i)
                elif parse_type == 'm':
                    self.TOKENS = self._group_by_milestone(self.TOKENS, i)
                else:
                     print(f"Invalid parse option: {parse_type}.")
            self.TOKENS['term_str'] = self.TOKENS.token_str.str.replace(r'[\W_]+', '', regex=True).str.lower()
            return self
        else:
            raise("Source not imported. Please run .import_source()")

    def _group_by_milestone(self, df, ohco_level):
        """Group and chunk text by milestone, such as chapter headers"""
        
        # DEFINITIONS

        # The name of the div (content object level) to be created
        div_name = self.ohco_pats[ohco_level][0]

        # The milestone pattern to used to infer the div
        div_pat = self.ohco_pats[ohco_level][1]
        
        # Notify 
        print(f"by milestone {div_pat}")
        
        # The parent div (content object level)
        if ohco_level - 1 < 0:
            src_div_name = 'line' # If we are working with the raw table of lines
        else:
            src_div_name = self.ohco_names[ohco_level - 1] 
            
        # The name of the column to apply the pattern
        src_col = f"{src_div_name}{self.src_col_suffix}"
        
        # The new column
        dst_col = f'{div_name}{self.src_col_suffix}'

        # The suffix of the id for the new table
        id_suffix = self._ohco_type['m']
        
        # ACTIONS

        # Identify lines with milestone markers
        div_lines = df[src_col].str.contains(div_pat, regex=True, case=False) # May want to parametize case
        
        # Add a new column with the ids for the milestones
        df.loc[div_lines, div_name] = [i+1 for i in range(df.loc[div_lines].shape[0])]
        
        # Forward fill to include members of the div
        df[div_name] = df[div_name].ffill()
        
        # Remove everything before first div
        df = df.loc[~df[div_name].isna()] 
        
        # Remove lines milestone markers
        df = df.loc[~div_lines] 
        
        # Cast values to ints (from floats)
        df[div_name] = df[div_name].astype('int')
        
        # Make a big doc string from the named lines
        df = df.groupby(self.ohco_names[:ohco_level+1])[src_col].apply(lambda x: '\n'.join(x)).to_frame(dst_col)
        
        # Strip the new doc string
        df[dst_col] = df[dst_col].str.strip()
                
        # Rename index
        df.index.name = f"{div_name}{id_suffix}"
        
        # Return new dataframe
        return df

    def _split_by_delimitter(self, df, ohco_level):
        """Split and chunk text by a delimmitter, for paragraphs, sentences, and tokens"""
        
        # DEFINITIONS

        # The name of the div (content object level) to be created
        div_name = self.ohco_pats[ohco_level][0]

        # The milestone pattern to used to infer the div
        div_pat = self.ohco_pats[ohco_level][1]
        
        # Notify
        print(f"by delimitter {div_pat}")
        
        # The suffix of the id for the new table
        id_suffix = self._ohco_type['d']

        # The parent div (content object level) -- we assume this is not the first parsing
        src_div_name = self.ohco_names[ohco_level-1]
        
        # The name of the column to apply the pattern
        src_col = f"{src_div_name}{self.src_col_suffix}"
        
        # Tne new column
        dst_col = f'{div_name}{self.src_col_suffix}'
        
        # The new index
        dst_index = df.index.names + [div_name + id_suffix]
        
        # ACTIONS
        
        # Split source column by pattern and stack into new table
        df = df[src_col].str.split(div_pat, expand=True).stack().to_frame(dst_col) #.copy()
        
        # Name index
        df.index.names = dst_index
        
        # Remove join content (e.g. new lines)
        df[dst_col] = df[dst_col].str.replace(self.join_pat, ' ', regex=True)
        
        # Remove empty lines
        df = df[~df[dst_col].str.contains(r'^\s*$', regex=True)]
        
        # Return
        return df

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

    def gather_tokens(self, level=0, collapse=False):
        """Gather tokens into strings for arbitrary OHCO level."""
        max_level = len(self.OHCO) - 2 # Can't gather tokens at the token level :)
        if level > max_level:
            print(f"Level {level} too high. Try between 0 and {max_level}")
        else:
            level_name = self.OHCO[level]
            idx = self.TOKENS.index.names[:level+1]
            return self.TOKENS.groupby(idx).term_str.apply(lambda x: x.str.cat(sep=' '))\
                .to_frame(f'{level_name}_str')


if __name__ == '__main__':
    src_file = '../data/gutenberg/pg42324.txt'
    # ohco_pats = [('chap', r'^(?:INTRODUCTION|PREFACE|LETTER|CHAPTER)\.?\b', 'm')]
    ohco_pats = [('chap', r'^(?:LETTER|CHAPTER)\b', 'm')]
    clip_pats = [r'START', r'END']
    test= TextImporter(src_file=src_file, ohco_pats=ohco_pats, clip_pats=clip_pats)
    test.import_source().parse_tokens()
    print(test.TOKENS.head())
    print(test.gather_tokens(1))