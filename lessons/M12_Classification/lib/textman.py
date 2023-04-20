import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import sqlite3

def import_source(src_file, start_line=None, end_line=None, col_name='line', id_name='line_id', strip=True):
    """Imports a raw text file into a dataframe of lines. Removes extra lines."""
    print('src_file', src_file)
    df = pd.DataFrame({col_name:open(src_file,'r').readlines()})
    if not start_line:
        start_line = 0
    if not end_line:
        end_line = len(df.index)
    df = df.loc[start_line:end_line]
    df.index.name = id_name
    if strip:
        df[col_name] = df[col_name].str.strip()
    return df
 
def group_by_milestone(df, div_name, div_pat, src_idx, src_col, tmp_col='div_idx', id_suffix='_id'):
    """Splits text lines by a milestone and groups them into units, such as a chapters from chapter headings."""
    df[div_name] = df[src_col].str.match(div_pat)
    df2 = df.loc[df[div_name], src_col].copy().reset_index(drop=True).to_frame()\
        .rename(columns={src_col:div_name})
    df2.index.name = div_name + id_suffix
    df[tmp_col] = None
    df[tmp_col] = df[df[div_name]].apply(lambda x: x.index)
    df[tmp_col] = df[tmp_col].ffill()
    df[tmp_col] = df[tmp_col].astype('int')
    df2[div_name] = df.groupby(tmp_col)[src_col].apply(lambda x: '\n'.join(x[:]))\
        .to_frame().reset_index(drop=True)
    df2.index.name = div_name + id_suffix
    return df2
 
def split_by_delimitter(df, div_name=None, div_pat=None, src_col=None, join_pat='\n', id_suffix='_id'):
    """Splits lines in smaller units of text based on a delimitter."""
    df2 = df[src_col].str.split(div_pat, expand=True).stack().to_frame()\
        .rename(columns={0:div_name}).copy()
    df2.index.names = df.index.names + [div_name + id_suffix]
    df2[div_name] = df2[div_name].str.replace(join_pat, ' ')
    df2 = df2[~df2[div_name].str.match(r'^\s*$')]
    return df2
 
def gather_tokens(df, level=0, col='token', glue=' ', collapse=False):
    """Recombines units into larger units of text."""
    idx = df.index.names[:level+1]
    df2 = df.groupby(idx)[col].apply(lambda x: glue.join(x))
    return df2
 
def normalize_tokens(df, token_col='token'):
    """Lowercases and strips tokens."""
    df['term_str'] = df[token_col].str.lower().str.strip()
    return df
 
def add_pos_to_tokens(tokens, idx=['chap_id','para_id','sent_id'], token_col='token'):
    """Adds POS tages to tokens."""
    df = tokens.groupby(idx).token.apply(lambda x: nltk.pos_tag(x.tolist()))\
        .apply(pd.Series).stack()\
        .to_frame().reset_index()\
        .rename(columns={'level_{}'.format(len(idx)):'token_id', 0:'pos'})\
        .set_index(idx + ['token_id'])
    tokens['pos'] = df.pos.apply(lambda x: x[1])
    return tokens

def create_vocab(df, col='term_str'):
    """Creates a vocabulary table from a token table."""
    terms = df[col].value_counts()\
        .to_frame().reset_index()\
        .rename(columns={'index':'term_str',col:'n'})\
        .sort_values('term_str').reset_index(drop=True)
    terms.index.name = 'term_id'
    terms['f'] = terms.n.div(terms.n.sum())
    return terms
 
def add_stems_to_vocab(vocab):
    """Adds stems to vocabulary table."""
    ps = PorterStemmer()
    vocab['stem'] = vocab['term_str'].apply(lambda x: ps.stem(x))
    return vocab
 
def link_tokens_to_vocab(tokens, vocab, drop=False):
    """Adds term_id column to tokens table."""
    tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
          .set_index('term_str').term_id)
    if drop:
        del(tokens['term_str'])
    return tokens
 
# Todo: Codify these assumptions in config
def identify_stopwords(vocab):
    """Identifies stopwords in the vocabulary table."""
    sw = set(stopwords.words('english'))
    vocab['sw'] = vocab.apply(lambda x:
        x.term_str in sw
        or len(x.term_str) <= 2
        or not x.term_str.isalpha()
        or x.n < 3,
        axis=1)
    vocab['go'] = ~vocab.sw
    return vocab
 
def remove_stopwords(df, vocab, term_id_col='term_id'):
    """Removes stopwords from the tokens table."""
    df = df[df[term_id_col].isin(vocab[vocab.go].index.values)].copy()
    return df
 
def create_doc_table(tokens, index=['chap_id', 'para_id']):
    """Create the documents table from tokens tables."""
    doc = tokens.groupby(index).term_id.count()\
        .to_frame().rename(columns={'term_id':'n'})
    return doc
 
def create_bow(tokens, idx, index_name='doc_id'):
    """Create a bag-of-words table from tokens table."""
    col = idx[-1]
    bow = tokens.groupby(idx)[col].count()\
        .to_frame().rename(columns={col:'n'})
    if index_name:
        bow.index.name = index_name
    return bow
 
def create_dtm(bow, fill_val=0):
    """Create a document-term matrix from a bag-of-words."""
    dtm = bow.unstack().fillna(fill_val)
    dtm.columns = dtm.columns.droplevel(0)
    return dtm
 
def compute_term_freq(dtm, vocab):
    dtm_tf = dtm.apply(lambda x: x / x.sum(), 1)
    vocab['tf_sum'] = dtm_tf.sum()
    return dtm, vocab
 
def compute_inv_doc_freq(dtm, vocab):
    N = len(dtm.index)
    dtm_idf = dtm.apply(lambda x: N / x[x > 0].count())
    vocab['idf'] = dtm_idf
    return dtm_idf, vocab
 
def compute_tfidf(dtm, vocab, doc, bow, sw=False):
    N = len(dtm.index)
    dtm_tfidf = dtm.apply(lambda row: row / row.sum(), 1)\
        .apply(lambda col: col * np.log10(N/col[col > 0].count()))
    vocab['tfidf_sum'] = dtm_tfidf.sum()
    doc['tfidf_sum'] = dtm_tfidf.sum(1)
    bow['tfidf'] = dtm_tfidf.stack().to_frame().rename(columns={0:'tfidf'})
    return dtm_tfidf, vocab, doc, bow
 
def get_term_id(vocab, term):
    term_id = vocab[vocab.term==term].index[0]
    return term_id
 
def get_term(vocab, term_id):
    term = vocab.loc[term_id].term
    return term
 
def create_tokens_and_vocab(paras, idx=['chap_id','para_id','sent_id'], 
                            src_col='para', pos=False, drop=False):
    """Composite function to genereate tokens and vocabulary tables from units (e.g. paragraphs)."""
    cfg = dict(
        sent = dict(
            div_name = 'sent',
            div_pat = r'(?:[":;.?!\(\)]|--)',
            src_col = src_col,
            join_pat = ' '
        ),
        token = dict(
           div_name = 'token',
           div_pat = r'\W+',
           src_col = 'sent',
           join_pat = ' '
        )
    )
    
    sents = split_by_delimitter(paras, **cfg['sent'])
    tokens = split_by_delimitter(sents, **cfg['token'])
    tokens = normalize_tokens(tokens)
    if pos:
        tokens = add_pos_to_tokens(tokens, idx=idx)
    vocab = create_vocab(tokens)
    vocab = add_stems_to_vocab(vocab)
    vocab = identify_stopwords(vocab)
    tokens = link_tokens_to_vocab(tokens, vocab, drop=drop)
    tokens = remove_stopwords(tokens, vocab)
    return tokens, vocab
 
def add_doc_len_features(df, str_col, prefix='doc_'):
    len = prefix + 'len'
    df[len] = df[str_col].str.len()
    df[prefix + 'z'] = (df[len] - df[len].mean()).div(df[len].str())
    df[prefix + 's'] = (df[len] / df[len].max()).multiply(100).round().astype('int')
    df[prefix + 'p'] = df[len] / df[len].sum()
    df[prefix + 'h'] = df[prefix+'p'].multiply(np.log2(df[prefix+'p'])) * -1
    return df
 
def put_to_db(db, df, table_name, index=True, if_exists='replace'):
    r = df.to_sql(table_name, db, index=index, if_exists=if_exists)
    return r
 
def get_from_db(db, table_name):
    df = pd.read_sql("SELECT * FROM {}".format(table_name), db)
    return df
 
def get_table(table, db_file, fields='*', index_col=None):
    if type(fields) is list:
        fields = ','.join(fields)
    with sqlite3.connect(db_file) as db:
        return pd.read_sql("select {} from {}".format(fields, table), db, index_col=index_col)

def get_sql(sql, db_file, params=None):
    with sqlite3.connect(db_file) as db:
        return pd.read_sql(sql, db, params)
