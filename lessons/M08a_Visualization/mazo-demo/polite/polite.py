import re
import os
import sys
import pandas as pd
from lxml import etree
from scipy import stats
import gzip
from sqlalchemy import create_engine

class Polite():
    """
    MALLET parameters used: 'output-topic-keys', 'output-doc-topics',
    'word-topic-counts-file', 'topic-word-weights-file',
    'xml-topic-report', 'xml-topic-phrase-report',
    'diagnostics-file', 'output-state'
    """

    class TableDef():
        def __init__(self, index=[], cols=[]):
            self.cols = cols
            self.index = index

    schema = dict(
        DOC = TableDef(['doc_id']),
        DOCTOPIC_NARROW = TableDef(['doc_id', 'topic_id']),
        DOCTOPIC = TableDef(['doc_id']),
        DOCWORD = TableDef(['doc_id', 'word_id']),
        PHRASE = TableDef(['phrase_str']),
        TOPIC = TableDef(['topic_id']),
        TOPICPHRASE = TableDef(['topic_id', 'topic_phrase']),
        TOPICWORD_DIAGS = TableDef(['topic_id', 'word_id']),
        TOPICWORD_NARROW = TableDef(['word_id', 'topic_id']),
        TOPICWORD_WEIGHTS = TableDef(['topic_id', 'word_str']),
        TOPICWORD = TableDef(['word_id']),
        VOCAB = TableDef(['word_id'])
    ) 

    def __init__(self, config_file, tables_dir='./', save_mode='csv'):
        """Initialize MALLET with trial name"""
        self.config_file = config_file
        self.tables_dir = tables_dir
        self._convert_config_file()
        self.save_mode = save_mode

        if self.save_mode == 'sql':
            engine = create_engine(f'sqlite:///{self.tables_dir}model.db', echo=True)
            self.db = engine.connect()

    def __del__(self):
        if self.save_mode == 'sql':
            self.db.close()

    def save_table(self, df, table_name):
        self.schema[table_name].cols = df.columns
        if self.save_mode == 'sql':
            df.to_sql(table_name, self.db, if_exists='replace', index=True)    
        elif self.save_mode == 'csv':
            df.to_csv(self.tables_dir + f'{table_name}.csv')

    def get_table(self, table_name):
        index_cols = self.schema[table_name].index
        if self.save_mode == 'sql':
            df = pd.read_sql_table(table_name, self.db, 
            index_col = index_cols)
        elif self.save_mode == 'csv':
            df = pd.read_csv(self.tables_dir + f'{table_name}.csv', 
                index_col=index_cols)
        else:
            raise ValueError("No save method!")
        return df

    def _convert_config_file(self):
        """Converts the MALLLET config file into a Python dictionary."""
        self.config = {}
        with open(self.config_file, 'r') as cfg:
            for line in cfg.readlines():
                if not re.match(r'^#', line):
                    a, b = line.split()
                    b = b.strip()
                    if re.match(r'^\d+$', b):
                        b = int(b)
                    elif re.match(r'^\d+\.\d*$', b):
                        b = float(b)
                    elif re.match(r'^TRUE$', b, flags=re.IGNORECASE):
                        b = True
                    elif re.match(r'^FALSE$', b, flags=re.IGNORECASE):
                        b = False
                    self.config[a] = b
        
        # config = pd.DataFrame(self.config)

    def get_source_file(self, src_file_key):
        src_file = self.config[src_file_key]
        if not os.path.isfile(src_file):
            print(f"File {src_file} for {src_file_key} does not exist. Try running MALLET first.")
            sys.exit(1)
        else:
            return src_file

    def import_table_state(self):
        """Import the state file into docword table"""
        src_file = self.get_source_file('output-state')
        with gzip.open(src_file, 'rb') as f:
            docword = pd.DataFrame(
                [line.split() for line in f.readlines()[3:]],
                columns=['doc_id', 'src', 'word_pos', 'word_id', 'word_str', 'topic_id'])
        docword = docword[['doc_id', 'word_id', 'word_pos', 'topic_id']]
        docword = docword.astype('int')
        docword = docword.set_index(['doc_id', 'word_id'])

        # SAVE
        self.save_table(docword, 'DOCWORD')


    def import_table_topic(self):
        """Import data into topic table"""
        src_file = self.get_source_file('output-topic-keys')
        topic = pd.read_csv(src_file, sep='\t', header=None, index_col='topic_id',
            names=['topic_id', 'topic_alpha', 'topic_words'])
        topic['topic_alpha_zscore'] = stats.zscore(topic.topic_alpha)

        # SAVE
        self.save_table(topic, 'TOPIC')

        
    def import_tables_topicword_and_word(self):
        """Import data into topicword and word tables"""
        src_file = self.get_source_file('word-topic-counts-file')
        WORD = []
        TOPICWORD = []
        with open(src_file, 'r') as src:
            for line in src.readlines():
                row = line.strip().split()
                word_id, word_str = row[0:2]
                WORD.append((int(word_id), word_str))
                for item in row[2:]:
                    topic_id, word_count = item.split(':')
                    TOPICWORD.append((int(word_id), int(topic_id), int(word_count)))

        # May use schema for indexes
        word = pd.DataFrame(WORD, columns=['word_id', 'word_str']).set_index('word_id')
        topicword = pd.DataFrame(TOPICWORD, columns=['word_id', 'topic_id', 'word_count'])\
            .set_index(['word_id', 'topic_id'])
        topicword_wide = topicword.unstack(fill_value=0)
        topicword_wide.columns = topicword_wide.columns.droplevel(0)
        topicword_wide = topicword_wide / topicword_wide.sum()

        src_file2 = self.get_source_file('topic-word-weights-file')
        topicword_w = pd.read_csv(src_file2,  sep='\t', names=['topic_id','word_str','word_wgt'])\
            .set_index(['topic_id','word_str'])

        # COMBINE TOPICWORD_NARROW AND TOPICWORD_WEIGHTS
        # Note that word weights are just smoothed values
        # So we really should only save the smoothing beta parameter, e.g. .01
        # Get beta from self.config['beta']
        # Should have model table for this stuff ... import the config file
        # topicword_w = topicword_w.reset_index()
        # topicword_w['word_id'] = topicword_w.word_str.map(word.reset_index().set_index('word_str').word_id)
        # topicword_w = topicword_w.set_index(['topic_id','word_id'])
        # topicword['word_wgt'] = topicword_w.word_wgt
        
        # SAVE
        self.save_table(word, 'VOCAB')
        self.save_table(topicword, 'TOPICWORD_NARROW')
        self.save_table(topicword_wide, 'TOPICWORD')
        # self.save_table(topicword_w, 'TOPICWORD_WEIGHTS')


    def import_table_doctopic(self):
        """Import data into doctopic table"""
        src_file = self.get_source_file('output-doc-topics')
        doctopic = pd.read_csv(src_file, sep='\t', header=None)
        cols = ['doc_id', 'doc_tmp'] + [t for t in range(doctopic.shape[1]-2)]
        doctopic.columns = cols
        doctopic = doctopic.set_index('doc_id')
        doc = doctopic.doc_tmp.str.split(',', expand=True).iloc[:, :2]
        doc.columns = ['src_doc_id', 'doc_label']
        doc.index.name = 'doc_id'
        doctopic = doctopic.drop('doc_tmp', axis=1)
        doctopic_narrow = doctopic.unstack().to_frame('topic_weight')
        doctopic_narrow.index.names = ['doc_id', 'topic_id']
        doctopic_narrow['topic_weight_zscore'] = stats.zscore(doctopic_narrow.topic_weight)
        
        # SAVE
        self.save_table(doctopic, 'DOCTOPIC')
        self.save_table(doc, 'DOC')
        self.save_table(doctopic_narrow, 'DOCTOPIC_NARROW')


    def import_table_topicphrase(self):
        """Import data into topicphrase table"""
        src_file = self.get_source_file('xml-topic-phrase-report')
        TOPICPHRASE = []
        tree = etree.parse(src_file)
        for topic in tree.xpath('/topics/topic'):
            topic_id = int(topic.xpath('@id')[0])
            for phrase in topic.xpath('phrase'):
                phrase_weight = float(phrase.xpath('@weight')[0])
                phrase_count = int(phrase.xpath('@count')[0])
                topic_phrase = phrase.xpath('text()')[0]
                TOPICPHRASE.append((topic_id, topic_phrase, phrase_weight, phrase_count))
        topicphrase = pd.DataFrame(TOPICPHRASE, 
            columns=['topic_id', 'topic_phrase', 'phrase_weight', 'phrase_count'])

        # Add phrase list to TOPIC
        # MOVE TO add_topic_glosses()
        topic = self.get_table('TOPIC')
        topic['phrases'] = topicphrase.groupby('topic_id').apply(lambda x: ', '.join(x.topic_phrase))

        topicphrase.set_index(['topic_id', 'topic_phrase'], inplace=True)

        phrase = topicphrase.value_counts('topic_phrase').to_frame('n_topics').sort_index()
        phrase['n_words'] = topicphrase.groupby('topic_phrase').phrase_count.sum().sort_index()
        M = topicphrase.sort_index().reset_index()
        M['topic_name'] = M.topic_id.astype('str').str.zfill(2)
        phrase['topic_list'] = M.groupby('topic_phrase').topic_name.apply(lambda x: ' '.join(x))\
            .to_frame('topic_list').sort_index()
        del(M)
        phrase['topic_weight_mean'] = topicphrase.groupby(['topic_phrase']).mean().phrase_weight.sort_index()

        topic['topic_label'] = topic.index.astype('str').str.zfill(2) + ': ' + topic.topic_words

        # SAVE
        self.save_table(phrase, 'PHRASE')
        self.save_table(topicphrase, 'TOPICPHRASE')
        self.save_table(topic, 'TOPIC')


    def add_topic_glosses(self):
        """Add glosses to topic table"""
        topicphrase = self.get_table('TOPICPHRASE')
        topic = self.get_table('TOPIC')
        topic['topic_gloss'] = topicphrase['phrase_weight'].unstack().idxmax(1)
        
        # SAVE
        self.save_table(topic, 'TOPIC')

    def add_diagnostics(self):
        """Add diagnostics data to topics and topicword_diags tables"""
        src_file = self.get_source_file('diagnostics-file')
        
        TOPIC = []
        TOPICWORD = []

        # Schema
        tkeys = ['id', 'tokens', 'document_entropy', 'word-length', 'coherence',
                 'uniform_dist', 'corpus_dist',
                 'eff_num_words', 'token-doc-diff', 'rank_1_docs',
                 'allocation_ratio', 'allocation_count',
                 'exclusivity']
        tints = ['id', 'tokens']
        wkeys = ['rank', 'count', 'prob', 'cumulative', 'docs', 'word-length', 'coherence',
                 'uniform_dist', 'corpus_dist', 'token-doc-diff', 'exclusivity']
        wints = ['rank', 'count', 'docs', 'word-length']

        tree = etree.parse(src_file)
        for topic in tree.xpath('/model/topic'):
            tvals = []
            for key in tkeys:
                xpath = f'@{key}'
                if key in tints:
                    tvals.append(int(float(topic.xpath(xpath)[0])))
                else:
                    tvals.append(float(topic.xpath(xpath)[0]))
            TOPIC.append(tvals)
            for word in topic.xpath('word'):
                wvals = []
                topic_id = tvals[0]  # Hopefully
                wvals.append(topic_id)
                word_str = word.xpath('text()')[0]
                wvals.append(word_str)
                for key in wkeys:
                    xpath = f'@{key}'
                    if key in wints:
                        wvals.append(int(float(word.xpath(xpath)[0])))
                    else:
                        wvals.append(float(word.xpath(xpath)[0]))
                TOPICWORD.append(wvals)

        tkeys = ['topic_{}'.format(re.sub('-', '_', k)) for k in tkeys]
        wkeys = ['topic_id', 'word_str'] + wkeys
        wkeys = [re.sub('-', '_', k) for k in wkeys]

        topic_diags = pd.DataFrame(TOPIC, columns=tkeys).set_index('topic_id')
        
        topics = self.get_table('TOPIC')
        topics = pd.concat([topics, topic_diags], axis=1)
        topicword_diags = pd.DataFrame(TOPICWORD, columns=wkeys).set_index(['topic_id', 'word_str'])

        word = self.get_table('VOCAB').reset_index().set_index('word_str')
        topicword_diags['word_id'] = topicword_diags.apply(lambda x: word.loc[x.name[1]].word_id, axis=1)
        topicword_diags = topicword_diags.reset_index().set_index(['topic_id', 'word_id'])

        # SAVE
        self.save_table(topics, 'TOPIC')
        self.save_table(topicword_diags, 'TOPICWORD_DIAGS')

    def do_all(self):
        """Run all importers and adders"""
        self.import_table_state()
        self.import_table_topic()
        self.import_tables_topicword_and_word()
        self.import_table_doctopic()
        self.import_table_topicphrase()
        self.add_diagnostics()
        self.add_topic_glosses()


