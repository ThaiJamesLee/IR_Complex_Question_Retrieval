# -*- coding: utf-8 -*-
"""
1. process paragraph collection, query and page(structure) data
2. make train-test dataset
3. data store in file 'processed data', use pickle.load(open(path, 'rb')) to load data
@author wanting lin
"""

import pandas as pd
import random
from trectools import TrecQrel
import trec_car.read_data
import numpy as np
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Preprocess(object):

    def __init__(self, process_type, para_path, page_path, *query_path):
        """ get data and structure from file
        Parameters
        ----------
        para_path(str): paragraph collection input file path
        page_path(str): page file input file path
        query_path(str, optional): query file input file path

        Attributes
        ----------
        paragraphs(:obj:'list' of :obj:'Para') : paragraph collection
        paragraph_ids(:obj:'list' of 'str')) : paragraph_id
        documents(:obj:'list' of :obj:'list' of 'str')) : processed paragraph
        raw_query(): query from page information
        processed_query(): processed query
        page_content(): page paragraphs
        page_name(): page title, headings
        article_paragraph(dict {article_id:para_id})
        article_section(dict {article_id:section_id})
        section_paragraph(dict {section_id:para_id})
        para_art_sec(dict {para_id:article_id/section_id})
        train(dataframe): query(raw), q0, docid, rel
        test(dataframe): query(raw), q0, docid, rel
        process_train(dataframe): query(processed), q0, docid, rel
        process_test(dataframe): query(processed), q0, docid, rel
        """
        self.paragraphs, self.paragraph_ids, self.documents = Preprocess.load_paragraph(para_path, process_type)
        self.y_true, self.raw_query, self.processed_query = Preprocess.load_query(process_type, *query_path)
        self.page_content, self.page_name = Preprocess.load_page(page_path)
        self.article_paragraph = {}
        self.article_section = {}
        self.section_paragraph = {}
        self.para_art_sec = {}
        self.para_in_page = []
        self.section_in_page = []
        self.paragraph_article_section_relation()
        self.train, self.test = self.create_train_test()
        self.process_train = Preprocess.process_query_in_test(self.train, self.raw_query, self.processed_query)
        self.process_test = Preprocess.process_query_in_test(self.test, self.raw_query, self.processed_query)
        pickle.dump(self.process_train, open('process_data/process_train.pkl', 'wb'))
        pickle.dump(self.process_test, open('process_data/process_test.pkl', 'wb'))

    @staticmethod
    def preprocess(process_type, docs):
        # set preprocess parameters
        stemmer = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()
        regex = re.compile(f'[{re.escape(string.punctuation)}]')
        stopword = set(stopwords.words("english"))
        stopword.update('and')
        new_punctuation = np.array(string.punctuation)
        # preprocess data
        processed_doc = []
        if process_type == 'stem':
            for idx, doc in enumerate(docs):
                if type(doc) == trec_car.read_data.Paragraph:
                    doc = doc.get_text()
                processed_doc.append(" ".join(
                    [
                        stemmer.stem(i)
                        for i in regex.sub(' ', doc).split()
                        if (i != " ") & (i not in stopword) & (not i.isdigit()) & (i not in new_punctuation)
                    ]
                ))
        elif process_type == 'lemma':
            for idx, doc in enumerate(docs):
                if type(doc) == trec_car.read_data.Paragraph:
                    doc = doc.get_text()
                processed_doc.append(" ".join(
                    [
                        wordnet_lemmatizer.lemmatize(i).lower()
                        for i in regex.sub(' ', doc).split()
                        if (i != " ") & (i not in stopword) & (not i.isdigit()) & (i not in new_punctuation)
                    ]
                ))
        print(f'finish preprocess...')
        return processed_doc

    @classmethod
    def load_paragraph(cls, path, process_type):
        """ load the paragraph collection
        Parameters
        ----------
        para_path(str): paragraph collection input file path
        process_type(str): stem or lemma
        Returns
        -------
        paragraph(:obj: list of :obj:'Para')
        paragraph_ids(:obj:'list' of 'str')
        documents(:obj:'list' of :obj:'list' of 'str')
        """
        paragraphs = []
        paragraph_ids = []
        for para in trec_car.read_data.iter_paragraphs(open(path, "rb")):
            paragraphs.append(para)
            paragraph_ids.append(para.para_id)
        print(f'number of paragraphs: {len(paragraph_ids)}')

        pickle.dump(paragraphs, open("process_data/paragraphs.pkl", "wb"))
        pickle.dump(paragraph_ids, open("process_data/paragraph_ids.pkl", "wb"))
        documents = cls.preprocess(process_type, paragraphs)
        pickle.dump(documents, open(f"process_data/{process_type}_processed_paragraph.pkl", "wb"))
        print('finish loading paragraph')
        return paragraphs, paragraph_ids, documents

    @classmethod
    def load_query(cls, process_type, *args):
        """ load the query collection
        Parameters
        ----------
        para_path(str): query collection input file path
        process_type(str): stem or lemma
        Returns
        -------
        y_true(:obj: dataframe)
        raw_query(:obj:'list' of 'str')
        processed_query(:obj:'list' of :obj:'list' of 'str')
        """
        y_true = TrecQrel(args[0]).qrels_data
        processed_query = []
        for idx in range(1, len(args)):
            y_true = y_true.append(TrecQrel(args[idx]).qrels_data)
        raw_query = np.unique((list(y_true['query'])))
        for idx, query in enumerate(raw_query):
            # first 7 digits 'enwiki:' for every query is the same
            processed_query.append(query[7:].replace("/", " ").replace("%20", " "))
        processed_query = cls.preprocess(process_type, processed_query)
        print(f'finish loading query, num of query: {len(raw_query)}')
        pickle.dump(y_true, open("process_data/y_true.pkl", "wb"))
        pickle.dump(raw_query, open("process_data/raw_query.pkl", "wb"))
        pickle.dump(processed_query, open(f"process_data/{process_type}_processed_query.pkl", "wb"))
        return y_true, raw_query, processed_query

    @classmethod
    def load_page(cls, path):
        page_content = []
        page_name = []
        for page in trec_car.read_data.iter_annotations(open(path, "rb")):
            page_content.append(page)
            page_name.append(page.page_id)
        print(f'number of pages: {len(page_content)}')
        return page_content, page_name

    def allocate_page(self, section, section_name, page):
        sub_paragraph = []
        for item in section.children:
            if type(item) == trec_car.read_data.Para:
                self.para_in_page.append(item.paragraph.para_id)
                sub_paragraph.append(item.paragraph.para_id)
                self.para_art_sec[item.paragraph.para_id] = page.page_id
            elif type(item) == trec_car.read_data.List:
                self.para_in_page.append(item.body.para_id)
                self.para_art_sec[item.body.para_id] = page.page_id
            else:
                section_name = section_name + item.headingId
                self.section_in_page.append(section_name)
                self.allocate_page(item, section_name, page)
        self.section_paragraph[section_name] = sub_paragraph

    def paragraph_article_section_relation(self):
        for page in self.page_content:
            self.para_in_page = []
            self.section_in_page = []
            for item in page.skeleton:
                if type(item) == trec_car.read_data.Para:
                    self.para_in_page.append(item.paragraph.para_id)
                    self.para_art_sec[item.paragraph.para_id] = page.page_id
                elif type(item) == trec_car.read_data.List:
                    self.para_in_page.append(item.body.para_id)
                    self.para_art_sec[item.body.para_id] = page.page_id
                elif type(item) == trec_car.read_data.Section:
                    section_name = f'{page.page_id} / {item.headingId}'
                    self.section_in_page.append(section_name)
                    self.allocate_page(item, section_name, page)
                else:
                    print('Nonetype')
            self.article_paragraph[page.page_id] = self.para_in_page
            self.article_section[page.page_id] = self.section_in_page
        pickle.dump(self.article_paragraph, open("process_data/article_paragraphs.pkl", "wb"))
        pickle.dump(self.article_section, open("process_data/article_sections.pkl", "wb"))
        pickle.dump(self.section_paragraph, open("process_data/section_paragraphs.pkl", "wb"))
        pickle.dump(self.para_art_sec, open("process_data/paragraph_article_section.pkl", "wb"))

    @staticmethod
    def random_select_noise(this_item, all_item, index):
        select_item = random.sample((list(i for i in all_item if i not in this_item)), 1)
        select_paragraphs = index.get(select_item[0])
        return select_item, select_paragraphs

    def remove_intersect(self, true, noisy):
        inter = list(set(true).intersection(noisy))
        if len(inter) > 0:
            for j in inter:
                noisy.remove(j)
        for j in list(noisy):
            if j not in self.paragraph_ids:
                noisy.remove(j)
        return noisy

    def create_train_test(self):
        """ Create a train and test dataset (with noise)
        train set : all true paragraphs, 5 from other section, 5 from other article
        test set : all true paragraphs, and same amount paragraphs from other article

        """

        train = self.y_true[self.y_true['query'] == self.raw_query[0]]
        test = self.y_true[self.y_true['query'] == self.raw_query[0]]
        for idx, query in enumerate(self.raw_query):
            print(query)
            print(f"no {idx + 1} / {len(self.raw_query)}")
            # add true
            # print(f"raw true docs: {len(self.y_true[self.y_true['query'] == query])}")
            train = train.append(self.y_true[self.y_true['query'] == query])
            test = test.append(self.y_true[self.y_true['query'] == query])
            true_paragraphs = self.y_true[self.y_true['query'] == query]['docid']
            # add noise
            for para in true_paragraphs:
                this_article = self.para_art_sec[para].split("/")[0]
                noise_paragraph = []
                # train set
                if '/' not in self.para_art_sec[para]:
                    # print("it's article")
                    while len(noise_paragraph) < 10:
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article,
                                                                                         self.article_paragraph.keys(),
                                                                                         self.article_paragraph)
                        if len(other_paragraphs) > 10 - len(noise_paragraph):
                            noisy_sample = set(random.sample(other_paragraphs, 10 - len(noise_paragraph)))  # randomized
                            noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))
                            noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                else:
                    # print("it's section")
                    noise_paragraph = []
                    this_section = self.para_art_sec[para]
                    sections = self.article_sections.get(this_article)  # section under this article
                    # from other section
                    while len(noise_paragraph) < 5:
                        other_section, other_section_paragraphs = Preprocess.random_select_noise(this_section, sections,
                                                                                                 self.section_paragraph)
                        if len(other_section_paragraphs) > 5 - len(noise_paragraph):
                            noisy_sample = set(random.sample(other_section_paragraphs, 5 - len(noise_paragraph)))
                        else:
                            noisy_sample = set(random.sample(other_section_paragraphs, len(other_section_paragraphs)))
                        noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                    # from other article
                    while len(noise_paragraph) < 10:
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article,
                                                                                         self.article_paragraph.keys(),
                                                                                         self.article_paragraph)
                        if len(other_paragraphs) > 10 - len(noise_paragraph):
                            noisy_sample = set(random.sample(other_paragraphs, 10 - len(noise_paragraph)))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))
                        noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))

                # print(f'len noise: {len(noise_paragraph)}')
                train = train.append(pd.DataFrame([[query, 0, i, 0] for i in noise_paragraph], columns=train.columns))

                # test set
                same_article_paragraphs = self.remove_intersect(true_paragraphs,
                                                                list(set(self.article_paragraph.get(this_article))))
                if len(same_article_paragraphs) > 0:
                    other_article_paragraphs = []
                    while len(other_article_paragraphs) < len(same_article_paragraphs):
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article,
                                                                                         self.article_paragraph.keys(),
                                                                                         self.article_paragraph)
                        if len(other_paragraphs) > len(same_article_paragraphs) - len(other_article_paragraphs):
                            noisy_sample = set(random.sample(other_paragraphs, len(same_article_paragraphs) - len(
                                other_article_paragraphs)))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))
                        other_article_paragraphs.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                    same_article_paragraphs.extend(other_article_paragraphs)
                    test = test.append(
                        pd.DataFrame([[query, 0, i, 0] for i in same_article_paragraphs], columns=test.columns))

        train.drop_duplicates(inplace=True)
        test.drop_duplicates(inplace=True)
        test.sort_values(by="query", inplace=True)
        train.sort_values(by="query", inplace=True)
        pickle.dump(train, open('process_data/simulated_train.pkl', 'wb'))
        pickle.dump(test, open('process_data/simulated_test.pkl', 'wb'))
        return train, test

    @staticmethod
    def process_query_in_test(df, raw_query, processed_query):
        process_df = df
        # print(f'num of query: {len(raw_query)}')
        for q in raw_query:
            value = processed_query[list(raw_query).index(q)]
            process_df.loc[df['query'] == q, 'query'] = value
        return process_df


obj = Preprocess('lemma', 'test200-train/train.pages.cbor-paragraphs.cbor',
                 "test200-train//train.pages.cbor",
                 "test200-train/train.pages.cbor-toplevel.qrels",
                 "test200-train/train.pages.cbor-hierarchical.qrels",
                 "test200-train/train.pages.cbor-article.qrels")
