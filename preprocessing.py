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


class Preprocess:
    def __init__(self):
        self.paragraphs = []        # paragraph data
        self.paragraph_ids = []     # paragraph id
        self.documents = []         # processed paragraph
        self.y_true = []            # query information (query, q0, docid, rel)
        self.raw_query = []         # query from page information
        self.processed_query = []   # processed query
        self.page_content = []      # page paragraphs
        self.page_name = []         # page title, headings
        self.article_paragraph = {} # dict {article_id:para_id}
        self.article_section = {}   # dict {article_id:section_id}
        self.section_paragraph = {} # dict {section_id:para_id}
        self.para_art_sec = {}      # dict {para_id:article_id/section_id}
        self.para_in_page = []
        self.section_in_page = []
        # set preprocess parameters
        self.stemmer = PorterStemmer()
        self.regex = re.compile(f'[{re.escape(string.punctuation)}]')
        self.stopword = set(stopwords.words("english"))
        self.stopword.update('and')
        self.new_punctuation = np.array(string.punctuation)

    def preprocess(self, docs):
        processed_doc = []
        # preprocess data
        for idx, doc in enumerate(docs):
            if type(doc) == trec_car.read_data.Paragraph:
                doc = doc.get_text()
            processed_doc.append(" ".join(
                [
                    self.stemmer.stem(i)
                    for i in self.regex.sub(' ', doc).split()
                    if(i != " ") & (i not in self.stopword) & (not i.isdigit()) & (i not in self.new_punctuation)
                ]
            ))
        print(f'finish preprocess...')
        return processed_doc

    def load_paragraph(self, path):
        for para in trec_car.read_data.iter_paragraphs(open(path, "rb")):
            self.paragraphs.append(para)
            self.paragraph_ids.append(para.para_id)
        print(f'number of paragraphs: {len(self.paragraph_ids)}')
        self.documents = self.preprocess(self.paragraphs)
        pickle.dump(self.paragraphs, open("processed_data/paragraphs.pkl", "wb"))
        pickle.dump(self.paragraph_ids, open("processed_data/paragraph_ids.pkl", "wb"))
        pickle.dump(self.documents, open("processed_data/processed_paragraph.pkl", "wb"))
        print('finish loading paragraph')

    def load_query(self, *args):
        self.y_true = TrecQrel(args[0]).qrels_data
        for idx in range(1, len(args)):
            self.y_true = self.y_true.append(TrecQrel(args[idx]).qrels_data)
        pickle.dump(self.y_true, open("processed_data/y_true.pkl", "wb"))
        self.raw_query = list(self.y_true['query'])
        self.raw_query = np.unique(self.raw_query)
        pickle.dump(self.raw_query, open("processed_data/raw_query.pkl", "wb"))
        for idx, query in enumerate(self.raw_query):
            self.processed_query.append(query[7:].replace("/", " ").replace("%20", " ")) # first 7 digits 'enwiki:' for every query is the same
        self.processed_query = self.preprocess(self.processed_query)
        pickle.dump(self.processed_query, open("processed_data/processed_query.pkl", "wb"))
        print('finish loading query')

    def load_page(self, path):
        for page in trec_car.read_data.iter_annotations(open(path, "rb")):
            self.page_content.append(page)
            self.page_name.append(page.page_id)
        print(f'number of pages: {len(self.page_content)}')

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
        pickle.dump(self.article_paragraph, open("processed_data/article_paragraphs.pkl", "wb"))
        pickle.dump(self.article_section, open("processed_data/article_sections.pkl", "wb"))
        pickle.dump(self.section_paragraph, open("processed_data/section_paragraphs.pkl", "wb"))
        pickle.dump(self.para_art_sec, open("processed_data/paragraph_article_section.pkl", "wb"))

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
        """
        Create a train and test dataset (with noise)
        train set : all true paragraphs, 5 from other section, 5 from other article
        test set : all true paragraphs, and same amount paragraphs from other article
        """
        print(f"number of queries {len(self.raw_query)}")
        train = self.y_true[self.y_true['query'] == self.raw_query[0]]
        test = self.y_true[self.y_true['query'] == self.raw_query[0]]
        for idx, query in enumerate(self.raw_query):
            print(f"no {idx+1} / {len(self.raw_query)}")
            # add true
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
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article, self.article_paragraph.keys(), self.article_paragraph)
                        if len(other_paragraphs) < 10:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))  # randomized
                            noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, 10))
                            noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                else:
                    # print("it's section")
                    noise_paragraph = []
                    this_section = self.para_art_sec[para]
                    sections = self.article_sections.get(this_article)  # section under this article
                    # from other section
                    while len(noise_paragraph) < 5:
                        other_section, other_section_paragraphs = Preprocess.random_select_noise(this_section, sections, self.section_paragraph)
                        if len(other_section_paragraphs) > 5 - len(noise_paragraph):
                            noisy_sample = set(random.sample(other_section_paragraphs, 5 - len(noise_paragraph)))
                        else:
                            noisy_sample = set(random.sample(other_section_paragraphs, len(other_section_paragraphs)))
                        noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                    # from other article
                    while len(noise_paragraph) < 10:
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article, self.article_paragraph.keys(), self.article_paragraph)
                        if len(other_paragraphs) > 10 - len(noise_paragraph):
                            noisy_sample = set(random.sample(other_paragraphs, 10 - len(noise_paragraph)))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))
                        noise_paragraph.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                train = train.append(pd.DataFrame([[query, 0, i, 0] for i in noise_paragraph], columns=train.columns))

                # test set
                same_article_paragraphs = self.remove_intersect(true_paragraphs, list(set(self.article_paragraph.get(this_article))))
                if len(same_article_paragraphs) > 0:
                    other_article_paragraphs = []
                    while len(other_article_paragraphs) < len(same_article_paragraphs):
                        other_article, other_paragraphs = Preprocess.random_select_noise(this_article, self.article_paragraph.keys(), self.article_paragraph)
                        if len(other_paragraphs) > len(same_article_paragraphs) - len(other_article_paragraphs):
                            noisy_sample = set(random.sample(other_paragraphs, len(same_article_paragraphs)-len(other_article_paragraphs)))
                        else:
                            noisy_sample = set(random.sample(other_paragraphs, len(other_paragraphs)))
                        other_article_paragraphs.extend(self.remove_intersect(true_paragraphs, noisy_sample))
                    same_article_paragraphs.extend(other_article_paragraphs)
                    test = test.append(pd.DataFrame([[query, 0, i, 0] for i in same_article_paragraphs], columns=test.columns))

        train.drop_duplicates(inplace=True)
        test.drop_duplicates(inplace=True)
        test.sort_values(by="query", inplace=True)
        train.sort_values(by="query", inplace=True)
        pickle.dump(train, open('processed_data/simulated_train.pkl', 'wb'))
        pickle.dump(test, open('processed_data/simulated_test.pkl', 'wb'))


def main():
    instance = Preprocess()
    instance.load_paragraph('test200/test200-train/train.pages.cbor-paragraphs.cbor')
    instance.load_query("test200/test200-train/train.pages.cbor-toplevel.qrels", \
                        "test200/test200-train/train.pages.cbor-hierarchical.qrels", \
                        "test200/test200-train/train.pages.cbor-article.qrels")
    instance.load_page("test200/test200-train//train.pages.cbor")
    instance.paragraph_article_section_relation()
    instance.create_train_test()


if __name__ == main():
    main()