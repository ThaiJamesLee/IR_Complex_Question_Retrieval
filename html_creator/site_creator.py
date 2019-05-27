# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'

import pickle

class HTMLCreator:

    def __init__(self, docs, retrieved_docs, relevant_docs, query):
        """

        :param docs: a dict containing docid: str, where str contains the paragraph
        :param relevant_docs: a list of docid sorted in order to generate the site
        """
        if relevant_docs is None or docs is None:
            raise Exception('Invalid Input: Inputs cannot be None!')
        self.docs = docs
        self.relevant_docs = relevant_docs
        self.retrieved_docs = retrieved_docs
        self.query = query
        self.para_section = pickle.load(open('process_data/paragraph_article_section.pkl', 'rb'))

    def generate_wiki(self, title='Example Wiki', resources='resources'):
        """
        Generates an html page with document text of relevant_docs.
        :param title: A title for the page.
        :param resources: If necessary, add recource folder.
        :return:
        """
        title_n = title.replace(' ', '_')
        part_first = f"""
        <!DOCTYPE html>
        <head>
            <title>{title}</title>
            <link rel="stylesheet" href="{resources}/style.css"
        </head>
        <body>
        <h1>{title}</h1>
        <hr>
        """

        part_end = """        
        <h3>Created For</h3>
        <li>University of Mannheim</li>
        <li>IR Project FSS-2019</li>
        <h3>Authors</h3>
        <li>Chinhan Chen</li>
        <li>Duc Tai Ly</li>
        <li>Wan-Ting Lin</li>
        </body>
        """

        section_1 = ''

        for docid, section in self.relevant_docs.items():
            section_1 += f'<h3>True Paragraph</h3> '
            for s_id in section:
                section_1 += f'<hr><p>{self.docs[s_id]} </p> '

        section_2 = ''
        for k, docids in self.retrieved_docs.items():
            section_2 += f'<h3>Retrieved Paragraph -- {k}</h3> '
            for s_id in docids:
                section_2 += f'<h3>{self.para_section[s_id]}</h3><hr><p>{self.docs[s_id]} </p> '

        full_html = f'{part_first}<h2>{self.query}</h2><section>{section_1}</section><section>{section_2}</section{part_end}'

        file = open(f"{title_n}.html", "w")
        file.write(full_html)
        file.close()





