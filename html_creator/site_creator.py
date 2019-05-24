# -*- coding: utf-8 -*-
__author__ = 'Duc Tai Ly'


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
        <h2>Created For</h2>
        <p>University of Mannheim</p>
        <p>IR Project FSS-2019</p>
        <h3>Authors</h3>
        <li>Chinhan Chen</li>
        <li>Duc Tai Ly</li>
        <li>Wan-Ting Lin</li>
        </body>
        """

        section_1 = ''

        for docid, section in self.relevant_docs.items():
            section_1 += f'<h3>{section}</h3><hr><p>{self.docs[docid]} </p> '

        section_2 = ''
        for docid, section in self.retrieved_docs.items():
            section_2 += f'<h3>{section}</h3><hr><p>{self.docs[docid]} </p> '

        full_html = f'{part_first}<section>{section_1}</section><section>{section_2}</section{part_end}'

        file = open(f"{title_n}.html", "w")
        file.write(full_html)
        file.close()





