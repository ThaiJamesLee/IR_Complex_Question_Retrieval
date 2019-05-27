from html_creator.site_creator import HTMLCreator
from cache_embedding_vectors import Caching
import pickle


def generate_example_wiki_page():
    c = Caching(process_type='lemma')

    docs = c.create_doc_terms_as_string()

    query = 'enwiki:Khattak/Demographics'

    # relevant documents according to rankers
    retrieved_docs = { 'BM25':  {
                                     '6284b56c08ada24d89e1b22ee9ca55610b77793d',
                                      '0a63a9a1f478c5a3e96bdb7f2979c1caf9ad56d4',
                                      '50515f39d9350586976641a528caae67ae70cf1f'
                                 },
                       'TF-IDF': {
                                       '0a63a9a1f478c5a3e96bdb7f2979c1caf9ad56d4',
                                        '6284b56c08ada24d89e1b22ee9ca55610b77793d',
                                        'f5ca03bcdc7855a64e3d839b855fea9c59cf16d0'
                                   },
                       'GloVe': {
                                      '4ffd18e545f5004be8a9f95dc8f3e1a17e3980b6',
                                       'f34d7307b783232b2114de68e82f391cd5c04889',
                                       'e1b56980a8375b067e3f8abea87f65ff596b841c'
                                  }
                       }

    relevant_docs = {'304fb772c8a730dfc8b42afee83fe3fb38798dcf':
                         ['513f9ee9fd808d078ddc0aed27303ed313f040e2', '031f6a9ed221ca752881c23b86d87514411ab66a',
                          'a16c5ad8f48d8b4f50aea43a259a66cb490df929']
                     }
    HTMLCreator(query=query, docs=docs, retrieved_docs=retrieved_docs, relevant_docs=relevant_docs).generate_wiki()


generate_example_wiki_page()