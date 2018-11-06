# Domain_Specific_Embeddings

### This project analyses the following 6 embeddings in order to investigate how well the semantic relations of the geological mineralisation terms are represented in those 6 word embeddings.

1. Word2Vec Pre-trained
   Source: GoogleNews, trained on 100 billion tokens, vocabulary size is 3 million
2. FastText Pre-trained
   Source: Web crawl, trained on 600 billion tokens, vocabulary size is 2 million
3. Word2Vec Raw
   Source: WAMEX dataset after pre-processing (cleaning, tokenising and lemmatising),
   trained on 42.6 million tokens,
   vocabulary size is 8,562 (terms occurred over 300 are selected)   
4. FastText Raw
   Source: WAMEX dataset after pre-processing (cleaning, tokenising and lemmatising),
   trained on 42.6 million tokens,
   vocabulary size is 8,730 (terms occurred over 300 are selected)
5. Word2Vec Terms
   Source: WAMEX dataset after pre-processing and filtering (keeping only geological entities/terms using domain dictionary),
   trained on 2.8 million tokens,
   vocabulary size is 837 (terms occurred over 100 are selected)
6. FastText Terms
   Source: WAMEX dataset after pre-processing and filtering (keeping only geological entities/terms  using domain dictionary),
   trained on 2.8 million tokens,
   vocabulary size is 838 (terms occurred over 100 are selected)
   
### "Domain_Vocabulary" Folder
This folder contains the list of geological terms such as names for commodities, minerals, rocks, geological eras, locations, and straigraphic names in Western Australia.

All word embedding models are created/loaded and then analysed with similarity and analogy queries using gensim framework.

### Script files:
1. Embedding Word2Vec PreTrained.py
   Loads google news vectors and facilitates queries
   
2. Embedding FastText PreTrained.py
   Loads web crawl vectors and facilitates queries

3. Embedding Word2Vec WAMEX all.py
   Reads the data files in the folder "wamex/data/wamex_xml", pre-processes the data, creates Word2Vec vectors and facilitates queries

4. Embedding FastText WAMEX all.py
   Reads the data files in the folder "wamex/data/wamex_xml", pre-processes the data, creates FastText vectors and facilitates queries

5. Embedding Word2Vec WAMEX entities data.py
   Reads the text file "wamex/data/WAMEX_geological_entities_allfiles.txt", creates Word2Vec vectors and facilitates queries
   Note: Use file 7 and 8 for creating "WAMEX_geological_entities_allfiles.txt" file or use the already created file in data folder.
   
6. Embedding FastText WAMEX entities data.py
   Reads the text file "wamex/data/WAMEX_geological_entities_allfiles.txt", creates FastText vectors and facilitates queries

7. Annotate_Geological_Terms.py file creates annotated files in the folder "wamex/data/wamex_xml_annotated" using the domain vocabulary.
   
8. AnnotatedEntitiesOnly_WAMEX_AllFiles.py file creates WAMEX_geological_entities_allfiles.txt file
   by getting only terms/entities from annotated files in the folder "wamex/data/wamex_xml_annotated/"
   
9. Visualisation t_SNE.py
   Visualises the word vectors as clusters using t-SNE

