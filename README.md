# Domain_Specific_Embeddings

The project analyses 6 embeddings in terms of the geological terms.

1. Word2Vec pre-trained embedding
   Source: GoogleNews, trained on 100 billion tokens, vocabulary size is 3 million
2. FastText pre-trained embedding
   Source: Web crawl, trained on 600 billion tokens, vocabulary size is 2 million
3. Word2Vec Raw
   Source: WAMEX dataset after pre-processing (cleaning, tokenising and lemmatising), trained on 42.6 million tokens,
   vocabulary size is 8,562 (terms occurred over 300 are selected)   
4. FastText Raw
   Source: WAMEX dataset after pre-processing (cleaning, tokenising and lemmatising), trained on 42.6 million tokens,
   vocabulary size is 8,730 (terms occurred over 300 are selected)
5. Word2Vec Terms
   Source: WAMEX dataset after pre-processing and filtering (keeping only geological entities/terms), trained on 2.8 million tokens,
   vocabulary size is 837 (terms occurred over 300 are selected)
6. FastText Terms
   Source: WAMEX dataset after pre-processing and filtering (keeping only geological entities/terms), trained on 2.8 million tokens,
   vocabulary size is 838 (terms occurred over 300 are selected)
   
