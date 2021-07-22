# Master_Thesis
Designing a system capable of handling natural language queries is a non-trivial task and can be approached in various ways. One possible approach is to compile a database of
plausible queries and responses, then compare the user input against them and return the best match's response. This, in turn, requires some similarity measure to provide a
quantitative score of similarity between two natural-language sentences. Some of the important similarity measures are

        • Cosine Similarity
  
        • Levenshtein Distance
   
        • Word Mover's Distance
  
Using different kind of pre-trained embeddings such as GloVe[[1]](#1), Word2Vec[[2]](#2)[[3]](#3), these similarity methods have been implemented in different NLP frameworks 
like Gensim[[4]](#4), FLAIR[[5]](#5), SpaCy. The primary research question that my thesis attempts to answer is "How can we successfully compare noisy textual sentences, and what are
the different type of noisy data we can consider in this kind of implementation?"

The main contribution of this work has been listed below:

   • Formulation of requirements for the mentioned similarity measures.
  
   • Evaluation of selected similarity measures for existing English Language benchmark datasets : STS (Semantic Textual Similarity)[[6]](#6)[[7]](#7)
      & OPUS (Open Language Paraphrase Corpus)[[8]](#8).
  
   • Introduction of noisy texts to the existing datasets in the context of typographic errors[[9]](#9).
  
   • Evaluation of the similarity methods for the datasets with and without noise and error calculation.
  
   • Analysis of the absolute errors derived for different methods on the same input data with noise.
        
        
        
## References
<a id="1">[1]</a>
 Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove:  Globalvectors for word representation. In Proceedings of the 2014 conference on empirical methods in
 natural language processing (EMNLP), pages 1532–1543, 2014.
 
<a id="2">[2]</a>
 Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

<a id="3">[3]</a>
 Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Dis-tributed representations of words and phrases and their compositionality. 
 In Advances in neural information processing systems, pages 3111–3119, 2013.
 
<a id="4">[4]</a>
 Radim Rehurek and Petr Sojka. Software framework for topic modelling withlarge corpora. In Proceedings of the LREC 2010 Workshop on New Challengesfor NLP Frameworks. 
 Citeseer, 2010.

<a id="5">[5]</a>
 Alan Akbik, Tanja Bergmann, Duncan Blythe, Kashif Rasul, Stefan Schweter, and Roland Vollgraf. Flair: An easy-to-use framework for state-of-the-art nlp. In Proceedings
 of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 54–59, 2019.

<a id="6">[6]</a>
 Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. Semeval-2012task 6: A pilot on semantic textual similarity. In* SEM 2012: The First JointConference on Lexical
 and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the SixtInternational Workshop on Semantic
 Evaluation (SemEval 2012), pages 385–393,2012.

<a id="7">[7]</a>
 Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. *sem 2013 shared task: Semantic textual similarity. In Second joint conference on lexical and
 computational semantics (* SEM), volume 1: proceedings of the Main conference and the shared task: semantic textual similarity, pages 32–43, 2013.
 
 <a id="8">[8]</a>
 Mathias Creutz. Open subtitles paraphrase corpus for six languages. arXiv preprint arXiv:1809.06142, 2018.
 
 <a id="9">[9]</a>
 Kshitij Shah and Gerard de Melo. Correcting the autocorrect: Context-aware typographical error correction via training data augmentation. arXiv preprintarXiv:2005.01158, 2020.
 
 
