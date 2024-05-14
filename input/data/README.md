* Initial documents
- Raw : raw texts just after OCR processing

* Cleaning
- Raw+TreeTagger : output of TreeTagger with specific classes kept from the raw texts
    - raw texts are first cleaned from lone punctuation/punctuation that is not directly after or before a letter ("d'abord" the quote is not lone)
    - TreeTagger is launched and the following classes are removed : ['ADV', 'DET:ART', 'DET:POS', 'KON', 'PRO', 'PRO:DEM', 'PRO:IND', 'PRO:PER', 'PRO:POS', 'PRO:REL', 'PRP:det']
    - some ambiguities (like "cours|cour" for the name "cours", or "le|la" for term "les") are resolved by taking the first proposition
- Raw+RNNTagger : output of RNNTagger from the raw texts

* Disambiguation
- Raw+Babelfy : output of BabelFy from the raw texts
- Raw+TreeTagger+Babelfy : output of BabelFy from the cleaned texts after TreeTagger
- Raw+RNNTagger+Babelfy : output of BabelFy from the cleaned texts after TreeTagger

