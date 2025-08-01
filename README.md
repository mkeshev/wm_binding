# WMbinding
Code for a working memory model of sentence processing as binding morphemes to syntactic positions


## How to run:

torch_vecs.py softmax_temp root_size root_cos agr_size num_vals pos_size pos_cos

For several distractors, separate cosine values (both for roots and for agreement) by a comma with no intervening space; e.g., 0.1,0.8. 

Number features of target and distractor(s) are encoded as 0 (singular) and 1 (plural). E.g., use 101 for 'plural, singular, plural'. The number feature of the root must be first. For several distractors, their order must be consistent across root_cos, pos_cos and num_val.
