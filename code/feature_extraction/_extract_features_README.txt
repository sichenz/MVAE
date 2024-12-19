# README: extract_features directory


## Actively used subdirectories and scripts:

 - new_logos = subdirectory for new logos to process, including text, BP, etc.
 
 - new_logo_outputs = where the results for new logos are stored, AND WHERE YOU CAN INPUT THE INDUSTRY TAGS

 - post-process_new_logos.ipynb = needed for processing new logos (see adding_new_logos_README.txt); given the results from extract_features_new_logo.py, this script does the following:
   * assigns the colors in new logos to the existing clusters
   * assigns the hull of the new logo to existing clusters
   * assigns the marks of the new logo to existing clusters
   
 - process_new_logo_outputs.R = given the outputs of extract_features_new_logo.py and post-process_new_logos.ipynb, this R script creates the final set of visual features for logos
 
 
 
 
## Scripts used in creating visual features for original data (i.e., tokenization):
 
 - cluster_hulls.ipynb = original hull clustering script
 
 - label_(orig)(scaled)_features.py = interactive python script used to look at what shapes were assigned to each cluster, and then type a label for that cluster
 
 - new_cluster_features.ipynb = original mark clustering script
 
 - fix_errors_erosion-char_higher.py = second stage of feature extraction to fix errors
 
 - extract_run_erosion-char_higher.py = first stage of feature extraction
 
 - logo_features_wrapper_erosion.py = main feature extraction FUNCTIONS
 

 
 
## Miscellaneous other scripts:

 - shiny subdirectory = "app" showing how the feature extraction works

 - assess_tuning.ipynb = testing tuning parameters of feature extraction, using example logo

 - quantize_example.ipynb = created example of quantization used in paper
 
 - combined_feature_extraction.ipynb = illustrates each step of the feature extraction process; now only used for illustration, not for actual processing
 
  - label_fonts.py = script to label each one of the letters from the original data (don't ever need to run again)
  
  - font_template_matching.ipynb = development of the template matching procedure
  
  - add_amp_remove_dashes.ipynb = basically what it says: adding ampersand and removing dash from font dictionary
  
  - segment_fonts.ipynb = take the original font images and create font dictionary
  
  - add_wide_add1.ipynb, label_add1.py, process_wide_fonts_add1.ipynb = adding wide fonts to segment_fonts.ipynb