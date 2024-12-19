library(tidyverse)

# Limit features to those that appear in at least 30 firms (~5% of total)

bin = read.csv("~/Dropbox/1_proj/logos/data/y_bin_all_py2.csv", row.names = 1)
cat = read.csv("~/Dropbox/1_proj/logos/data/y_mult_ncolors_py2.csv", row.names = 1)
indus = read.csv("~/Dropbox/1_proj/logos/data/industry_codes_b2bc.csv", row.names = 1)

newbin = bin[,colSums(bin) >= 25]

newcat = cat
newcat[] = NA

# Relabel dom colors:

# old colors:
# 0 "black",
# 1 "blue_dark",
# 2 "blue_light",
# 3 "blue_medium",
# 4 "brown",
# 5 "green_dark",
# 6 "green_light",
# 7 "grey_dark",
# 8 "grey_light",
# 9 "orange",
# 10 "red",
# 11 "red_dark",
# 12 "yellow"

newdc = ifelse(cat[,1] %in% c(5,6), 5,
          ifelse(cat[,1] %in% c(7,8), 7,
          ifelse(cat[,1] %in% c(10,11), 10, 
          ifelse(cat[,1] %in% c(4,9,12), 12, cat[,1]))))

newdc = as.numeric(as.factor(newdc)) - 1

# new color key:
# 0 "black",
# 1 "blue_dark",
# 2 "blue_light",
# 3 "blue_medium",
# 4 "green"
# 5 "grey"
# 6 "red"
# 7 "other" (yellow-brown-orange)

newcat[,1] = newdc

old_levels = list()
changed = rep("Yes", ncol(cat))


new_bm = fct_lump_min(as.factor(cat[,j]), 30)

for(j in 2:ncol(cat)){
  redone = fct_lump_min(as.factor(cat[,j]), 30)
  old_levels[[j]] = levels(redone)
  levels(redone) = 0:(length(levels(redone))-1)
  
  if(length(table(redone)) == length(table(cat[,j]))){
    redone = cat[,j]
    old_levels[[j]] = levels(as.factor(cat[,j]))
    changed[j] = "No"
  }
  newcat[,j] = redone
}

write.csv(newbin, file = "~/Dropbox/1_proj/logos/data/y_bin_filtered.csv", row.names = TRUE)
write.csv(newcat, file = "~/Dropbox/1_proj/logos/data/y_cat_filtered.csv", row.names = TRUE)
