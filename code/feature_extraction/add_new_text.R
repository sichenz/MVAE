require(quanteda)

name <- "mcdonalds-old"
new_desc <- read.table(paste("~/Dropbox/1_proj/logos/code/extract_features/new_logos/", name, ".txt", sep=""), sep = "#")
new <- corpus(as.character(new_desc[1,1]))

new_dtfm <- dfm(tokens(new, remove_punct = TRUE,
                                             remove_numbers = TRUE,
                                             remove_url = TRUE,
                                             remove_twitter = TRUE,
                                             remove_hyphen = TRUE), 
                             remove = c("company",stopwords("english")), 
                             stem = TRUE)

cur_text <- read.csv("~/Dropbox/1_proj/logos/data/web_dtfm20_binary.csv", row.names = 1)
newrow <- 1*(colnames(cur_text) %in% colnames(new_dtfm))
newrow_df <- data.frame(t(newrow))
colnames(newrow_df) <- colnames(cur_text)
rownames(newrow_df) <- name
write.csv(newrow_df, file=paste("~/Dropbox/1_proj/logos/code/extract_features/new_logo_outputs/", name, "_newrow_binary.csv", sep=""))


# For the word cloud:
# wctext = as.matrix(new_dtfm)
# wctext = wctext[,(colnames(wctext) %in% colnames(cur_text))]
# repped = c()
# for(i in 1:length(wctext)){
#   repped = c(repped, rep(names(wctext)[i], wctext[i]))
# }
# paste(repped, collapse=" ")
