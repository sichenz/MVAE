setwd("/media/ryan/hdd/Dropbox/1_proj/logos/")
bd <- read.csv("data/new_sources/brandirectory/brandirectory_500.csv")
f500 <- read.csv("data/new_sources/fortune_500.csv")

bd$name <- tolower(bd$name)
f500$name <- tolower(f500$name)
f500$name <- gsub(" ", "-", gsub("[[:punct:]]", "", gsub(".com"," com", f500$name)))

pmatch(f500$name, bd$name, duplicates.ok = FALSE)
partial <- sapply(as.character(bd$name), function(x) startsWith(gsub(" ", "-", f500$name), x))
match <- apply(partial, 1, function(x) ifelse(any(x), which(x), NA))
as.character(bd$name[na.omit(match)])

drop.words <- function(name){
  dropped <- name[!(name %in% tolower(c("The","Company","Corporation","com","&","Co.","Cos.",
                     "Incorporated","Group","Holding","Companies","L.P.","Inc",
                     "Inc.","Co","Corp.","USA","Holdings")))]
  paste(dropped, collapse="-")
}

oldnames <- f500$name
newnames <- do.call(c,lapply(strsplit(as.character(f500$name), " "), drop.words))
f500$name <- newnames

newmatch <- pmatch(newnames, bd$name)
bd$name[which(match != pmatch(newnames, bd$name))]


merge.matches <- function(matches){
  if(all(is.na(matches))){
    out <- NA
  } else if(is.na(matches[1]) & !is.na(matches[2])){
    out <- matches[2]
  } else if(is.na(matches[2]) & !is.na(matches[1])){
    out <- matches[1]
  } else {
    out <- matches[2]
  }
  
  out
}

merged <- apply(cbind(match, pmatch(newnames, bd$name)), 1, merge.matches)
missing <- newnames[is.na(merged)]

write.csv(cbind(c(as.character(bd$name), missing), c(rep("BD",nrow(bd)), rep("F500",length(missing)))), file="data/new_sources/f500_bd_combined_names.csv")

all_brands <- as.data.frame(cbind(c(as.character(bd$name), missing), c(rep("BD",nrow(bd)), rep("F500",length(missing)))))
names(all_brands) <- c("name","source")

all_brands <- all_brands[order(all_brands$name, all_brands$source),]



