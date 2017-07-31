library(dplyr)
library(tidyr)
library(ggplot2)

# data_files = as.matrix(expand.grid(split=c("train", "valid"), variable=c("S1", "S2", "labels")))
# df = do.call(rbind, lapply(1:nrow(data_files), function(i) {
#   print(i)
#   split = data_files[i,"split"]
#   variable = data_files[i,"variable"]
#   return(data.frame(
#     split=as.character(split),
#     variable=as.character(variable),
#     value=readLines(paste("data/ptb/", split, "_", variable, ".txt", sep=""), warn=F)))
# }))

df = do.call(rbind, lapply(c("train"#, "valid"
                             ), function(split) {
  variables = c("S1", "S2", "labels")
  splitdf = do.call(cbind, lapply(variables, function(variable) {
    return(readLines(paste("data/ptb/", split, "_", variable, ".txt", sep=""), warn=F))
  }))
  splitdf = splitdf %>% as.data.frame 
  colnames(splitdf) = variables
  splitdf = splitdf %>% mutate(split=split)
  return(splitdf)
}))

df %>%
  ggplot(., aes(x=labels)) +
  geom_bar(stat="count") +
  theme(axis.text.x=element_text(angle=90, hjust=1))

df = df %>% filter(labels!="and") %>%
  mutate(full_sentence = paste(S1, "   ", labels, "   ", S2))
n_but = nrow(df %>% filter(labels=="but" & split=="train"))
n_but
n_but / 1722
n_because = nrow(df %>% filter(labels=="because" & split=="train"))
n_because
n_because / 1132

df %>%
  ggplot(., aes(x=labels)) +
  geom_bar(stat="count") +
  theme(axis.text.x=element_text(angle=90, hjust=1))

head(df$full_sentence)
