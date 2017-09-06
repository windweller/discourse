# data from manual:
# 
# The PDTB Research Group. 2008.
# The PDTB 2.0. Annotation Manual. Technical Report IRCS-08-01.
# Institute for Research in Cognitive Science, University of Pennsylvania.

source("~/Settings/startup.R")
library(dplyr)
library(tidyr)
library(ggplot2)
total = 18459
df = read.csv("discourse_connectives_from_pdtb.csv",
              col.names = c("connective", "freq", "ignore")) %>%
  mutate(percent = freq/total*100) %>%
  mutate(connective = factor(connective, levels=connective[order(freq)]))

most_frequent = df %>% filter(percent>=2) %>% select(connective, percent)

df %>% filter(percent>=1) %>% select(connective, percent)

favorite_markers = c("after", "also", "as", "because", "for example", "however", "if", "when", "while", "then")
favorites = df %>% filter(connective %in% favorite_markers)

print(favorites %>% select(-freq), row.names=F)
nrow(favorites)
