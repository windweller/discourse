# data from manual:
# 
# The PDTB Research Group. 2008.
# The PDTB 2.0. Annotation Manual. Technical Report IRCS-08-01.
# Institute for Research in Cognitive Science, University of Pennsylvania.

library(dplyr)
library(tidyr)
library(ggplot2)
total = 18459
df = read.csv("discourse_connectives_from_pdtb.csv",
              col.names = c("connective", "freq")) %>%
  mutate(percent = freq/total*100) %>%
  mutate(connective = factor(connective, levels=connective[order(freq)]))

most_frequent = df %>% filter(percent>=2) %>% select(connective, percent)

print(most_frequent, row.names=F)
nrow(most_frequent)
