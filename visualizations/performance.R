df = read.csv("performance.csv")
library(ggplot2)
library(dplyr)
library(tidyr)
agg = df %>% group_by(marker) %>%
  summarise(performance=mean(performance))
df %>% mutate(marker = factor(marker, levels=agg$marker[order(agg$performance)])) %>%
  ggplot(., aes(x=marker, y=performance, fill=run)) +
  geom_bar(stat="identity", position="dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("performance.png", width=10)
