# install.packages("ggplot2")
# install.packages("tidyr")
# install.packages("dplyr")
library(ggplot2)
library(tidyr)
library(dplyr)

wide_df = read.csv("data/lengths.csv")
names(wide_df) = c("len1", "len2", "connective", "split", "dataset")
wide_df = wide_df %>% filter(split != "test")
df = wide_df %>% mutate(file = paste(dataset, ",", split)) %>%
  gather("var", "Length", 1:2) %>%
  mutate(`Sentence` = factor(
    var, levels=c("len1", "len2"),
    labels=c("before connective", "after connective"))) %>%
  # mutate(split = factor(split, levels=c("train", "valid"),
  #                       labels=c("training", "validation"))) %>%
  mutate(plotwindow = paste(connective, ", ", split, sep=""))

df %>% group_by(dataset, split) %>%
  summarise(mean(Length <= 35))

df %>% filter(var=="len1") %>% 
  group_by(split, dataset, connective) %>%
  summarise(N=length(split)) %>% ungroup

df %>% mutate(plotwindow = paste(
  connective, ", ",
  dataset, ", ", split, sep="")) %>%
  ggplot(., aes(x=Length, colour=`Sentence`, fill=`Sentence`)) +
  # geom_density(alpha=1/2) +
  geom_histogram(alpha=1/2, binwidth=2) +
  # geom_vline(xintercept = 35) +
  facet_wrap(#connective 
    ~ plotwindow, scale="free_y", ncol=4) +
  theme_bw() +
  xlim(0, 100) #+
  # theme(legend.position = "none")
ggsave("lengths.pdf", width=10, height=4)

wide_df %>% mutate(
  ratio = len1/len2,
  logratio=log(ratio),
  difference = len2 - len1
  ) %>%
  ggplot(., aes(x=difference)) +
  geom_histogram(binwidth=5) +
  facet_wrap(dataset ~ connective, scale="free_y") +
  xlim(-50, 50) +
  geom_vline(xintercept = 0, colour="gray")