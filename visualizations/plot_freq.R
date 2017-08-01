df = read.csv("wiki_dist.csv", col.names = c("marker", "freq"), header = F) %>%
  mutate(prop = freq/sum(freq)) %>%
  mutate(marker = factor(marker, levels=marker[order(freq)]))
library(ggplot2)
library(dplyr)
library(tidyr)

df %>% ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity")
df %>% ggplot(., aes(x=marker, y=prop)) +
  geom_bar(stat="identity")
df[order(df$prop),] %>% select(marker, freq) %>% filter(freq > 50000)
df[order(df$prop),] %>% select(marker, freq) %>% filter(freq <= 50000)

thresholded_data = df[order(df$prop),] %>% 
  mutate(freq = ifelse(freq > 50000, 50000, freq)) %>%
  mutate(prop = freq/sum(freq))

thresholded_data %>% ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity")
thresholded_data %>% ggplot(., aes(x=marker, y=prop)) +
  geom_bar(stat="identity")

sum(thresholded_data$freq)
thresholded_data

min_sentence_length = 5
s1_maxlen = 50
s2_maxlen = s1_maxlen
# ration is len 1 / len 2
max_ratio = 5.0
min_ratio = 1/max_ratio

rounding_factor = 5
lengths = read.csv("wiki_sent_length.csv", col.names = c("marker", "len1", "len2")) %>%
  mutate(ratio = len2 / len1,
         rounded_ratio = round(ratio*rounding_factor)/rounding_factor,
         r1 = rounded_ratio,
         r2 = round(1/ratio*rounding_factor)/rounding_factor) %>%
  mutate(skew = ifelse(len1>len2, "s1", ifelse(len2>len1, "s2", "equal")),
         round_skew = ifelse(rounded_ratio==1, "equal", ifelse(len1>len2, "s1", "s2"))) %>%
  mutate(skew = factor(skew, levels=c("s1", "equal", "s2")),
         round_skew = factor(round_skew, levels=c("s1", "equal", "s2"))) %>%
  mutate(keep = ifelse(
    len1<=s1_maxlen & 
      len2<=s2_maxlen & 
      len1 >= min_sentence_length & 
      len2 >= min_sentence_length &
      ratio <= max_ratio &
      ratio >= min_ratio, T, F))

rejections = lengths %>%
  group_by(marker, keep) %>%
  summarise(N=length(keep)) %>% 
  as.data.frame
rejections %>% group_by(marker) %>%
  summarise(keep_proportion = sum(N[keep==T])/sum(N)) %>%
  as.data.frame

keep = lengths %>% filter(keep) %>%
  group_by(marker) %>% 
  summarise(freq=length(marker))
keep[order(keep$freq),] %>% select(marker, freq) %>% filter(freq > 50000)
keep[order(keep$freq),] %>% select(marker, freq) %>% filter(freq <= 50000)

thresholded_data = keep[order(keep$freq),] %>% 
  mutate(freq = ifelse(freq > 50000, 50000, freq)) %>%
  mutate(prop = freq/sum(freq)) %>%
  mutate(marker = factor(marker, levels=marker[order(freq)])) %>%
  as.data.frame

thresholded_data %>% ggplot(., aes(x=marker, y=freq)) +
  geom_bar(stat="identity")
ggsave("visualizations/thresholded_freq_plot.png")
thresholded_data

sum(thresholded_data$freq)

lengths %>% gather("sentence", "length", c(len1, len2)) %>% 
  ggplot(., aes(x=length, fill=sentence)) +
  geom_histogram() + 
  facet_wrap(~marker, scale="free_y")

lengths %>% gather("sentence", "length", c(len1, len2)) %>% 
  ggplot(., aes(x=length, fill=sentence)) +
  geom_histogram(position='identity', alpha=1/2) + 
  facet_wrap(~marker, scale="free")


lengths %>%
  filter(keep) %>%
  # filter(round_skew!="equal") %>%
  ggplot(., aes(x=rounded_ratio, fill=round_skew)) +
  geom_histogram(position="identity", binwidth=1/rounding_factor) +
  facet_wrap(~marker, scale="free")
ggsave("visualizations/ratio_histogram.png")

# lengths %>%
#   filter(keep) %>%
#   gather("direction", "ratio", c(r1, r2)) %>%
#   # filter(round_skew!="equal") %>%
#   ggplot(., aes(x=ratio, fill=round_skew, group=paste(round_skew, direction))) +
#   geom_histogram(position="identity", binwidth=1/rounding_factor, alpha=1/2) +
#   facet_wrap(~marker, scale="free")

# nrow(lengths %>% filter(skew=="s1"))
# nrow(lengths %>% filter(skew=="s2"))
# nrow(lengths %>% filter(skew=="equal"))
# lengths %>% ggplot(., aes(x=skew)) +
#   geom_bar(stat="count")

