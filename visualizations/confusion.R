source("~/Settings/startup.R")

get_confusions = function(filename) {
  df = read.csv(filename)
  confusions = table(df) %>% as.data.frame() %>%
    group_by(labels) %>%
    mutate(prop_gold = Freq / sum(Freq)) %>%
    ungroup()
  diagonal = confusions %>% filter(labels==preds)
  accuracy_levels = (0:13)[diagonal$labels[order(diagonal$prop_gold)]]
  accuracy_labels = c("because", "although", "but", "when",
                      "for example", "before", "after", "however",
                      "so", "still", "though", "meanwhile",
                      "while", "if")[accuracy_levels+1]
  confusions = confusions %>% 
    mutate(labels = factor(labels, levels=accuracy_levels, labels=accuracy_labels),
           preds = factor(preds, levels=accuracy_levels, labels=accuracy_labels))
  return(confusions)
}

books = get_confusions("books_LSTM_1_1024_all_tempmax_e1_confusion_test.csv") %>%
  mutate(corpus = "BookCorpus")
wiki = get_confusions("wikitext_confusion_test.csv") %>%
  mutate(corpus = "Wikitext-103")

df = rbind(books, wiki) %>%
  ungroup() %>%
  as.data.frame() %>%
  mutate(corpus = factor(corpus, levels=c("Wikitext-103", "BookCorpus")))

intuitive_levels = c(
  "if", "because",
  "however", "still", "though", "but", "although", "while",
  "when", "before", "after",
  "so", "for example", "meanwhile")

## use http://colorbrewer2.org/ to find optimal divergent color palette (or set own)
# color_palette <- colorRampPalette(c("#3794bf", "#FFFFFF", "#df8640"))(l

df %>%
  filter(corpus=="Wikitext-103") %>%
  mutate(labels = factor(labels, levels=intuitive_levels),
         preds = factor(preds, levels=intuitive_levels),
         Confusion = log(prop_gold+0.001)) %>%
  ggplot(., aes(x=preds, y=labels, fill=Confusion)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") + 
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  # scale_fill_gradientn(colours = c("#67a9cf", "#f7f7f7", "#ef8a62")) +
  # scale_fill_gradientn(colours=c("#132B43", "#56B1F7", "white"),
  #                      breaks=c(-1, -3, -5),
  #                      guide="colourbar") +
  # scale_fill_gradientn(colours=c("#132B43", "#56B1F7", "white"),
  #                      breaks=c(-1, -3, -5),
  #                      guide="colourbar") +
  # scale_fill_gradient(low="#B22222", high="white") +
  # scale_fill_gradient2(low="white", mid="#54aff5", high="#122c43") +
  #132B43
  # scale_fill_gradient2(low = "white", mid = "#56B1F7",high="#132B43",
  #                       space = "Lab", na.value = "grey50", guide = "colourbar") +
  xlab("Model Classification")
ggsave("confusion_intuition_zoom.png", width=5, height=4)

cluster_levels = c("after", "before",
                   "so", "though", "however", "still",
                   "but", "while", "although", "because",
                   "if", "when",
                   "for example", "meanwhile")

cluster_levels = c(
  "for example", "meanwhile",
  "however", "still",
  "though", "so", 
  "after", "before",
  "but", "while", "although", "because",
  "if", "when")

## complete ordering
cluster_levels = c(
  "for example", "meanwhile",
  "but", "if", "when", "while", "although", "because",
  "though", "so", "after", "before",
  "however", "still"
)

df %>%
  filter(corpus=="Wikitext-103") %>%
  mutate(labels = factor(labels, levels=cluster_levels),
         preds = factor(preds, levels=cluster_levels),
         Confusion = log(prop_gold+0.001)) %>%
  ggplot(., aes(x=preds, y=labels, fill=Confusion)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_tile() +
  # facet_wrap(~corpus) +
  ylab("True Label") +
  scale_fill_gradientn(colours = c(
    "#0571b0",
    "#92c5de",
    "#f7f7f7",
    "#f4a582",
    "#ca0020")) +
  # scale_fill_gradient(low="#a7d9ff", high="#3d7eb1") +
  # scale_fill_gradient2(low="white", mid="#54aff5", high="#122c43") +
  #132B43
  # scale_fill_gradient2(low = "white", mid = "#56B1F7",high="#132B43",
  #                       space = "Lab", na.value = "grey50", guide = "colourbar") +
  xlab("Model Classification")
ggsave("confusion_cluster_zoom.png", width=5, height=4)

wiki_vecs = df %>% select(labels, preds, prop_gold, corpus) %>% 
  spread("preds", "prop_gold") %>% 
  filter(corpus=="Wikitext-103") %>% 
  select(-corpus)
wiki_distances = wiki_vecs %>% select(-labels) %>% dist
plot(hclust(wiki_distances, method="mcquitty"), labels=wiki_vecs$labels)
plot(hclust(wiki_distances, method="average"), labels=wiki_vecs$labels)

#### best?
plot(hclust(wiki_distances, method="complete"), labels=wiki_vecs$labels)

plot(hclust(wiki_distances, method="ward.D"), labels=wiki_vecs$labels)
plot(hclust(wiki_distances, method="ward.D2"), labels=wiki_vecs$labels)
plot(hclust(wiki_distances, method="single"), labels=wiki_vecs$labels)

# the agglomeration method to be used.
# This should be (an unambiguous abbreviation of)
# one of "ward.D", "ward.D2", "single", "complete",
# "average" (= UPGMA), "mcquitty" (= WPGMA), 
# "median" (= WPGMC) or "centroid" (= UPGMC).

# 
# confusions %>%
#   ggplot(., aes(x=preds, y=labels, fill=log(prop_gold+0.001))) +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   geom_tile() +
#   ylab("true label") +
#   xlab("model classification")
# ggsave("zoomed_confusion.png", width=8, height=6)
# 
# # confusions %>% 
# #   mutate(labels = factor(labels, levels=my_levels, labels=my_labels),
# #          preds = factor(preds, levels=my_levels, labels=my_labels)) %>%
# #   ggplot(., aes(x=preds, y=labels, fill=prop_gold)) +
# #   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
# #   geom_tile() +
# #   ylab("true label") +
# #   xlab("model classification")
# # ggsave("confusion.png", width=8, height=6)