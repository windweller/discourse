source("~/Settings/startup.R")

draw_confusion = function(runID, clustering="") {
  filename = paste(runID, "_classifications.csv", sep="")
  df = read.csv(filename) %>%
    mutate(labels = num(labels),
           preds = num(preds))
  print(paste("overall:", mean(df$labels==df$preds)))
  print(paste("mean:",
              with(df %>% group_by(labels) %>% 
                     summarise(accuracy = mean(labels==preds)),
                   mean(accuracy))))
  class_labels = c(
    "after", "also", "although", "and", "as", 
    "because", "before", "but", "for example", 
    "however", "if", "meanwhile", "so", "still", 
    "then", "though", "when", "while")[1:18]
  df = df %>%
    mutate(labels = factor(labels, levels=0:17, labels=class_labels),
           preds = factor(preds, levels=0:17, labels=class_labels))
  confusions = table(df) %>% as.data.frame() %>%
    mutate(
      labels = char(labels),
      preds = char(preds)
    ) %>%
    group_by(labels) %>%
    mutate(prop_gold = Freq / sum(Freq)) %>%
    ungroup() %>%
    group_by(preds) %>%
    mutate(prop_classifications = Freq / sum(Freq)) %>%
    ungroup()
  diagonal = confusions %>% filter(labels==preds)
  
  if (clustering != "") {
    if (clustering == "classifications") {
      vecs = confusions %>% select(labels, preds, prop_classifications) %>%
        spread("preds", "prop_classifications")
    } else if (clustering == "gold") {
      vecs = confusions %>% select(labels, preds, prop_gold) %>%
        spread("preds", "prop_gold")
    } else {
      stop("error 2398")
    }
    distances = vecs %>% select(-labels) %>% dist
    # plot(hclust(distances, method="complete"), labels=vecs$labels)
    cluster_levels = vecs$labels[hclust(distances, method="complete")$order]
    confusions = confusions %>%
      mutate(labels = factor(labels, levels=cluster_levels),
             preds = factor(preds, levels=cluster_levels))
  }
  
  p = confusions %>%
    mutate(Confusion = log(prop_gold+0.001)) %>%
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
    xlab("Model Classification")
  print(p)
  ggsave(paste(
    runID,
    ifelse(clustering=="", "", paste("_", clustering, "_clustering", sep="")),
    ".png", sep=""), width=6, height=4)
}

draw_confusion("RUN001", clustering="gold")
draw_confusion("RUN001")
draw_confusion("RUN002", clustering="gold")
draw_confusion("RUN002")

