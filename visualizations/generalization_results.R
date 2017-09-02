df = read.csv("generalization_results.csv")
source("~/Settings/startup.R")

tasks = df %>% group_by(task) %>% summarise(performance = mean(performance))

df = df %>% mutate(task = factor(task, levels = tasks$task[order(tasks$performance)]))

df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, shape=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  scale_shape_manual(values=1:10) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set.png", width=10, height=6)

df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="dep") %>%
  filter(layers==2) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, shape=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom1.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="string") %>%
  filter(layers==1) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom2.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  filter(ssplit=="dep") %>%
  filter(layers==1) %>%
  filter(split=="rand") %>%
  mutate(group = paste(ssplit, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=marker_set, colour=marker_set)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_marker_set_zoom3.png", width=5, height=4)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=ssplit, colour=ssplit)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_extraction_ssplit.png", width=10, height=6)


df %>% 
  filter(marker_set %in% c("but/because", "five")) %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, layers, split)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=ssplit, colour=ssplit)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_extraction_ssplit_zoom.png", width=10, height=6)


df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers.png", width=10, height=6)

df %>% 
  filter(marker_set=="but/because") %>%
  filter(ssplit=="string") %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers_zoom1.png", width=5, height=4)

df %>% 
  filter(marker_set %in% c("but/because", "five")) %>%
  filter(ssplit=="string") %>%
  filter(split=="rand") %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split)) %>%
  mutate(layers = factor(layers)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, fill=layers, colour=layers)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  facet_wrap(~ group) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("which_num_layers_zoom2.png", width=8, height=4)


new_df = df %>% 
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(marker_set, ssplit, split, layers))
new_df %>%
  ggplot(., aes(x=task, y=performance, colour=group, shape=group)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  scale_shape_manual(values=1:20) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("all_generalization_experiments.png", width=12, height=8)



df %>% 
  filter(marker_set %in% c("SNLI", "five", "five (books)", "InferSent", "SkipThought")) %>%
  filter(ssplit=="string" | is.na(ssplit)) %>%
  filter( (marker_set=="SNLI" | split=="rand") | marker_set%in%c("SkipThought", "InferSent") ) %>%
  filter(layers==1 | is.na(layers)) %>%
  filter(!(task %in% c("SICK-Relatedness", "SICK-Entailment"))) %>%
  rename( training_task = marker_set) %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(training_task, model)) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, colour=group, shape=group)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=0.7) +
  # geom_line(aes(group=group)) +
  # facet_wrap(~ model) +
  scale_shape_manual(values=1:15) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("task_and_model.png", width=8, height=4)


df %>% 
  mutate(
    task = factor(task, 
                  levels=c("SICK-Relatedness", "SICK-Entailment", 
                           "MR", "SST", "CR", "TREC", "MPQA", 
                           "SUBJ"))) %>%
  filter(marker_set %in% c("SNLI", "five (books)", "InferSent", 
                           "SkipThought")) %>%
  filter(marker_set!="SNLI" | model=="temporal max pooling") %>%
  filter(marker_set!="SNLI") %>%
  # filter(!(task %in% c("SICK-Relatedness", "SICK-Entailment"))) %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(model = factor(marker_set,
                        levels=c("SkipThought", "InferSent", "five (books)"), 
                        labels=c("SkipThought", "InferSent", "Discourse"))) %>%
  filter(model != "Discourse" & model != "InferSent") %>%
  ggplot(., aes(x=task, y=performance, colour=model#, shape=model
  )) +
  geom_point(alpha=1, size=3) +
  scale_shape_manual(values=1:15) +
  scale_color_manual(values=c("#D45954", "#7BDB45", "#11DBE3")) +
  theme_black +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0, 100)
ggsave("skipthought.png", width=8, height=6)



df %>% 
  mutate(
    task = factor(task, 
                  levels=c("SICK-Relatedness", "SICK-Entailment", 
                           "MR", "SST", "CR", "TREC", "MPQA", 
                           "SUBJ"))) %>%
  filter(marker_set %in% c("SNLI", "five (books)", "InferSent", 
                           "SkipThought")) %>%
  filter(marker_set!="SNLI" | model=="temporal max pooling") %>%
  filter(marker_set!="SNLI") %>%
  # filter(!(task %in% c("SICK-Relatedness", "SICK-Entailment"))) %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(model = factor(marker_set,
                        levels=c("SkipThought", "InferSent", "five (books)"), 
                        labels=c("SkipThought", "InferSent", "Discourse"))) %>%
  filter(model != "Discourse") %>%
  ggplot(., aes(x=task, y=performance, colour=model#, shape=model
  )) +
  geom_point(alpha=1, size=3) +
  scale_shape_manual(values=1:15) +
  scale_color_manual(values=c("#D45954", "#7BDB45", "#11DBE3")) +
  theme_black +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0, 100)
ggsave("sota_comparison.png", width=8, height=6)



df %>% 
  mutate(
    task = factor(task, 
                  levels=c("SICK-Relatedness", "SICK-Entailment", 
                           "MR", "SST", "CR", "TREC", "MPQA", 
                           "SUBJ"))) %>%
  filter(marker_set %in% c("SNLI", "five (books)", "InferSent", 
                           "SkipThought")) %>%
  filter(marker_set!="SNLI" | model=="temporal max pooling") %>%
  filter(marker_set!="SNLI") %>%
  # filter(!(task %in% c("SICK-Relatedness", "SICK-Entailment"))) %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(model = factor(marker_set,
                        levels=c("SkipThought", "InferSent", "five (books)"), 
                        labels=c("SkipThought", "InferSent", "Discourse"))) %>%
  ggplot(., aes(x=task, y=performance, colour=model#, shape=model
                )) +
  geom_point(alpha=1, size=3) +
  scale_shape_manual(values=1:15) +
  scale_color_manual(values=c("#D45954", "#7BDB45", "#11DBE3")) +
  theme_black +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylim(0, 100)
ggsave("best_performing_model_comparison.png", width=8, height=6)


df %>%
  filter(marker_set %in% c("five", "SNLI", "five (books)")) %>%
  filter(model == "temporal max pooling") %>%
ggplot(., aes(x=task, y=performance, colour=marker_set, shape=marker_set
)) +
  geom_point(alpha=0.7) +
  scale_shape_manual(values=1:15) +
  scale_color_manual(values=c("#D45954", "#7BDB45", "#11DBE3")) +
  facet_wrap(~model) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("temporal_max_pooling.png", width=8, height=6)

df %>%
  filter(marker_set %in% c("five", "SNLI", "five (books)")) %>%
  filter(model == "temporal max pooling") %>%
  ggplot(., aes(x=task, y=performance, colour=marker_set, shape=marker_set
  )) +
  geom_point(alpha=0.7) +
  scale_shape_manual(values=1:15) +
  scale_color_manual(values=c("#D45954", "#7BDB45", "#11DBE3")) +
  facet_wrap(~model) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("temporal_max_pooling.png", width=8, height=6)


df %>% 
  filter(marker_set %in% c("SNLI", "five", "five (books)", "InferSent", "SkipThought")) %>%
  filter(ssplit=="string" | is.na(ssplit)) %>%
  filter( (marker_set=="SNLI" | split=="rand") | marker_set%in%c("SkipThought", "InferSent") ) %>%
  filter(layers==1 | is.na(layers)) %>%
  # filter(!(task %in% c("SICK-Relatedness", "SICK-Entailment"))) %>%
  rename( training_task = marker_set) %>%
  mutate(performance = ifelse(task=="SICK-Relatedness",
                              performance*100, performance)) %>%
  mutate(group = paste(training_task, model)) %>%
  filter(!(group %in% c("five base LSTM",
                        "five temporal mean pooling",
                        "SNLI base LSTM"))) %>%
  # mutate(split = factor(ifelse(is.na(split), "unknown", char(split)))) %>%
  ggplot(., aes(x=task, y=performance, colour=group, shape=group)) +
  # geom_bar(stat="identity", position="dodge") +
  geom_point(alpha=1, size=2) +
  # geom_line(aes(group=group)) +
  # facet_wrap(~ model) +
  scale_shape_manual(values=1:15) +
  # ylim(0,100) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("task_and_model.png", width=8, height=4)
