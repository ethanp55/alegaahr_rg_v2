library(DescTools)

# Train
df <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed.csv")

model <- lm(df$Rewards ~ df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Agent", conf.level=0.95)
print(tukey)

# Test
df <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_test.csv")

model <- lm(df$Rewards ~ df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Agent", conf.level=0.95)
print(tukey)

# Changers
df <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_test_changers.csv")

model <- lm(df$Rewards ~ df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Agent", conf.level=0.95)
print(tukey)

# Self-play
df <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_self_play.csv")

model <- lm(df$Rewards ~ df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Agent", conf.level=0.95)
print(tukey)

# Smart
df <- read.csv("/Users/mymac/masters/aat-games/generalized/analysis/pennies_game/combined_compressed_smart.csv")

model <- lm(df$Rewards ~ df$Agent)
ANOVA <- aov(model)
tukey <- TukeyHSD(x=ANOVA, "df$Agent", conf.level=0.95)
print(tukey)
