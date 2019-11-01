#four_yr_fu_ids <- filter(total, Years_bl >= 4)$RID #3284
#fu_avail <- total[which(total$RID %in% four_yr_fu_ids),]
#BA_fu_ids <- filter(fu_avail, PLATFORM_ID == "BILEACID")$RID
#final <- fu_avail[which(fu_avail$RID %in% BA_fu_ids),]

#install.packages("Hmisc")
#install.packages("ADNIMERGE_0.0.1.tar.gz", repos = NULL, type = "source")
#install.packages(corrplot)

#library(Hmisc) 
library(ADNIMERGE) 
library(ggplot2) 
library(dplyr)
library(corrplot)

# Load Data 
adnimerge <- read.delim("ADNIMERGE.csv", sep = ',', stringsAsFactors = FALSE)
adni_dict <- read.delim("ADNIMERGE_DICT.csv", sep =',')

postproc <- read.delim("Biospecimen_Results/ADMC_BA_POSTPROC_06_28_18.csv", sep = ',', stringsAsFactors = FALSE)
postproc_dict <- read.delim("Biospecimen_Results/ADMC_BA_POSTPROC_06_28_18_DICT.csv", sep = ',', stringsAsFactors = FALSE)


# Initial number of participants in ADNIMERGE & Postproc:
count(distinct(adnimerge, RID)) #2231
count(distinct(postproc, RID)) #1671

group_by(adnimerge, RID) %>% summarise(n_distinct(RID))

z <- adnimerge[which(adnimerge$RID %in% postproc$RID),]
count(distinct(z,RID))
group_by(z, DX_bl) %>% summarise(n_distinct(RID))


# Discard irrelevant variables & merge LMCI and EMCI diagnosis to MCI 
postproc2 <- postproc[ , -c(5:9,34:57)] 
adnimerge2 <- adnimerge[ , c(1,3,8:11,15,19:21,60,103:105,109:112)] %>%
  mutate(DX_bl = ifelse(DX_bl == "LMCI"|DX_bl == "EMCI", 'MCI', DX_bl)) %>%
  arrange(RID,VISCODE)

#count(distinct(adnimerge2, RID))

# Remove patients with SMC and include only those with data avail for three years
#table(postproc2$SUBJECT_FLAG)
group_by(adnimerge2, DX_bl) %>% summarise(n_distinct(RID))    #292 
  
adnimerge2 <- adnimerge2 %>% 
  filter(DX_bl != 'SMC')  %>% 
  filter(Years_bl <= 3.2) %>% 
  group_by(RID) %>% 
  mutate(max_data = (max(M))) %>%
  #ifelse(DX_bl != 'AD', filter(max_data == 36), )
  filter(max_data == 36) %>%
  ungroup(RID)

#postproc2 <- filter(postproc2, SUBJECT_FLAG != 1)  #107
#postproc_dict %>% filter(FLDNAME == 'SUBJECT_FLAG') %>% select(TEXT) 
#postproc_dict %>% filter(FLDNAME == "TLCA_CDCA_LOGTRANSFORMFLAG") %>% select(TEXT) 

# Merge data & rename variables
total <- left_join(adnimerge2,postproc2, by = c("RID", "VISCODE")) %>% 
  rename("GENDER" = PTGENDER, "EDUCATION" = PTEDUCAT) %>% arrange(RID, VISCODE)

total$GENDER <-  as.factor(total$GENDER)

# Total number of participants in ADNIMERGE & BA datasets
count(distinct(total,RID)) #765
count(distinct(postproc2,RID)) #1671


# Only participants with BA info available 
final <- total[which(total$RID %in% postproc2$RID),]

#`%not in%` <- function (x, table) is.na(match(x, table, nomatch=NA_integer_))

# Final number of participants:
count(distinct(final,RID)) #738

# Distribution of Baseline Diagnosis:
bl_distribution <- final %>% 
  group_by(DX_bl) %>% 
  summarise(n_distinct(RID))
bl_distribution

# Distribution of Final Diagnosis: 
m36_distribution <- final %>% 
  filter(VISCODE == "m36") %>% 
  group_by(DX) %>% 
  summarise(n_distinct(RID))
m36_distribution

MCI_convert <- filter(final, DX == "Dementia" & DX_bl == "MCI")
count(distinct(MCI_convert,RID))

# Diagnosis of MCI patients at m36
MCI_patients <- final %>% filter(DX_bl == 'MCI') 
#count(distinct(MCI,RID))

MCI_DX <- MCI_patients %>% filter(VISCODE == "m36") %>% select(DX)
table(MCI_DX) #10 empty values 

# Remove NAs for now
#MCI_patients <- MCI_patients %>% group_by(RID) %>% filter(DX != "")

# Check if CSF measures are available at M36 
CSF_check <- final %>% filter(DX_bl == 'MCI') %>% filter(VISCODE== 'm36')

dim(CSF_check[CSF_check$ABETA == '',])[1]  #490 missing values


# Create Conversion Variable
MCI_patients <- group_by(MCI_patients, RID) %>% 
  mutate(Convert = ifelse(DX == "Dementia", 1, 0)) %>%
  ungroup()

MCI_patients$Convert <- as.factor(MCI_patients$Convert)

check <- select(MCI_patients, RID,VISCODE, DX, Convert)


# MCI Baseline subset
MCI_bl <- MCI_patients %>% 
  filter(VISCODE == "bl") %>%
  select(RID, VISCODE, AGE, GENDER, EDUCATION, APOE4, CA:Convert)




# Look at correlation of mean values of each feature # 
#MCI$GENDER <- as.factor(MCI$GENDER)

predictors <- filter(MCI_patients, VISCODE == 'bl') %>%
  select(AGE, GENDER, EDUCATION, APOE4,DX,CA:Convert)
  
predictors$DX <- as.factor(predictors$DX)
lapply(predictors[,c(2,5)],levels)

predictors[] <- lapply(predictors,as.integer)
cor(predictors)

correlations <- cor(predictors)
correlation_plot <- corrplot(correlations, method = "circle", 
                             title = "Variable Correlations", tl.col = "black", 
                             order = "FPC", line = -2)
correlation_plot


MCI_patients[MCI_patients$VISCODE== 'm36',]$DX == "Dementia"

MCI_patients[VISCODE ==  'bl']$Final_DX <- 'Dementia'




