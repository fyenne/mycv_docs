---
Author: Siming yan
Title: credit fraud detection series
Reference: python金融大数据风控建模实战; kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud/

---

# credit fraud detection

## the main application:

### 智能风控和评分卡:

+ 借由信用信息, 评估借款人的偿还能力和意愿, 量化违约风险.
+ 申请评分卡, 行为评分卡, 催收评分卡. (反欺诈模型, 营销评分卡, 客户流失评分卡)

WOE IV



问题和挑战:

1. data skewness, 假设过去的客户大量为优质客户, 鲜有违约客户, 数据的采样方式如何解决. 

   + From imblearn.over_sampling import SMOTE (随机连线), boarderline smote, boostrap sampling, nearmiss

   + From imblearn.under_sampling import randomundersampler , enn

2. pca feature selection & other feature mixed up