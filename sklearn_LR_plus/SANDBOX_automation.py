'''
1. Create a simple linear regression for each column
    a. Hold these in some sort of sorted list or a queue
2. Take the model that has the least rss and add that column to multi LR model
3. Use sklearn_LR_plus.LR_metrics.get_coef_tests() to look at p values
4. IF any p-values are over q then remove that value from fields being used
5. Return to step 2
'''
import sklearn_LR_plus as LR_p

lr_ms = LR_p.MixedSelection(X, Y)

# A list of LrMetrics
simp_LRs = lr_ms.simple_linear_regressions()

features = list()

q = 0.5

# Iterate over simple linear regression adding each variable to the larger model and then testing it
for i in range(len(simp_LRs)):
    if simp_LRs[i].features > 1:
        raise RuntimeError('Linear Regressions had more than a single feature when they were supposed to be simple.')
    feature = simp_LRs[i].features[0]
    features.append(feature)

    temp_X = X[features]
    if i == 0:
        continue

    multi_reg = LR_p.LrMetrics(X, Y)
    for feat in multi_reg.get_high_p_features(q):
        features.remove(feat)










