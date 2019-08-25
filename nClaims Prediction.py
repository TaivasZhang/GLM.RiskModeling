#!/usr/bin/env python
# coding: utf-8

# # Predicting the Number of Claims
# 
# ## Data Perparation

# In[2]:


plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300


# In[3]:


inData = pandas.read_excel('Assignment6.xlsx', sheet_name = "HOCLAIMDATA", header = 0)
numPolicy = inData.shape[0]
numPolicy


# In[4]:


claimAmount = inData["amt_claims"]
nClaim = inData["num_claims"]
exposure = inData["exposure"]


# ## Calculate the Prameter for Poisson

# In[5]:


mm_lambda = inData.num_claims.mean()
print('Method of Moment Lambda is', mm_lambda)


# In[6]:


# Aggregate the data to produce the Frequency table
xCount = pandas.DataFrame.sort_index(pandas.value_counts(inData["num_claims"])).to_frame()
xCount["Count"] = xCount["num_claims"]
xCount["num_claims"] = xCount.index

sum_kx = numpy.inner(xCount["num_claims"], xCount["Count"])
sum_k = numpy.sum(xCount["Count"])
lamPoisson = (sum_kx / sum_k)
yPoisson = stats.poisson.pmf(xCount["num_claims"], lamPoisson)

print('Poisson MLE')
print('lambda = ', lamPoisson)


# ## Plot the Histogram and Comparison with Empirical Data

# In[7]:


# Plot the empirical histogram
yEmpirical = xCount["Count"] / sum_k
plt.bar(xCount["num_claims"], yEmpirical, align='center', edgecolor = 'black', fill = True, label = 'Empirical')
plt.plot(xCount["num_claims"], yPoisson, 'rs-', label = 'Poisson')
plt.title("Predicted versus Empirical")
plt.xlabel("Number of Claims")
plt.ylabel("Probability")
plt.xticks(range(10))
plt.grid(axis="both")
plt.legend(loc='upper right')
plt.savefig('histogram.png',  bbox_inches = 'tight')
plt.show()


# ## Forward Model Selection

# In[8]:


lnExposure = numpy.log(exposure)


# In[9]:


# Step 0: Build a GLM model with the Intercept term only to predict number of claim
X = numpy.full([numPolicy,1], 1.0)

poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
poissonGLM_results = poissonGLM.fit(method = 'newton')
print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
print(poissonGLM_results.summary())
d0 = poissonGLM_results.deviance


# ### Step 1. Adding the first variable

# In[10]:


# Step 1: Build a GLM model with one predictor to predict number of claim
predList = numpy.array(["aoi_tier", "marital", "primary_age_tier", "primary_gender", "residence_location"])

for predVar in predList:
   catPred = inData[[predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d0 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual, deviance) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid, ",", poissonGLM_results.deviance)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# ### The first selected variable is ```'residence_location'```
# 
# Now I call the result

# In[11]:


# Call the result
predList = numpy.array(["residence_location"])

for predVar in predList:
   catPred = inData[[predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d0 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual, deviance) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid, ",", poissonGLM_results.deviance)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# **Calculate the new base-deviance as d1**

# In[12]:


d1 = poissonGLM_results.deviance
d1


# ### Step 2 Adding one more variable

# In[13]:


# Step 2: Build a GLM model with "residence_location" and another predictor to predict number of claim
predList = numpy.array(["aoi_tier", "marital", "primary_age_tier", "primary_gender"])

for predVar in predList:
   catPred = inData[["residence_location", predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d1 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)   
   print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# ### Another variable  ```primary_age_tier``` should be added.
# 
# Now I call the result.

# In[14]:


# Call the result
predList = numpy.array(["primary_age_tier"])

for predVar in predList:
   catPred = inData[["residence_location", predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d1 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# **Calculate the new base-deviance as d2**

# In[15]:


d2 = poissonGLM_results.deviance
d2


# ### Step 3 Adding another Variable from ```"aoi_tier"```, ```"marital"```, and ```"primary_gender"```
# 

# In[16]:


# Step 3: Build a GLM model with "residence_location", "primary_age_tier" and another predictor to predict number of claim
predList = numpy.array(["aoi_tier", "marital", "primary_gender"])

for predVar in predList:
   catPred = inData[["residence_location", "primary_age_tier", predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d2 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# ### Variable ```primary_gender``` should also be incorporated.

# In[17]:


# Call the result
predList = numpy.array(["primary_gender"])

for predVar in predList:
   catPred = inData[["residence_location", "primary_age_tier", predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d2 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# In[18]:


d3 = poissonGLM_results.deviance
d3


# ### Step 4 Adding ```"aoi_tier"``` or ```"marital"```

# In[19]:


# Step 4: Build a GLM model with CAR_USE, HOMEKID, MSTATUS and another predictor to predict number of claim
predList = numpy.array(["aoi_tier", "marital"])

for predVar in predList:
   catPred = inData[["residence_location", "primary_age_tier", "primary_gender", predVar]].astype('category')
   X = pandas.get_dummies(catPred)
   X = sm.add_constant(X, prepend = True)

   poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
   poissonGLM_results = poissonGLM.fit(method = 'nm', maxiter = 500, ftol = 1e-7)
   Deviance = d3 - poissonGLM_results.deviance
   Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
   print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
   print(poissonGLM_results.summary())
   print("Deviance of the model = ", Deviance)
   print("Chi Square Sig =", Chi2Sig)


# ### Chi Square shows neither of the two variables makes significant difference if considered adding.
# ### We end up with a final model that contains 
# ### ```"residence_location"```, ```"primary_age_tier" ```& ```"primary_gender"```

# In[20]:


# Final Model
catPred = inData[["residence_location", "primary_age_tier", "primary_gender"]].astype('category')
X = pandas.get_dummies(catPred)
X = sm.add_constant(X, prepend = True)

poissonGLM = sm.GLM(nClaim, X, family = sm.families.Poisson(link = sm.families.links.log), offset = lnExposure)
poissonGLM_results = poissonGLM.fit(method = 'ncg', maxiter = 500)
modelParam = poissonGLM_results.params
Deviance = d0 - poissonGLM_results.deviance
Chi2Sig = stats.chi2.sf(Deviance, poissonGLM_results.df_model)
print("Log-likelihood (Value, dfModel, dfResidual) = ", poissonGLM_results.llf, ", ", poissonGLM_results.df_model, ", ", poissonGLM_results.df_resid)
print(poissonGLM_results.summary())
print("Deviance of the model = ", Deviance)
print("Chi Square Sig =", Chi2Sig)


# In[21]:


# Calculate predicted values
pred_nClaim = poissonGLM.predict(modelParam, X, offset = lnExposure)

plt.hist(pred_nClaim, bins = range(10), label = 'Predicted', density = True, align = 'left')
plt.bar(xCount["num_claims"], yEmpirical, align='center', edgecolor = 'black', fill = False, label = 'Empirical')
plt.title("Histogram")
plt.xlabel("Number of Claims")
plt.ylabel("Probability")
plt.xticks(range(10))
plt.grid(axis="both")
plt.legend(loc='upper right')
plt.show()

plt.scatter(x=nClaim, y=pred_nClaim)
plt.title("Predicted versus Observed")
plt.xlabel("Observed Number of Claims")
plt.ylabel("Predicted Number of Claims")
plt.xticks(range(10))
plt.yticks(range(10))
plt.grid(axis="both")
plt.show()


# ### End of number of Claims Prediction
