from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from config import *
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def to_numeric(s): return pd.to_numeric(s, errors='coerce')
def zscore(s):
    s=to_numeric(s); sd=s.std(ddof=0)
    return (s-s.mean())/sd if sd and not np.isnan(sd) else s*0
def robust_minmax(s):
    s=to_numeric(s); lo,hi=s.quantile(.01),s.quantile(.99); s=s.clip(lo,hi)
    return (s-lo)/(hi-lo) if hi>lo else s*0
def save_table(df,name): OUTPUT_DIR.mkdir(exist_ok=True); df.to_csv(OUTPUT_DIR/name,index=False)
def save_json(obj,name): OUTPUT_DIR.mkdir(exist_ok=True); (OUTPUT_DIR/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def read_required(path):
    if not path.exists(): raise FileNotFoundError(f'Missing required file: {path}')
    return pd.read_csv(path, low_memory=False)
def ols(data,y,xs,name):
    d=data[[y]+xs].replace([np.inf,-np.inf],np.nan).dropna(); X=sm.add_constant(d[xs]); model=sm.OLS(d[y],X).fit()
    tab=pd.DataFrame({'Predictor':model.params.index,'Beta':model.params.values,'Std_Error':model.bse.values,'t_value':model.tvalues.values,'p_value':model.pvalues.values,'CI_2.5':model.conf_int()[0].values,'CI_97.5':model.conf_int()[1].values})
    save_table(tab,name); return model

def load_data():
    pr=read_required(PR_FILE); comments=read_required(COMMENTS_FILE); milestones=read_required(MILESTONE_FILE)
    issues=pd.read_csv(ISSUES_FILE,low_memory=False) if ISSUES_FILE.exists() else None
    overview=pd.DataFrame([{'dataset':'pr_data','rows':len(pr),'columns':pr.shape[1]},{'dataset':'comments_with_sentiment','rows':len(comments),'columns':comments.shape[1]},{'dataset':'milestone_dataset','rows':len(milestones),'columns':milestones.shape[1]},{'dataset':'issues','rows':len(issues) if issues is not None else np.nan,'columns':issues.shape[1] if issues is not None else np.nan}])
    save_table(overview,'Table_Dataset_Overview.csv')
    return pr,comments,milestones,issues

def build_constructs(pr):
    df=pr.copy()
    df['risk_negativity']=-zscore(df['review_sentiment_avg'])
    df['risk_uncertainty']=zscore(df['uncertainty_score'])
    df['risk_disagreement']=zscore(df['reviewer_disagreement_level'])
    df['risk_escalation']=zscore(df['comment_escalation'])
    df['risk_low_politeness']=-zscore(df['politeness_score'])
    risk=['risk_negativity','risk_uncertainty','risk_disagreement','risk_escalation','risk_low_politeness']
    df['CRI']=df[risk].mean(axis=1)
    weights={'risk_escalation':.30,'risk_disagreement':.25,'risk_uncertainty':.20,'risk_negativity':.15,'risk_low_politeness':.10}
    df['CRI_weighted']=sum(df[k]*v for k,v in weights.items())
    df['SVI']=zscore(df['review_sentiment_std'])
    ces=[]
    for raw,new,sign in [('num_approvals','ces_approvals',1),('code_owner_involvement','ces_owner',1),('review_wait_time','ces_review_wait',-1),('response_time_avg','ces_response',-1),('merge_delay_days','ces_merge',-1)]:
        if raw in df.columns: df[new]=sign*zscore(df[raw]); ces.append(new)
    df['CES']=df[ces].mean(axis=1) if ces else np.nan
    health=[c for c in ['future_bug_fixes','requires_refactoring','causes_conflicts','requires_hotfix'] if c in df.columns]
    for c in health: df[c]=to_numeric(df[c]).fillna(0)
    df['PHI']=df[health].apply(lambda col:zscore(col),axis=0).mean(axis=1)
    phi_w={'future_bug_fixes':.30,'requires_refactoring':.20,'causes_conflicts':.25,'requires_hotfix':.25}
    df['PHI_weighted']=sum(zscore(df[c])*w for c,w in phi_w.items() if c in df.columns)
    df['PHI_high']=(df['PHI']>=df['PHI'].quantile(PHI_HIGH_QUANTILE)).astype(int)
    df['CRI_x_SVI']=df['CRI']*df['SVI']; df['CRI_weighted_x_SVI']=df['CRI_weighted']*df['SVI']
    save_table(df[['CRI','CRI_weighted','SVI','CES','PHI','PHI_weighted','PHI_high']].describe().T.reset_index(),'Table_Construct_Descriptives.csv')
    return df

def descriptive(df,comments,milestones):
    comments=comments.copy(); comments['sentiment_score']=to_numeric(comments['sentiment_score']); sent=comments['sentiment_score'].dropna()
    save_table(pd.DataFrame([{'mean':sent.mean(),'median':sent.median(),'std':sent.std(),'min':sent.min(),'max':sent.max(),'n':len(sent)}]),'Table_Sentiment_Statistics.csv')
    comp=pd.DataFrame([{'component':'Negativity','score':1-robust_minmax(df['review_sentiment_avg']).mean()},{'component':'Uncertainty','score':robust_minmax(df['uncertainty_score']).mean()},{'component':'Disagreement','score':robust_minmax(df['reviewer_disagreement_level']).mean()},{'component':'Escalation','score':robust_minmax(df['comment_escalation']).mean()},{'component':'Low Politeness','score':1-robust_minmax(df['politeness_score']).mean()}]).sort_values('score',ascending=False)
    save_table(comp,'Table_Communication_Risk_Components.csv'); save_table(milestones.describe(include='all').T.reset_index(),'Table_Milestone_Descriptives.csv')
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10,6)); plt.hist(sent,bins=50,edgecolor='black',linewidth=.4); plt.axvline(sent.mean(),linestyle='--',label=f'Mean={sent.mean():.3f}'); plt.axvline(sent.median(),linestyle=':',label=f'Median={sent.median():.3f}'); plt.xlabel('Sentiment score'); plt.ylabel('Number of comments'); plt.title('Distribution of Developer Sentiment'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Sentiment_Distribution.png',dpi=600); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(comp['component'],comp['score']); plt.ylabel('Normalized score'); plt.title('Communication-Risk Components'); plt.xticks(rotation=15,ha='right'); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Communication_Risk_Components.png',dpi=600); plt.close()

def validation(df):
    cols=['risk_negativity','risk_uncertainty','risk_disagreement','risk_escalation','risk_low_politeness']
    save_table(df[cols+['CRI','CRI_weighted','SVI','CES','PHI','PHI_weighted']].corr().reset_index(),'Table_Construct_Correlation_Matrix.csv')
    # VIF
    rows=[]
    for c in cols:
        other=[x for x in cols if x!=c]; m=ols(df[[c]+other].dropna(),c,other,f'VIF_helper_{c}.csv'); rows.append({'variable':c,'VIF':1/(1-m.rsquared) if m.rsquared<1 else np.inf})
    save_table(pd.DataFrame(rows),'Table_VIF_Communication_Risk.csv')
    data=df[cols].replace([np.inf,-np.inf],np.nan).dropna(); pca=PCA().fit(StandardScaler().fit_transform(data))
    save_table(pd.DataFrame({'component':[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],'explained_variance_ratio':pca.explained_variance_ratio_,'cumulative_variance':pca.explained_variance_ratio_.cumsum()}),'Table_PCA_Communication_Risk.csv')

def regressions(df):
    tech=[c for c in ['lines_added','lines_deleted','files_changed','commits_count','comment_count','review_comment_count','participants_count','has_tests','code_churn','test_coverage_change','cyclomatic_avg','author_experience','is_core_contributor','author_followers','merge_delay_days','num_TODO_FIXME','distinct_langs_changed'] if c in df.columns]
    mdf=df[tech+['CRI','CRI_weighted','SVI','CRI_x_SVI','CRI_weighted_x_SVI','PHI','PHI_weighted','PHI_high','CES']].replace([np.inf,-np.inf],np.nan).dropna()
    p2=ols(mdf,'CES',['CRI'],'Table_P2_CES_on_CRI.csv'); p2w=ols(mdf,'CES',['CRI_weighted'],'Table_P2_CES_on_CRI_weighted.csv'); t1=ols(mdf,'PHI',['CRI'],'Table_T1_PHI_on_CRI.csv'); t1w=ols(mdf,'PHI_weighted',['CRI_weighted'],'Table_T1_PHI_weighted_on_CRI_weighted.csv'); svi=ols(mdf,'PHI',['CRI','SVI'],'Table_PHI_on_CRI_SVI.csv'); t3=ols(mdf,'PHI',['CRI','SVI','CRI_x_SVI'],'Table_T3_Volatility_Interaction.csv'); tm=ols(mdf,'PHI',tech,'Table_Technical_Only_OLS.csv'); em=ols(mdf,'PHI',tech+['CRI','SVI'],'Table_Technical_Plus_SRDT_OLS.csv'); ewm=ols(mdf,'PHI_weighted',tech+['CRI_weighted','SVI'],'Table_Technical_Plus_SRDT_Weighted_OLS.csv')
    summary={'n_modeling':int(len(mdf)),'P2_CES_on_CRI':{'beta':float(p2.params.get('CRI',np.nan)),'p':float(p2.pvalues.get('CRI',np.nan)),'r2':float(p2.rsquared)},'P2_CES_on_CRI_weighted':{'beta':float(p2w.params.get('CRI_weighted',np.nan)),'p':float(p2w.pvalues.get('CRI_weighted',np.nan)),'r2':float(p2w.rsquared)},'T1_PHI_on_CRI':{'beta':float(t1.params.get('CRI',np.nan)),'p':float(t1.pvalues.get('CRI',np.nan)),'r2':float(t1.rsquared)},'T1_PHI_weighted_on_CRI_weighted':{'beta':float(t1w.params.get('CRI_weighted',np.nan)),'p':float(t1w.pvalues.get('CRI_weighted',np.nan)),'r2':float(t1w.rsquared)},'T3_interaction':{'beta':float(t3.params.get('CRI_x_SVI',np.nan)),'p':float(t3.pvalues.get('CRI_x_SVI',np.nan)),'r2':float(t3.rsquared)},'technical_only_r2':float(tm.rsquared),'technical_plus_srdt_r2':float(em.rsquared),'delta_r2':float(em.rsquared-tm.rsquared),'technical_plus_srdt_weighted_r2':float(ewm.rsquared)}
    save_json(summary,'Regression_Model_Summary.json'); return mdf,tech

def ml(mdf,tech):
    feature_sets={'Technical only':tech,'Technical + SRDT':tech+['CRI','SVI'],'Technical + weighted SRDT':tech+['CRI_weighted','SVI']}; y=mdf['PHI_high'].astype(int)
    models={'Logistic Regression':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000,class_weight='balanced'))]),'Random Forest':RandomForestClassifier(n_estimators=200,max_depth=10,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced'),'Gradient Boosting':GradientBoostingClassifier(random_state=RANDOM_STATE)}
    if HAS_XGB: models['XGBoost']=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=RANDOM_STATE)
    rows=[]; fitted={}; tr,te=train_test_split(np.arange(len(y)),test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    for fs,cols in feature_sets.items():
        X=mdf[cols].apply(pd.to_numeric,errors='coerce').fillna(0)
        for name,model in models.items():
            model.fit(X.iloc[tr],y.iloc[tr]); pred=model.predict(X.iloc[te]); prob=model.predict_proba(X.iloc[te])[:,1]
            rows.append({'Feature_Set':fs,'Model':name,'Evaluation':'Holdout','AUC':roc_auc_score(y.iloc[te],prob),'Accuracy':accuracy_score(y.iloc[te],pred),'F1':f1_score(y.iloc[te],pred),'Precision':precision_score(y.iloc[te],pred,zero_division=0),'Recall':recall_score(y.iloc[te],pred)}); fitted[(fs,name)]=(model,cols,X.iloc[te],y.iloc[te])
    cv=StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    for fs,cols in feature_sets.items():
        X=mdf[cols].apply(pd.to_numeric,errors='coerce').fillna(0)
        for name,model in models.items():
            fold=[]
            for train,test in cv.split(X,y):
                model.fit(X.iloc[train],y.iloc[train]); pred=model.predict(X.iloc[test]); prob=model.predict_proba(X.iloc[test])[:,1]
                fold.append({'AUC':roc_auc_score(y.iloc[test],prob),'Accuracy':accuracy_score(y.iloc[test],pred),'F1':f1_score(y.iloc[test],pred),'Precision':precision_score(y.iloc[test],pred,zero_division=0),'Recall':recall_score(y.iloc[test],pred)})
            avg=pd.DataFrame(fold).mean().to_dict(); avg.update({'Feature_Set':fs,'Model':name,'Evaluation':'CrossValidation'}); rows.append(avg)
    results=pd.DataFrame(rows); save_table(results,'Table_ML_Model_Comparison.csv'); return results,fitted

def explain(fitted):
    pref=None
    for key in [('Technical + weighted SRDT','XGBoost'),('Technical + SRDT','XGBoost'),('Technical + weighted SRDT','Random Forest'),('Technical + SRDT','Random Forest')]:
        if key in fitted: pref=key; break
    if pref is None: pref=list(fitted.keys())[0]
    model,cols,Xtest,ytest=fitted[pref]
    imp=None
    if hasattr(model,'feature_importances_'):
        imp=pd.DataFrame({'Feature':cols,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False); save_table(imp,'Table_Model_Feature_Importance.csv')
    perm=permutation_importance(model,Xtest,ytest,n_repeats=10,random_state=RANDOM_STATE,scoring='roc_auc',n_jobs=-1)
    ptab=pd.DataFrame({'Feature':cols,'Permutation_Importance_Mean':perm.importances_mean,'Permutation_Importance_SD':perm.importances_std}).sort_values('Permutation_Importance_Mean',ascending=False); save_table(ptab,'Table_Permutation_Importance.csv')
    try:
        import shap; explainer=shap.Explainer(model,Xtest); vals=explainer(Xtest); stab=pd.DataFrame({'Feature':cols,'Mean_Abs_SHAP':abs(vals.values).mean(axis=0)}).sort_values('Mean_Abs_SHAP',ascending=False); save_table(stab,'Table_SHAP_Importance.csv')
    except Exception as e:
        (OUTPUT_DIR/'SHAP_not_run.txt').write_text(str(e),encoding='utf-8'); stab=None
    return imp,ptab,stab

def figures(comments,df,ml_results,imp,ptab,stab):
    comments=comments.copy(); comments['sentiment_score']=to_numeric(comments['sentiment_score']); sent=comments['sentiment_score'].dropna(); FIGURE_DIR.mkdir(exist_ok=True)
    if 'issue_number' in comments.columns:
        g=comments.groupby('issue_number')['sentiment_score'].agg(['std','count']).dropna(); g=g[g['count']>=20]
        if len(g)>1:
            low,high=g['std'].idxmin(),g['std'].idxmax(); l=comments[comments['issue_number']==low]['sentiment_score'].reset_index(drop=True).head(50); h=comments[comments['issue_number']==high]['sentiment_score'].reset_index(drop=True).head(50)
            plt.figure(figsize=(8,5)); plt.plot(l.values,label=f'Low volatility issue {low}'); plt.plot(h.values,label=f'High volatility issue {high}'); plt.xlabel('Comment sequence'); plt.ylabel('Sentiment score'); plt.title('Actual Sentiment Trajectories'); plt.legend(); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Sentiment_Trajectories.png',dpi=600); plt.close()
    tmp=df[['CRI','PHI']].dropna(); tmp['CRI_decile']=pd.qcut(tmp['CRI'],10,duplicates='drop'); b=tmp.groupby('CRI_decile').agg(CRI_mean=('CRI','mean'),PHI_mean=('PHI','mean'),PHI_se=('PHI',lambda x:x.std()/(len(x)**.5))).reset_index(); save_table(b,'Table_CRI_PHI_Decile_Profile.csv')
    plt.figure(figsize=(8,5)); plt.errorbar(b['CRI_mean'],b['PHI_mean'],yerr=1.96*b['PHI_se'],marker='o'); plt.xlabel('Communication Risk Index'); plt.ylabel('Project Health Degradation Index'); plt.title('Risk Accumulation Pattern'); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Risk_Accumulation.png',dpi=600); plt.close()
    source=stab if stab is not None else (imp if imp is not None else ptab); x='Mean_Abs_SHAP' if stab is not None else ('Importance' if imp is not None else 'Permutation_Importance_Mean'); top=source.head(12).sort_values(x,ascending=True)
    plt.figure(figsize=(9,6)); plt.barh(top['Feature'],top[x]); plt.xlabel(x.replace('_',' ')); plt.title('Feature Importance from SRDT-Enhanced Model'); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Feature_Importance.png',dpi=600); plt.close()
    subset=ml_results[ml_results['Evaluation']=='Holdout']; pivot=subset.pivot_table(index='Model',columns='Feature_Set',values='AUC')
    plt.figure(figsize=(9,5)); pivot.plot(kind='bar',ax=plt.gca()); plt.ylabel('AUC'); plt.title('Predictive Performance Comparison'); plt.xticks(rotation=15,ha='right'); plt.tight_layout(); plt.savefig(FIGURE_DIR/'Figure_Model_Comparison_AUC.png',dpi=600); plt.close()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True); FIGURE_DIR.mkdir(exist_ok=True)
    pr,comments,milestones,issues=load_data(); df=build_constructs(pr); descriptive(df,comments,milestones); validation(df); mdf,tech=regressions(df); ml_results,fitted=ml(mdf,tech); imp,ptab,stab=explain(fitted); figures(comments,df,ml_results,imp,ptab,stab)
    print('SRDT reproducibility pipeline completed. See outputs/ and figures/.')
