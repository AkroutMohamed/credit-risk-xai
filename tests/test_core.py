import numpy as np
import pandas as pd
import pytest
from src.data import normalize, audit
from src.ml import split_data, fit_models, metrics, compare, MODELS
from src.artifacts import save_bundle, load_bundle

@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    n = 180
    debt = rng.normal(size=n)
    delay = rng.integers(0, 4, size=n)
    p = 1/(1+np.exp(-(-1 + debt + delay/2)))
    return pd.DataFrame({"debt":debt,"late_payments":delay,"segment":rng.choice(["A","B"],n),
                         "default":rng.binomial(1,p)})

def test_normalize_binary():
    out = normalize(pd.DataFrame({"ID":[1,2,3,4],"label":["yes","no","yes","no"],"x":[0,1,2,3]}),"label")
    assert set(out.default) == {0,1}
    assert "ID" not in out
    assert audit(out).loc["x","missing"] == 0

def test_split_models_metrics(toy):
    split = split_data(toy,["debt","late_payments","segment"],test_size=0.25)
    assert not set(split.x_train.index) & set(split.x_test.index)
    model = fit_models(split,["Logistic Regression"])["Logistic Regression"]
    probs = model.predict_proba(split.x_test)[:,1]
    m = metrics(split.y_test,probs,0.4)
    assert 0 <= m["ROC-AUC"] <= 1
    assert m["TP"]+m["TN"]+m["FP"]+m["FN"] == len(split.y_test)
    assert m["Gini"] == pytest.approx(2*m["ROC-AUC"]-1)
    assert len(compare({"LR":model},split)) == 1

def test_all_models_fit(toy):
    split = split_data(toy,["debt","late_payments","segment"])
    for model in fit_models(split,MODELS).values():
        assert model.predict_proba(split.x_test).shape == (len(split.x_test),2)

def test_bundle(toy,tmp_path):
    split=split_data(toy,["debt","late_payments","segment"])
    model=fit_models(split,["Decision Tree"])["Decision Tree"]
    path=save_bundle(model,split.x_train,tmp_path/"bundle.joblib")
    assert load_bundle(path)["model"].predict_proba(split.x_test).shape[1] == 2
